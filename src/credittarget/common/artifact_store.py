"""Dual-mode local/S3 artifact storage with atomic JSON/text operations and SHA-256 hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from credittarget.config import StorageConfig
from credittarget.exceptions import CreditTargetError


def sha256_text(text: str) -> str:
    """Computes SHA-256 hex digest of a UTF-8 text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dict(payload: Any) -> str:
    """Computes deterministic canonical SHA-256 hex digest of a JSON-serializable structure."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ArtifactStore:
    """Handles persistence of evaluation reports, plans, metrics, and lineage artifacts."""

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.is_s3 = config.backend == "s3"
        self._s3_client: Any = None
        if self.is_s3:
            import boto3

            self._s3_client = boto3.client("s3")

    def _resolve_local_path(self, rel_path: str) -> Path:
        base = Path(self.config.base_dir)
        full_path = (base / rel_path).resolve()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def _s3_bucket_and_key(self, rel_path: str) -> tuple[str, str]:
        bucket = self.config.s3_bucket
        prefix = self.config.s3_prefix.strip("/")
        clean_rel = rel_path.strip("/")
        key = f"{prefix}/{clean_rel}" if prefix else clean_rel
        return bucket, key

    def write_text(self, rel_path: str, content: str) -> str:
        """Atomically writes text to local filesystem or S3 and returns the storage URI."""
        if self.is_s3:
            bucket, key = self._s3_bucket_and_key(rel_path)
            self._s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )
            return f"s3://{bucket}/{key}"

        dest_path = self._resolve_local_path(rel_path)
        # Atomic local file write
        temp_dir = dest_path.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            tf.write(content)
            temp_name = tf.name
        os.replace(temp_name, dest_path)
        return str(dest_path)

    def read_text(self, rel_path: str) -> str:
        """Reads text content from local filesystem or S3."""
        if self.is_s3:
            bucket, key = self._s3_bucket_and_key(rel_path)
            try:
                response = self._s3_client.get_object(Bucket=bucket, Key=key)
                return response["Body"].read().decode("utf-8")
            except Exception as e:
                raise CreditTargetError(f"Failed to read s3://{bucket}/{key}: {e}") from e

        dest_path = self._resolve_local_path(rel_path)
        if not dest_path.is_file():
            raise CreditTargetError(f"Artifact not found: {dest_path}")
        return dest_path.read_text(encoding="utf-8")

    def write_json(self, rel_path: str, payload: Any) -> str:
        """Serializes and writes payload as canonical JSON."""
        json_str = json.dumps(payload, indent=2, sort_keys=True, default=str)
        return self.write_text(rel_path, json_str)

    def read_json(self, rel_path: str) -> Any:
        """Reads and deserializes JSON artifact."""
        raw = self.read_text(rel_path)
        return json.loads(raw)

    def exists(self, rel_path: str) -> bool:
        """Checks if artifact exists in local store or S3."""
        if self.is_s3:
            bucket, key = self._s3_bucket_and_key(rel_path)
            try:
                self._s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except Exception:
                return False

        dest_path = Path(self.config.base_dir) / rel_path
        return dest_path.is_file()
