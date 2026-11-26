"""Vercel serverless function entrypoint for CreditTarget FastAPI."""

from __future__ import annotations

import os
import sys

# Ensure src directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ensure models directory is located relative to root
os.chdir(root_dir)

from credittarget.serving.api import app, handler  # noqa: E402

__all__ = ["app", "handler"]
