"""Model training pipeline: Logistic baseline, LightGBM, Platt calibration, and MLflow logging."""

from __future__ import annotations

import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credittarget.common.artifact_store import ArtifactStore, sha256_dict
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig
from credittarget.logging_utils import get_logger
from credittarget.modeling.calibration import PlattCalibratedClassifier
from credittarget.modeling.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)
from credittarget.modeling.policy import TargetingPolicyEngine
from credittarget.modeling.splits import get_temporal_splits
from credittarget.schemas.contracts import compute_schema_hash

logger = get_logger("ModelingTraining")


def get_git_sha() -> str:
    """Attempts to retrieve current Git SHA, defaulting safely to 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def train_single_model(
    model_type: str,  # "propensity" or "pd"
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    config: AppConfig,
    mlflow_client: ModelRegistryClient,
    lineage_tags: dict[str, Any],
) -> tuple[Any, Any, dict[str, float]]:
    """Trains baseline Logistic Regression and Champion LightGBM with Platt calibration."""
    logger.info("training_model_started", model_type=model_type, target_col=target_col)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = build_preprocessor()

    X_train_raw = train_df[feature_cols]
    y_train = np.ascontiguousarray(train_df[target_col].values, dtype=np.int32)

    X_val_raw = val_df[feature_cols]
    y_val = np.ascontiguousarray(val_df[target_col].values, dtype=np.int32)

    X_oot_raw = oot_df[feature_cols]
    y_oot = np.ascontiguousarray(oot_df[target_col].values, dtype=np.int32)

    # Fit preprocessor strictly on train
    preprocessor.fit(X_train_raw)
    X_train = np.ascontiguousarray(preprocessor.transform(X_train_raw))
    X_val = np.ascontiguousarray(preprocessor.transform(X_val_raw))

    # 1. Baseline Logistic Regression
    baseline = LogisticRegression(max_iter=1000, random_state=config.seed)
    baseline.fit(X_train, y_train)

    # 2. Champion Candidate LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=config.seed,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    # 3. Platt Calibration on Month 9 Validation set
    calibrated = PlattCalibratedClassifier(base_estimator=lgbm)
    calibrated.fit(X_val, y_val)

    # Complete pipeline bundle (preprocessor + calibrated classifier)
    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", calibrated),
        ]
    )

    # Evaluate predictions
    val_preds = full_pipeline.predict_proba(X_val_raw)[:, 1]
    oot_preds = full_pipeline.predict_proba(X_oot_raw)[:, 1]

    from sklearn.metrics import brier_score_loss, roc_auc_score

    val_auc = float(roc_auc_score(y_val, val_preds))
    oot_auc = float(roc_auc_score(y_oot, oot_preds))
    oot_brier = float(brier_score_loss(y_oot, oot_preds))
    oot_gini = float(2.0 * oot_auc - 1.0)

    metrics = {
        "val_roc_auc": val_auc,
        "oot_roc_auc": oot_auc,
        "oot_gini": oot_gini,
        "oot_brier": oot_brier,
    }

    # Register in MLflow Model Registry with alias @challenger
    reg_name = (
        config.mlflow.propensity_model_name
        if model_type == "propensity"
        else config.mlflow.pd_model_name
    )

    with mlflow_client.start_run(run_name=f"train_{model_type}"):
        mlflow_client.log_lineage_tags(
            git_sha=lineage_tags["git_sha"],
            gold_delta_version=lineage_tags["gold_delta_version"],
            feature_schema_hash=lineage_tags["feature_schema_hash"],
            config_hash=lineage_tags["config_hash"],
            additional_tags={"model_type": model_type},
        )
        import mlflow

        mlflow.log_params(
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "calibrator": "PlattCalibratedClassifier",
            }
        )
        mlflow.log_metrics(metrics)

        mlflow_client.log_model_and_register(
            model=full_pipeline,
            artifact_path=f"model_{model_type}",
            registered_model_name=reg_name,
            initial_alias="challenger",
        )

    logger.info("training_model_completed", model_type=model_type, metrics=metrics)
    return full_pipeline, baseline, metrics


def _train_worker(
    input_parquet: str,
    output_parquet: str,
    metrics_json: str,
    config_json: str,
    lineage_json: str,
) -> None:
    """Isolated child worker process for LightGBM/Platt fitting without JVM collision."""
    import json

    with open(config_json, encoding="utf-8") as f:
        config = AppConfig.model_validate(json.load(f))
    with open(lineage_json, encoding="utf-8") as f:
        lineage_tags = json.load(f)

    artifact_store = ArtifactStore(config.storage)
    mlflow_client = ModelRegistryClient(config.mlflow)
    pdf = pd.read_parquet(input_parquet)
    train_df, val_df, oot_df = get_temporal_splits(pdf, config)

    # 1. Train Propensity Model
    prop_model, prop_base, prop_metrics = train_single_model(
        model_type="propensity",
        target_col="accepted_offer_30d",
        train_df=train_df,
        val_df=val_df,
        oot_df=oot_df,
        config=config,
        mlflow_client=mlflow_client,
        lineage_tags=lineage_tags,
    )

    # 2. Train Probability of Default (PD) Model
    pd_model, pd_base, pd_metrics = train_single_model(
        model_type="pd",
        target_col="defaulted_12m",
        train_df=train_df,
        val_df=val_df,
        oot_df=oot_df,
        config=config,
        mlflow_client=mlflow_client,
        lineage_tags=lineage_tags,
    )

    # 3. Generate Evaluation Predictions on Val + OOT
    eval_df = pd.concat([val_df, oot_df], ignore_index=True)
    eval_df["propensity_score"] = prop_model.predict_proba(
        eval_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    eval_df["pd_score"] = pd_model.predict_proba(eval_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[
        :, 1
    ]

    # Evaluate targeting policy and naive comparison
    policy_engine = TargetingPolicyEngine(config)
    scored_eval_df = policy_engine.evaluate_policy_pandas(eval_df)
    policy_engine.compute_policy_uplift_comparison(scored_eval_df, artifact_store=artifact_store)

    scored_eval_df.to_parquet(output_parquet, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "propensity_metrics": prop_metrics,
                "pd_metrics": pd_metrics,
            },
            f,
        )


def train_all_models(
    spark: SparkSession,
    config: AppConfig,
) -> dict[str, Any]:
    """Orchestrates end-to-end model training, MLflow registration, and validation table generation."""
    import json
    import os
    import sys
    import tempfile

    delta_mgr = DeltaTableManager(spark)
    mlflow_client = ModelRegistryClient(config.mlflow)

    gold_path = f"{config.storage.delta_dir}/gold_customer_features"
    gold_df = delta_mgr.read_table(gold_path)
    gold_version = delta_mgr.get_latest_version(gold_path)
    schema_hash = compute_schema_hash(gold_df.schema)
    config_hash = sha256_dict(config.model_dump())
    git_sha = get_git_sha()

    lineage_tags = {
        "git_sha": git_sha,
        "gold_delta_version": gold_version,
        "feature_schema_hash": schema_hash,
        "config_hash": config_hash,
    }

    pdf = pd.DataFrame(gold_df.toPandas())

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_parquet = os.path.join(tmp_dir, "gold.parquet")
        output_parquet = os.path.join(tmp_dir, "eval_preds.parquet")
        metrics_json = os.path.join(tmp_dir, "metrics.json")
        config_json = os.path.join(tmp_dir, "config.json")
        lineage_json = os.path.join(tmp_dir, "lineage.json")

        pdf.to_parquet(input_parquet, index=False)
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f)
        with open(lineage_json, "w", encoding="utf-8") as f:
            json.dump(lineage_tags, f)

        cmd = [
            sys.executable,
            "-m",
            "credittarget.modeling.train",
            "--input-parquet",
            input_parquet,
            "--output-parquet",
            output_parquet,
            "--metrics-json",
            metrics_json,
            "--config-json",
            config_json,
            "--lineage-json",
            lineage_json,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("isolated_training_failed", stdout=res.stdout, stderr=res.stderr)
            raise RuntimeError(
                f"Isolated model training failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        with open(metrics_json, encoding="utf-8") as f:
            trained_res = json.load(f)

        scored_eval_df = pd.read_parquet(output_parquet)
        spark_eval_df = spark.createDataFrame(scored_eval_df)
        pred_path = f"{config.storage.delta_dir}/model_validation_predictions"
        delta_mgr.write_overwrite_table(spark_eval_df, pred_path, partition_cols=["snapshot_month"])

        prop_model = mlflow_client.load_model_by_alias(
            config.mlflow.propensity_model_name, "challenger"
        )
        pd_model = mlflow_client.load_model_by_alias(config.mlflow.pd_model_name, "challenger")

        return {
            "propensity_model": prop_model,
            "pd_model": pd_model,
            "propensity_metrics": trained_res["propensity_metrics"],
            "pd_metrics": trained_res["pd_metrics"],
            "lineage": lineage_tags,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--lineage-json", required=True)
    args = parser.parse_args()

    _train_worker(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        metrics_json=args.metrics_json,
        config_json=args.config_json,
        lineage_json=args.lineage_json,
    )
