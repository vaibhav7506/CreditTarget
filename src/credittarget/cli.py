"""Command line interface (Typer) for CreditTarget platform operations."""

from __future__ import annotations

import sys

import typer
import uvicorn

from credittarget.common.artifact_store import ArtifactStore
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import load_config
from credittarget.data_gen.synthetic_bank import generate_synthetic_bank_data
from credittarget.governance.lineage import build_lineage_manifest
from credittarget.governance.reporter import ModelRiskValidationReporter
from credittarget.logging_utils import configure_logging, get_logger
from credittarget.modeling.train import train_all_models
from credittarget.pipelines.bronze import BronzeStage
from credittarget.pipelines.gold import GoldStage
from credittarget.pipelines.silver import SilverStage
from credittarget.serving.batch_score import BatchScoreStage
from credittarget.spark_session import get_spark_session

app = typer.Typer(
    name="credittarget",
    help="CreditTarget: Distributed Banking Customer Targeting & Model Risk Lifecycle Platform",
    no_args_is_help=True,
)
logger = get_logger("CLI")


@app.command()
def generate(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to YAML configuration profile."
    ),
) -> None:
    """Generate synthetic 5-table relational banking data."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    typer.echo(
        f"Generating synthetic bank data for {cfg.datagen.num_customers} customers across 12 snapshots..."
    )
    tables = generate_synthetic_bank_data(spark, cfg)
    for name, df in tables.items():
        typer.echo(f" - {name}: {df.count()} rows")
    typer.echo("Data generation complete.")


@app.command()
def bronze(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Run Bronze stage: schema enforcement and corrupt row quarantine."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    stage = BronzeStage(spark, cfg)
    res = stage.run()
    typer.echo(
        f"Bronze stage status: {res.status} | Output rows: {res.output_rows} | Quarantined: {res.quarantine_rows}"
    )
    if res.status != "SUCCESS":
        sys.exit(1)


@app.command()
def silver(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Run Silver stage: entity normalization, dedup, and vectorized NLP enrichment."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    stage = SilverStage(spark, cfg)
    res = stage.run()
    typer.echo(f"Silver stage status: {res.status} | Output rows: {res.output_rows}")
    if res.status != "SUCCESS":
        sys.exit(1)


@app.command()
def gold(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Run Gold stage: PIT joins, rolling SQL windows, BroadcastHashJoin, and feature store assembly."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    stage = GoldStage(spark, cfg)
    res = stage.run()
    typer.echo(f"Gold stage status: {res.status} | Output rows: {res.output_rows}")
    if res.status != "SUCCESS":
        sys.exit(1)


@app.command()
def train(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Train Propensity and PD models, calibrate with Platt on Val, and log to MLflow @challenger."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    typer.echo("Training models...")
    res = train_all_models(spark, cfg)
    typer.echo(f"Propensity OOT Gini: {res['propensity_metrics']['oot_gini']:.4f}")
    typer.echo(f"PD OOT Gini: {res['pd_metrics']['oot_gini']:.4f}")
    typer.echo("Training complete and models registered with alias @challenger.")


@app.command()
def validate(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Run distributed validation checks and evaluate enterprise Model Risk governance gates."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    from credittarget.validation.gates import GovernancePolicyEvaluator

    evaluator = GovernancePolicyEvaluator(cfg)
    typer.echo("Evaluating Model Risk governance gates...")
    res = evaluator.evaluate_gates(spark)
    typer.echo(f"Governance Gate Verdict: {res['verdict']}")
    for rule in res["rules"]:
        status_symbol = "PASS" if rule["passed"] else "FAIL"
        typer.echo(f" - [{status_symbol}] {rule['rule_name']}: {rule['message']}")
    if res["verdict"] == "FAIL":
        sys.exit(1)


@app.command()
def score(
    snapshot_month: str = typer.Option(
        "2025-12", "--snapshot-month", "-m", help="Snapshot month to score."
    ),
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Run distributed batch scoring for target month using champion models and EV policy."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    stage = BatchScoreStage(spark, cfg)
    res = stage.run(snapshot_month=snapshot_month)
    typer.echo(
        f"Batch scoring completed for {snapshot_month} | Status: {res.status} | Scored rows: {res.output_rows}"
    )
    if res.status != "SUCCESS":
        sys.exit(1)


@app.command()
def report(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
) -> None:
    """Generate Model Risk Validation Memo and lineage manifest."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    artifact_store = ArtifactStore(cfg.storage)
    delta_mgr = DeltaTableManager(spark)
    mlflow_client = ModelRegistryClient(cfg.mlflow)

    typer.echo("Assembling lineage manifest...")
    build_lineage_manifest(cfg, artifact_store, delta_mgr, mlflow_client)

    typer.echo("Generating Grounded Model Risk Validation Memo...")
    reporter = ModelRiskValidationReporter(cfg)
    reporter.generate_memo()
    memo_path = f"{cfg.storage.base_dir}/governance/model_risk_validation_memo.md"
    typer.echo(f"Validation memo generated successfully at: {memo_path}")


@app.command("run-all")
def run_all(
    config: str = typer.Option(
        "configs/local.yaml", "--config", "-c", help="Path to config profile."
    ),
    snapshot_month: str = typer.Option(
        "2025-12", "--snapshot-month", "-m", help="Target scoring snapshot month."
    ),
) -> None:
    """Execute complete end-to-end platform pipeline from data gen to governance memo."""
    configure_logging()
    cfg = load_config(config)
    spark = get_spark_session(cfg)
    typer.echo("=== [1/8] Generating Synthetic Banking Data ===")
    raw_tables = generate_synthetic_bank_data(spark, cfg)

    typer.echo("=== [2/8] Bronze Medallion Stage ===")
    b_res = BronzeStage(spark, cfg, raw_tables=raw_tables).run()
    if b_res.status != "SUCCESS":
        sys.exit(1)

    typer.echo("=== [3/8] Silver Medallion Stage ===")
    s_res = SilverStage(spark, cfg).run()
    if s_res.status != "SUCCESS":
        sys.exit(1)

    typer.echo("=== [4/8] Gold Feature Store Stage ===")
    g_res = GoldStage(spark, cfg).run()
    if g_res.status != "SUCCESS":
        sys.exit(1)

    typer.echo("=== [5/8] Model Training & MLflow Registration ===")
    train_all_models(spark, cfg)

    typer.echo("=== [6/8] Distributed Validation & Governance Gates ===")
    from credittarget.validation.gates import GovernancePolicyEvaluator

    evaluator = GovernancePolicyEvaluator(cfg)
    eval_res = evaluator.evaluate_gates(spark)
    if eval_res["verdict"] == "FAIL":
        sys.exit(1)

    typer.echo("=== [7/8] Batch Scoring Champion Model ===")
    score_res = BatchScoreStage(spark, cfg).run(snapshot_month=snapshot_month)
    if score_res.status != "SUCCESS":
        sys.exit(1)

    typer.echo("=== [8/8] Lineage & Model Risk Validation Memo ===")
    artifact_store = ArtifactStore(cfg.storage)
    delta_mgr = DeltaTableManager(spark)
    mlflow_client = ModelRegistryClient(cfg.mlflow)
    build_lineage_manifest(cfg, artifact_store, delta_mgr, mlflow_client)
    reporter = ModelRiskValidationReporter(cfg)
    reporter.generate_memo()

    typer.echo("=== CreditTarget E2E Pipeline Run COMPLETED Successfully ===")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="API bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="API bind port."),
) -> None:
    """Start FastAPI inference service with Prometheus monitoring."""
    uvicorn.run("credittarget.serving.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
