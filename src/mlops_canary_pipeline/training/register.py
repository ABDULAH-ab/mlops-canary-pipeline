"""MLflow Model Registry helper.

Registers the best baseline and candidate models so they can be loaded
via ``mlflow.pyfunc.load_model("models:/<name>/<version>")``.

Usage:

    python -m mlops_canary_pipeline.training.register
"""

from __future__ import annotations

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "canary-pipeline")


def register_best_models() -> dict[str, str]:
    """Find best baseline + candidate runs and register them."""
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()

    experiment = client.get_experiment_by_name(_experiment_name())
    if experiment is None:
        logger.error("Experiment '%s' not found", _experiment_name())
        return {}

    registered: dict[str, str] = {}
    for role, model_type in [("baseline", "xgboost"), ("candidate", "lightgbm")]:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.role = '{role}' AND tags.model_type = '{model_type}'",
            order_by=["metrics.val_f1 DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("No runs found for role=%s", role)
            continue

        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        reg_name = f"fraud-detection-{role}"

        result = mlflow.register_model(model_uri, reg_name)
        logger.info(
            "Registered %s → %s version %s  (val_f1=%.4f)",
            role, reg_name, result.version,
            best_run.data.metrics.get("val_f1", 0),
        )
        registered[role] = f"{reg_name}/{result.version}"

    return registered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    registered = register_best_models()
    if registered:
        print("\n✅  Registered models:")
        for role, name in registered.items():
            print(f"    {role}: {name}")
    else:
        print("\n⚠️  No models registered.  Run training first.")
