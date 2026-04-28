"""Model training with MLflow experiment tracking.

Trains both a **baseline** (XGBoost) and a **candidate** (LightGBM) model,
matching the FinTech case study in Gaddam (2023) §4.  Each run logs
hyper-parameters, evaluation metrics and model artifacts to MLflow.

Usage:

    python -m mlops_canary_pipeline.training.train
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_canary_pipeline.data.ingest import load_dataset
from mlops_canary_pipeline.data.preprocess import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "canary-pipeline")


def _evaluate(model, X, y) -> dict[str, float]:
    """Return a dict of KPI metrics."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    latency_ms = (time.perf_counter() - start) * 1000 / len(X)  # per-sample

    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "latency_ms": round(latency_ms, 4),
    }


def _save_model(model, scaler, name: str) -> Path:
    """Persist model + scaler to ``models/<name>/``."""
    model_dir = _MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved %s → %s", name, model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, **kwargs):
    """Train an XGBoost classifier (baseline)."""
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model, params


def train_lightgbm(X_train, y_train, **kwargs):
    """Train a LightGBM classifier (candidate)."""
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 250,
        "max_depth": 8,
        "learning_rate": 0.05,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline() -> dict:
    """Full pipeline: ingest → preprocess → train baseline + candidate.

    Returns a dict with metrics for both models.
    """
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # --- Data ---
    df = load_dataset()
    data = preprocess(df, apply_smote=True)

    # --- MLflow setup ---
    try:
        mlflow.set_tracking_uri(_tracking_uri())
        mlflow.set_experiment(_experiment_name())
        mlflow_ok = True
        logger.info("MLflow tracking at %s", _tracking_uri())
    except Exception as exc:
        logger.warning("MLflow not available (%s) — logging locally", exc)
        mlflow_ok = False

    results = {}

    # ---- Baseline: XGBoost (3 hyper-param runs) ----
    xgb_configs = [
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.15},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.10},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    ]
    best_baseline_f1 = -1.0
    best_baseline_model = None

    for i, cfg in enumerate(xgb_configs, 1):
        logger.info("Training XGBoost run %d/%d  %s", i, len(xgb_configs), cfg)
        model, params = train_xgboost(data.X_train, data.y_train, **cfg)
        val_metrics = _evaluate(model, data.X_val, data.y_val)
        test_metrics = _evaluate(model, data.X_test, data.y_test)

        if mlflow_ok:
            with mlflow.start_run(run_name=f"xgboost-run-{i}"):
                mlflow.log_params(params)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.set_tag("model_type", "xgboost")
                mlflow.set_tag("role", "baseline")
                mlflow.sklearn.log_model(model, "model")

        logger.info("  val_f1=%.4f  val_precision=%.4f  val_recall=%.4f",
                     val_metrics["f1"], val_metrics["precision"], val_metrics["recall"])

        if val_metrics["f1"] > best_baseline_f1:
            best_baseline_f1 = val_metrics["f1"]
            best_baseline_model = model
            results["baseline_val"] = val_metrics
            results["baseline_test"] = test_metrics

    _save_model(best_baseline_model, data.scaler, "baseline")

    # ---- Candidate: LightGBM (3 hyper-param runs) ----
    lgb_configs = [
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.08},
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.05},
        {"n_estimators": 350, "max_depth": 10, "learning_rate": 0.03},
    ]
    best_candidate_f1 = -1.0
    best_candidate_model = None

    for i, cfg in enumerate(lgb_configs, 1):
        logger.info("Training LightGBM run %d/%d  %s", i, len(lgb_configs), cfg)
        model, params = train_lightgbm(data.X_train, data.y_train, **cfg)
        val_metrics = _evaluate(model, data.X_val, data.y_val)
        test_metrics = _evaluate(model, data.X_test, data.y_test)

        if mlflow_ok:
            with mlflow.start_run(run_name=f"lightgbm-run-{i}"):
                mlflow.log_params(params)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.set_tag("model_type", "lightgbm")
                mlflow.set_tag("role", "candidate")
                mlflow.sklearn.log_model(model, "model")

        logger.info("  val_f1=%.4f  val_precision=%.4f  val_recall=%.4f",
                     val_metrics["f1"], val_metrics["precision"], val_metrics["recall"])

        if val_metrics["f1"] > best_candidate_f1:
            best_candidate_f1 = val_metrics["f1"]
            best_candidate_model = model
            results["candidate_val"] = val_metrics
            results["candidate_test"] = test_metrics

    _save_model(best_candidate_model, data.scaler, "candidate")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info(
        "Baseline  (XGBoost)  — test F1: %.4f  precision: %.4f  recall: %.4f",
        results["baseline_test"]["f1"],
        results["baseline_test"]["precision"],
        results["baseline_test"]["recall"],
    )
    logger.info(
        "Candidate (LightGBM) — test F1: %.4f  precision: %.4f  recall: %.4f",
        results["candidate_test"]["f1"],
        results["candidate_test"]["precision"],
        results["candidate_test"]["recall"],
    )
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_training_pipeline()
