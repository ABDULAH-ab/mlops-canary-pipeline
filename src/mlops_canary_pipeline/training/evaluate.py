"""Extended model evaluation with KPIs from Gaddam (2023) Table 2.

Computes precision, recall, F1, ROC-AUC, latency, and a simple fairness
index.  Results are printed as a comparison table and (optionally) logged
to MLflow.

Usage:

    python -m mlops_canary_pipeline.training.evaluate
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_canary_pipeline.data.ingest import load_dataset
from mlops_canary_pipeline.data.preprocess import preprocess

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"


def _load_model(name: str):
    """Load a persisted model from ``models/<name>/``."""
    model_path = _MODELS_DIR / name / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def compute_kpis(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute the full KPI set matching Table 2 of the paper."""
    # Latency measurement (per-sample average)
    n_runs = 5
    latencies: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y_pred = model.predict(X)
        elapsed = (time.perf_counter() - start) * 1000  # ms total
        latencies.append(elapsed / len(X))
    latency_ms = float(np.median(latencies))

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    # Simple fairness index: ratio of precision across two synthetic subgroups.
    # We split data at the median of the first feature as a proxy.
    mid = np.median(X[:, 0])
    group_a = X[:, 0] <= mid
    group_b = ~group_a
    prec_a = precision_score(y[group_a], y_pred[group_a], zero_division=0)
    prec_b = precision_score(y[group_b], y_pred[group_b], zero_division=0)
    fairness_index = min(prec_a, prec_b) / max(prec_a, prec_b, 1e-9)

    return {
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "latency_ms": round(latency_ms, 4),
        "fairness_index": round(fairness_index, 4),
    }


def evaluate_models() -> dict[str, dict[str, float]]:
    """Load baseline + candidate and evaluate both on the test set."""
    df = load_dataset()
    data = preprocess(df, apply_smote=False)  # No SMOTE on eval data

    results = {}
    for name in ("baseline", "candidate"):
        try:
            model = _load_model(name)
        except FileNotFoundError:
            logger.warning("Skipping %s — model file not found", name)
            continue
        kpis = compute_kpis(model, data.X_test, data.y_test)
        results[name] = kpis
        logger.info(
            "%s KPIs: precision=%.4f  recall=%.4f  f1=%.4f  auc=%.4f  "
            "latency=%.3fms  fairness=%.4f",
            name, kpis["precision"], kpis["recall"], kpis["f1"],
            kpis["roc_auc"], kpis["latency_ms"], kpis["fairness_index"],
        )

        # Classification report
        y_pred = model.predict(data.X_test)
        logger.info("\n%s classification report:\n%s", name,
                     classification_report(data.y_test, y_pred, zero_division=0))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = evaluate_models()

    if len(results) == 2:
        print("\n" + "=" * 60)
        print("KPI Comparison: Baseline vs Candidate")
        print("=" * 60)
        header = f"{'KPI':<20} {'Baseline':>12} {'Candidate':>12} {'Δ':>10}"
        print(header)
        print("-" * len(header))
        for kpi in results["baseline"]:
            b = results["baseline"][kpi]
            c = results["candidate"][kpi]
            delta = c - b
            sign = "+" if delta >= 0 else ""
            print(f"{kpi:<20} {b:>12.4f} {c:>12.4f} {sign}{delta:>9.4f}")
        print("=" * 60)
