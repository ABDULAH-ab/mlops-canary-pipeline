"""Synthetic fraud-detection dataset generator.

Produces a dataset that mimics the FinTech fraud-detection case study
described in Gaddam (2023) §4.  We use ``sklearn.datasets.make_classification``
with heavy class imbalance (~0.17 % fraud rate) so the pipeline can be
demonstrated end-to-end on any laptop without external data downloads.

Running this module directly writes the dataset to ``data/``:

    python -m mlops_canary_pipeline.data.ingest
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_N_SAMPLES = 100_000
_N_FEATURES = 28          # PCA-like features (mirrors Kaggle credit-card set)
_N_INFORMATIVE = 20
_N_REDUNDANT = 5
_FRAUD_RATIO = 0.02       # ~2 % positive class (sufficient for model learning)
_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def generate_fraud_dataset(
    n_samples: int = _N_SAMPLES,
    fraud_ratio: float = _FRAUD_RATIO,
    random_state: int = _RANDOM_STATE,
) -> pd.DataFrame:
    """Return a synthetic fraud-detection DataFrame.

    Columns: V1 … V28 (features), Amount (simulated), Class (0/1).
    """
    weight_legit = 1.0 - fraud_ratio
    X, y = make_classification(
        n_samples=n_samples,
        n_features=_N_FEATURES,
        n_informative=_N_INFORMATIVE,
        n_redundant=_N_REDUNDANT,
        n_clusters_per_class=2,
        weights=[weight_legit, fraud_ratio],
        flip_y=0.005,
        random_state=random_state,
    )

    feature_cols = [f"V{i}" for i in range(1, _N_FEATURES + 1)]
    df = pd.DataFrame(X, columns=feature_cols)

    # Simulate a realistic "Amount" column (log-normal)
    rng = np.random.default_rng(random_state)
    df["Amount"] = np.round(rng.lognormal(mean=3.5, sigma=1.8, size=n_samples), 2)
    df["Class"] = y

    logger.info(
        "Generated dataset: %d rows, fraud rate %.2f%% (%d positives)",
        len(df),
        y.mean() * 100,
        y.sum(),
    )
    return df


def save_dataset(df: pd.DataFrame, data_dir: str | Path | None = None) -> Path:
    """Persist the DataFrame to ``data/fraud_dataset.csv``."""
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "fraud_dataset.csv"
    df.to_csv(path, index=False)
    logger.info("Saved dataset → %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the fraud dataset from disk.  Generate it first if missing."""
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    path = data_dir / "fraud_dataset.csv"
    if not path.exists():
        logger.warning("Dataset not found at %s — generating …", path)
        df = generate_fraud_dataset()
        save_dataset(df, data_dir)
        return df
    df = pd.read_csv(path)
    logger.info("Loaded dataset from %s (%d rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = generate_fraud_dataset()
    out = save_dataset(df)
    print(f"\n✅  Dataset saved to {out}")
    print(f"    Rows : {len(df):,}")
    print(f"    Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
