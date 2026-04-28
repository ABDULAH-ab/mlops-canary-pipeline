"""Data preprocessing — scaling, splitting and SMOTE oversampling.

Running this module directly preprocesses the dataset and reports statistics:

    python -m mlops_canary_pipeline.data.preprocess
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_TEST_SIZE = 0.2
_VAL_SIZE = 0.15   # fraction of the *training* split
_RANDOM_STATE = 42
_TARGET_COL = "Class"


@dataclass
class SplitData:
    """Container for train / validation / test arrays."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def preprocess(
    df: pd.DataFrame,
    *,
    apply_smote: bool = True,
    test_size: float = _TEST_SIZE,
    val_size: float = _VAL_SIZE,
    random_state: int = _RANDOM_STATE,
) -> SplitData:
    """Clean, scale, split and (optionally) oversample the fraud dataset.

    Steps
    -----
    1. Drop any rows with NaN.
    2. Train / test split  (stratified).
    3. Further split train → train + val  (stratified).
    4. ``StandardScaler`` fit on train, transform all.
    5. SMOTE on training split only  (if *apply_smote*).
    """
    df = df.dropna().copy()
    feature_cols = [c for c in df.columns if c != _TARGET_COL]

    X = df[feature_cols].values
    y = df[_TARGET_COL].values

    # --- train / test split ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    # --- train / validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, stratify=y_train_full, random_state=random_state,
    )

    # --- scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # --- SMOTE (training split only) ---
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            logger.info(
                "After SMOTE — train: %d (fraud: %d, legit: %d)",
                len(X_train),
                int(y_train.sum()),
                int((y_train == 0).sum()),
            )
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed — skipping SMOTE.  "
                "Install with:  pip install imbalanced-learn"
            )

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from mlops_canary_pipeline.data.ingest import load_dataset

    df = load_dataset()
    data = preprocess(df)
    print(f"\n✅  Preprocessing complete")
    print(f"    Train : {len(data.X_train):,}  (fraud {int(data.y_train.sum()):,})")
    print(f"    Val   : {len(data.X_val):,}  (fraud {int(data.y_val.sum()):,})")
    print(f"    Test  : {len(data.X_test):,}  (fraud {int(data.y_test.sum()):,})")
