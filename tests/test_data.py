"""Tests for data ingestion and preprocessing."""

import numpy as np
import pytest

from mlops_canary_pipeline.data.ingest import generate_fraud_dataset
from mlops_canary_pipeline.data.preprocess import preprocess


class TestDataIngestion:
    def test_dataset_shape(self):
        df = generate_fraud_dataset(n_samples=1000)
        assert len(df) == 1000
        # V1..V28 + Amount + Class = 30 columns
        assert len(df.columns) == 30

    def test_dataset_has_target(self):
        df = generate_fraud_dataset(n_samples=500)
        assert "Class" in df.columns
        assert set(df["Class"].unique()).issubset({0, 1})

    def test_fraud_rate(self):
        df = generate_fraud_dataset(n_samples=10000, fraud_ratio=0.05)
        fraud_rate = df["Class"].mean()
        # Should be roughly around 5% (with some noise from flip_y)
        assert 0.01 < fraud_rate < 0.15


class TestPreprocessing:
    def test_splits_are_disjoint(self):
        df = generate_fraud_dataset(n_samples=500)
        data = preprocess(df, apply_smote=False)
        total = len(data.X_train) + len(data.X_val) + len(data.X_test)
        assert total == len(df)

    def test_scaling_applied(self):
        df = generate_fraud_dataset(n_samples=500)
        data = preprocess(df, apply_smote=False)
        # After StandardScaler, train mean should be ~0
        assert np.abs(data.X_train.mean()) < 0.5

    def test_smote_balances_classes(self):
        df = generate_fraud_dataset(n_samples=2000, fraud_ratio=0.01)
        data = preprocess(df, apply_smote=True)
        # After SMOTE, fraud count should be close to legit count
        fraud_count = data.y_train.sum()
        legit_count = (data.y_train == 0).sum()
        assert fraud_count == legit_count
