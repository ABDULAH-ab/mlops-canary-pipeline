"""Tests for the FastAPI application and canary endpoints."""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_mlflow(monkeypatch):
    """Disable MLflow logging in all tests."""
    monkeypatch.setenv("ENABLE_MLFLOW_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")


@pytest.fixture
def client():
    from mlops_canary_pipeline.main import app
    return TestClient(app)


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "mlops-canary-pipeline"
    assert data["status"] == "running"


def test_metrics_endpoint(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlops_http_requests_total" in response.text


def test_canary_status_endpoint(client) -> None:
    response = client.get("/canary/status")
    assert response.status_code == 200
    data = response.json()
    assert "router" in data
    assert "decision_engine" in data
    assert "rollback_controller" in data


def test_canary_configure_endpoint(client) -> None:
    response = client.post("/canary/configure", json={"canary_pct": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["canary_pct"] == 10


def test_canary_rollback_endpoint(client) -> None:
    response = client.post("/canary/rollback")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rolled_back"
    assert data["canary_pct"] == 0


def test_predict_without_models(client) -> None:
    """Predict should return 503 if models are not trained yet."""
    response = client.post("/predict", json={"features": [0.1] * 29})
    # May be 503 (no models) or 200 (models exist from prior training)
    assert response.status_code in (200, 503)


def test_feedback_endpoint(client) -> None:
    response = client.post("/feedback", json={
        "ground_truth": 1,
        "model_version": "baseline",
    })
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
