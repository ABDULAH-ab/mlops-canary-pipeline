"""FastAPI application with canary deployment endpoints.

This is the main entry-point for the MLOps Canary Pipeline service.
It replaces the demo scaffold with real model inference routed through
the Model Router, and exposes canary management endpoints alongside
Prometheus metrics for Grafana dashboards.

Run locally:

    uvicorn mlops_canary_pipeline.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from mlops_canary_pipeline.canary.config import CanaryConfig, DeploymentAction
from mlops_canary_pipeline.canary.decision_engine import DecisionEngine
from mlops_canary_pipeline.canary.kpi import KPICollector, KPISample
from mlops_canary_pipeline.canary.rollback import RollbackController
from mlops_canary_pipeline.canary.router import ModelRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Prometheus metrics ─────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "mlops_http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "mlops_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
LATEST_PREDICTION = Gauge(
    "mlops_latest_prediction_score",
    "Latest model prediction emitted by the predict endpoint",
)

# Canary-specific metrics
CANARY_TRAFFIC_PCT = Gauge(
    "canary_traffic_split_pct",
    "Percentage of traffic routed to the candidate model",
)
CANARY_KPI = Gauge(
    "canary_kpi_value",
    "Current KPI value per model",
    ["model_version", "kpi_name"],
)
CANARY_DECISION = Counter(
    "canary_decision_total",
    "Decision Engine actions",
    ["action"],
)
CANARY_ROLLBACK = Counter(
    "canary_rollback_total",
    "Total rollback events",
)
CANARY_PREDICTION_LATENCY = Histogram(
    "canary_prediction_latency_ms",
    "Per-request model inference latency in milliseconds",
    ["model_version"],
    buckets=[5, 10, 25, 50, 75, 100, 150, 200, 500],
)
CANARY_TTD = Gauge(
    "canary_ttd_seconds",
    "Time-to-Detect degradation in seconds (research metric)",
)

# ─── Global state ───────────────────────────────────────────────────────
canary_config = CanaryConfig()
router = ModelRouter(config=canary_config)
kpi_collector = KPICollector(config=canary_config)
decision_engine = DecisionEngine(collector=kpi_collector, config=canary_config)
rollback_controller = RollbackController()


# ─── Pydantic schemas ──────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Feature vector (V1…V28 + Amount = 29 values)",
        min_length=1,
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float
    tracking_uri: str


class FeedbackRequest(BaseModel):
    ground_truth: int = Field(..., description="Actual label (0 or 1)")
    model_version: str = Field("baseline", description="Which model served this request")


class CanaryConfigureRequest(BaseModel):
    canary_pct: int = Field(..., ge=0, le=100, description="Traffic % for candidate")


class CanaryConfigureResponse(BaseModel):
    canary_pct: int
    message: str


# ─── Helpers ────────────────────────────────────────────────────────────
def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _try_load_models() -> None:
    """Attempt to load models at startup; log warning if not available."""
    try:
        router.load_models()
        logger.info("Models loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("Models not found (%s) — run training first", exc)


# ─── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Canary Pipeline",
    version="0.2.0",
    description=(
        "KPI-gated progressive delivery for ML models.  "
        "Implements Gaddam (2023) with statistical significance gates."
    ),
)


@app.on_event("startup")
async def startup() -> None:
    _try_load_models()


# ─── Middleware ──────────────────────────────────────────────────────────
@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start_time
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    return response


# ═══════════════════════════════════════════════════════════════════════
#  Core endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "mlops-canary-pipeline",
        "version": "0.2.0",
        "status": "running",
        "tracking_uri": _tracking_uri(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Route a prediction through baseline or candidate model."""
    if not router.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded — run training first")

    features = np.array(payload.features)
    result = router.predict(features)

    # Record sample for KPI collection
    sample = KPISample(
        model_version=result.model_version,
        prediction=result.prediction,
        probability=result.probability,
        latency_ms=result.latency_ms,
    )
    kpi_collector.record(sample)

    # Update Prometheus metrics
    LATEST_PREDICTION.set(result.probability)
    CANARY_TRAFFIC_PCT.set(router.canary_pct)
    CANARY_PREDICTION_LATENCY.labels(result.model_version).observe(result.latency_ms)

    return PredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        tracking_uri=_tracking_uri(),
    )


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, str]:
    """Receive ground-truth labels for computing precision/recall KPIs."""
    kpi_collector.record_ground_truth(0, payload.ground_truth, payload.model_version)
    return {"status": "recorded"}


# ═══════════════════════════════════════════════════════════════════════
#  Canary management endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/canary/configure", response_model=CanaryConfigureResponse)
def canary_configure(payload: CanaryConfigureRequest) -> CanaryConfigureResponse:
    """Set the canary traffic split percentage."""
    router.canary_pct = payload.canary_pct
    CANARY_TRAFFIC_PCT.set(router.canary_pct)
    return CanaryConfigureResponse(
        canary_pct=router.canary_pct,
        message=f"Traffic split set to {router.canary_pct}%",
    )


@app.get("/canary/status")
def canary_status() -> dict:
    """Return current canary deployment state."""
    # Update KPI gauges
    for model in ("baseline", "candidate"):
        snap = kpi_collector.snapshot(model)
        CANARY_KPI.labels(model, "precision").set(snap.precision)
        CANARY_KPI.labels(model, "recall").set(snap.recall)
        CANARY_KPI.labels(model, "f1").set(snap.f1)
        CANARY_KPI.labels(model, "latency_ms").set(snap.latency_ms_mean)

    ttd = kpi_collector.ttd_seconds
    if ttd is not None:
        CANARY_TTD.set(ttd)

    return {
        "router": router.status(),
        "decision_engine": decision_engine.summary(),
        "rollback_controller": rollback_controller.summary(),
        "kpi_snapshots": {
            "baseline": kpi_collector.snapshot("baseline").__dict__,
            "candidate": kpi_collector.snapshot("candidate").__dict__,
        },
        "ttd_seconds": ttd,
        "config": canary_config.to_dict(),
    }


@app.post("/canary/evaluate")
def canary_evaluate() -> dict:
    """Trigger a single KPI evaluation cycle."""
    result = decision_engine.evaluate()

    # Update Prometheus counters
    CANARY_DECISION.labels(result.action.value).inc()

    # Act on the decision
    if result.action == DeploymentAction.ROLLBACK:
        rollback_controller.execute_rollback(router, result)
        CANARY_ROLLBACK.inc()
    elif result.action == DeploymentAction.PROMOTE:
        new_pct = router.advance_stage()
        CANARY_TRAFFIC_PCT.set(new_pct)

    return {
        "action": result.action.value,
        "reason": result.reason,
        "kpi_deltas": result.kpi_deltas,
        "chi2_p_value": result.chi2_p_value,
        "mann_whitney_p_value": result.mann_whitney_p_value,
        "canary_pct": router.canary_pct,
    }


@app.post("/canary/rollback")
def canary_rollback_manual() -> dict:
    """Manually trigger a rollback to baseline."""
    event = rollback_controller.manual_rollback(router, reason="Manual rollback via API")
    CANARY_ROLLBACK.inc()
    CANARY_TRAFFIC_PCT.set(0)
    return {
        "status": "rolled_back",
        "canary_pct": 0,
        "event": {
            "timestamp": event.timestamp,
            "reason": event.reason,
        },
    }


@app.post("/canary/start")
def canary_start() -> dict:
    """Start a canary rollout at the first traffic stage."""
    if not router.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    new_pct = router.advance_stage()
    CANARY_TRAFFIC_PCT.set(new_pct)
    return {
        "status": "started",
        "canary_pct": new_pct,
        "stage": router.current_stage_idx,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Prometheus metrics endpoint
# ═══════════════════════════════════════════════════════════════════════

@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
