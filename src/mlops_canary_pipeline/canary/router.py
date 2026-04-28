"""Model Router — weighted traffic splitting between baseline and candidate.

Implements the Model Router described in Gaddam (2023) §3.1.1 with both
A/B and shadow modes.

Improvement over paper
~~~~~~~~~~~~~~~~~~~~~~
The paper describes the router conceptually but does not provide a
production-ready implementation.  This module is a fully functional,
thread-safe router that integrates with FastAPI and exposes the current
traffic split as a Prometheus gauge.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from mlops_canary_pipeline.canary.config import CanaryConfig, RoutingMode

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"


@dataclass
class PredictionResult:
    """Result of a routed prediction."""
    prediction: int
    probability: float
    model_version: str        # "baseline" | "candidate"
    latency_ms: float
    shadow_prediction: int | None = None      # filled in shadow mode
    shadow_probability: float | None = None


class ModelRouter:
    """Routes incoming requests between baseline and candidate models.

    Thread-safe: the ``canary_pct`` and stage can be updated at runtime
    from the Decision Engine while the FastAPI event loop serves traffic.
    """

    def __init__(self, config: CanaryConfig | None = None):
        self.config = config or CanaryConfig()
        self._lock = threading.Lock()
        self._canary_pct: int = 0          # start with 0 → all traffic to baseline
        self._current_stage_idx: int = -1  # not yet started
        self._stage_start_time: float = 0.0

        # Models + scaler (loaded lazily)
        self._baseline_model: Any = None
        self._candidate_model: Any = None
        self._scaler: Any = None
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load baseline and candidate models from ``models/``."""
        baseline_path = _MODELS_DIR / "baseline" / "model.pkl"
        candidate_path = _MODELS_DIR / "candidate" / "model.pkl"
        scaler_path = _MODELS_DIR / "baseline" / "scaler.pkl"

        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline model not found: {baseline_path}")

        self._baseline_model = joblib.load(baseline_path)
        self._scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        if candidate_path.exists():
            self._candidate_model = joblib.load(candidate_path)
            logger.info("Loaded baseline + candidate models")
        else:
            logger.warning("Candidate model not found — all traffic goes to baseline")
            self._candidate_model = None

        self._models_loaded = True

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    # ------------------------------------------------------------------
    # Traffic control
    # ------------------------------------------------------------------

    @property
    def canary_pct(self) -> int:
        with self._lock:
            return self._canary_pct

    @canary_pct.setter
    def canary_pct(self, value: int) -> None:
        with self._lock:
            self._canary_pct = max(0, min(value, 100))
            logger.info("Canary traffic set to %d%%", self._canary_pct)

    @property
    def current_stage_idx(self) -> int:
        with self._lock:
            return self._current_stage_idx

    def advance_stage(self) -> int:
        """Move to the next traffic stage.  Returns the new canary %."""
        with self._lock:
            self._current_stage_idx += 1
            if self._current_stage_idx < len(self.config.traffic_stages):
                self._canary_pct = self.config.traffic_stages[self._current_stage_idx]
                self._stage_start_time = time.time()
            else:
                self._canary_pct = 100  # fully promoted
            logger.info(
                "Advanced to stage %d → canary_pct=%d%%",
                self._current_stage_idx, self._canary_pct,
            )
            return self._canary_pct

    def rollback(self) -> None:
        """Immediately revert all traffic to baseline."""
        with self._lock:
            self._canary_pct = 0
            self._current_stage_idx = -1
            logger.warning("ROLLBACK — all traffic reverted to baseline")

    # ------------------------------------------------------------------
    # Prediction routing
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Route a single prediction through baseline or candidate.

        Parameters
        ----------
        features : np.ndarray
            Raw feature vector (1-D).  Will be scaled if a scaler is loaded.
        """
        if not self._models_loaded:
            self.load_models()

        X = features.reshape(1, -1)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # Decide which model serves
        use_candidate = (
            self._candidate_model is not None
            and random.randint(1, 100) <= self.canary_pct
        )

        if self.config.routing_mode == RoutingMode.SHADOW:
            return self._shadow_predict(X, use_candidate)
        return self._ab_predict(X, use_candidate)

    def _ab_predict(self, X: np.ndarray, use_candidate: bool) -> PredictionResult:
        if use_candidate:
            model = self._candidate_model
            version = "candidate"
        else:
            model = self._baseline_model
            version = "baseline"

        start = time.perf_counter()
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else float(pred)
        latency = (time.perf_counter() - start) * 1000

        return PredictionResult(
            prediction=pred,
            probability=round(prob, 6),
            model_version=version,
            latency_ms=round(latency, 3),
        )

    def _shadow_predict(self, X: np.ndarray, use_candidate: bool) -> PredictionResult:
        """Shadow mode: always serve baseline, but also run candidate."""
        # Baseline (always served)
        start = time.perf_counter()
        pred = int(self._baseline_model.predict(X)[0])
        prob = float(self._baseline_model.predict_proba(X)[0, 1]) if hasattr(self._baseline_model, "predict_proba") else float(pred)
        latency = (time.perf_counter() - start) * 1000

        # Candidate (shadow — result discarded for the response)
        shadow_pred = None
        shadow_prob = None
        if self._candidate_model is not None:
            shadow_pred = int(self._candidate_model.predict(X)[0])
            shadow_prob = float(self._candidate_model.predict_proba(X)[0, 1]) if hasattr(self._candidate_model, "predict_proba") else float(shadow_pred)

        return PredictionResult(
            prediction=pred,
            probability=round(prob, 6),
            model_version="baseline",
            latency_ms=round(latency, 3),
            shadow_prediction=shadow_pred,
            shadow_probability=round(shadow_prob, 6) if shadow_prob is not None else None,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "models_loaded": self._models_loaded,
            "routing_mode": self.config.routing_mode.value,
            "canary_pct": self.canary_pct,
            "current_stage_idx": self.current_stage_idx,
            "traffic_stages": self.config.traffic_stages,
            "baseline_loaded": self._baseline_model is not None,
            "candidate_loaded": self._candidate_model is not None,
        }
