"""Rollback Controller — emergency brake for KPI degradation.

Implements §3.2 of Gaddam (2023).  When the Decision Engine emits a
ROLLBACK signal, this controller:

1. Reverts all traffic to the baseline model via the Model Router.
2. Logs the rollback event (timestamp, reason, KPI snapshot).
3. Increments the Prometheus ``canary_rollback_total`` counter.

Improvement over the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~
The paper describes PagerDuty alerting (proprietary).  We use open-source
Prometheus Alertmanager with tiered rules (critical / warning / info).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlops_canary_pipeline.canary.decision_engine import EvaluationResult
    from mlops_canary_pipeline.canary.router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class RollbackEvent:
    """Audit record for a single rollback."""
    timestamp: float
    reason: str
    baseline_precision: float
    candidate_precision: float
    canary_pct_before: int


class RollbackController:
    """Handles automatic and manual rollbacks.

    Keeps an audit log of every rollback for post-incident review.
    """

    def __init__(self) -> None:
        self._events: list[RollbackEvent] = []
        self._rollback_callback: list = []  # optional hooks

    @property
    def events(self) -> list[RollbackEvent]:
        return list(self._events)

    @property
    def total_rollbacks(self) -> int:
        return len(self._events)

    def execute_rollback(
        self,
        router: ModelRouter,
        result: EvaluationResult,
    ) -> RollbackEvent:
        """Perform an immediate rollback.

        1. Set canary_pct → 0 on the router.
        2. Record the event.
        3. Log at WARNING level.
        """
        pct_before = router.canary_pct

        # 1. Revert traffic
        router.rollback()

        # 2. Audit record
        event = RollbackEvent(
            timestamp=time.time(),
            reason=result.reason,
            baseline_precision=result.baseline_snapshot.precision,
            candidate_precision=result.candidate_snapshot.precision,
            canary_pct_before=pct_before,
        )
        self._events.append(event)

        # 3. Log
        logger.warning(
            "🔴  ROLLBACK #%d executed  |  reason: %s  |  "
            "baseline_prec=%.4f  candidate_prec=%.4f  |  "
            "traffic was at %d%%",
            len(self._events),
            event.reason,
            event.baseline_precision,
            event.candidate_precision,
            event.canary_pct_before,
        )

        # 4. Fire callbacks (e.g., Alertmanager webhook)
        for cb in self._rollback_callback:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Rollback callback failed: %s", exc)

        return event

    def manual_rollback(self, router: ModelRouter, reason: str = "manual") -> RollbackEvent:
        """Trigger a rollback manually (e.g., from the /canary/rollback endpoint)."""
        from mlops_canary_pipeline.canary.decision_engine import EvaluationResult
        from mlops_canary_pipeline.canary.kpi import KPISnapshot

        # Create a minimal EvaluationResult for the audit record
        dummy_result = EvaluationResult(
            action="rollback",
            baseline_snapshot=KPISnapshot(),
            candidate_snapshot=KPISnapshot(),
            reason=reason,
        )
        return self.execute_rollback(router, dummy_result)

    def register_callback(self, callback) -> None:
        """Register a function to call on every rollback (e.g., Alertmanager)."""
        self._rollback_callback.append(callback)

    def summary(self) -> dict:
        """JSON-serializable rollback history."""
        return {
            "total_rollbacks": self.total_rollbacks,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "reason": e.reason,
                    "baseline_precision": e.baseline_precision,
                    "candidate_precision": e.candidate_precision,
                    "canary_pct_before": e.canary_pct_before,
                }
                for e in self._events[-10:]   # last 10 for brevity
            ],
        }
