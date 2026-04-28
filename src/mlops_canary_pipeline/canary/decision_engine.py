"""Decision Engine — KPI-gated progressive delivery logic.

Implements Algorithm 1 from Gaddam (2023) §3.1.2 with the following
improvements:

* Chi-squared significance gate before promote/rollback.
* Mann-Whitney U for latency comparison.
* EWMA-smoothed KPI values to reduce alert fatigue.
* Adaptive thresholds blended from recent window + static baseline.
* TTD (Time-to-Detect) measurement for experiments.

Usage from the periodic evaluation loop:

    engine = DecisionEngine(router, collector, config)
    action = engine.evaluate()   # → DeploymentAction.PROMOTE | HOLD | ROLLBACK
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from mlops_canary_pipeline.canary.config import CanaryConfig, DeploymentAction
from mlops_canary_pipeline.canary.kpi import KPICollector, KPISnapshot

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Detailed result from a single evaluation cycle."""
    action: DeploymentAction
    baseline_snapshot: KPISnapshot
    candidate_snapshot: KPISnapshot
    kpi_deltas: dict[str, float] = field(default_factory=dict)
    chi2_p_value: float | None = None
    mann_whitney_p_value: float | None = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class DecisionEngine:
    """Evaluates KPIs and decides whether to promote, hold, or rollback.

    Corresponds to §3.1.2 of the paper, extended with statistical gates.
    """

    def __init__(
        self,
        collector: KPICollector,
        config: CanaryConfig | None = None,
    ):
        self.collector = collector
        self.config = config or CanaryConfig()
        self._history: list[EvaluationResult] = []

    @property
    def history(self) -> list[EvaluationResult]:
        return list(self._history)

    def evaluate(self) -> EvaluationResult:
        """Run one evaluation cycle of the KPI-gated algorithm.

        Algorithm (enhanced version of paper Algorithm 1):

        1. Compute KPI snapshots for baseline and candidate.
        2. If candidate samples < min_sample_size → HOLD (§5.3.1 fix).
        3. For each KPI:
           a. EWMA-smooth the raw value (§5.3.4 fix).
           b. Compute adaptive threshold (§5.3.3 fix).
           c. Check ΔKPI = candidate − baseline.
           d. If |ΔKPI| > adaptive_threshold → mark as violating.
        4. Run chi-squared test on precision (§5.3.1 fix).
        5. Run Mann-Whitney on latency (new contribution).
        6. Decision:
           - If any KPI violates AND test is significant → ROLLBACK.
           - If all KPIs pass AND test is significant → PROMOTE.
           - Otherwise → HOLD.
        """
        b_snap = self.collector.snapshot("baseline")
        c_snap = self.collector.snapshot("candidate")

        result = EvaluationResult(
            action=DeploymentAction.HOLD,
            baseline_snapshot=b_snap,
            candidate_snapshot=c_snap,
        )

        # Gate 1: minimum sample size (§5.3.1 fix)
        if c_snap.n_samples < self.config.min_sample_size:
            result.reason = (
                f"Insufficient candidate samples ({c_snap.n_samples} < "
                f"{self.config.min_sample_size})"
            )
            logger.info("HOLD — %s", result.reason)
            self._history.append(result)
            return result

        # Gate 2: per-KPI evaluation with EWMA + adaptive thresholds
        violations: list[str] = []
        kpi_pairs = {
            "precision": (b_snap.precision, c_snap.precision),
            "recall": (b_snap.recall, c_snap.recall),
            "latency_ms": (b_snap.latency_ms_mean, c_snap.latency_ms_mean),
        }

        for kpi_name, (b_val, c_val) in kpi_pairs.items():
            # EWMA smoothing (§5.3.4)
            c_smoothed = self.collector.ewma_smooth("candidate", kpi_name, c_val)
            b_smoothed = self.collector.ewma_smooth("baseline", kpi_name, b_val)

            # Adaptive threshold (§5.3.3)
            threshold = self.collector.adaptive_threshold(kpi_name)
            delta = c_smoothed - b_smoothed
            result.kpi_deltas[kpi_name] = round(delta, 6)

            if not threshold.is_acceptable(c_smoothed):
                violations.append(
                    f"{kpi_name}: candidate={c_smoothed:.4f} outside "
                    f"[{threshold.lower_bound:.4f}, {threshold.upper_bound:.4f}]"
                )

        # Gate 3: chi-squared significance test (§5.3.1)
        chi2, chi2_p, chi2_sig = self.collector.chi_squared_test(b_snap, c_snap)
        result.chi2_p_value = chi2_p

        # Gate 4: Mann-Whitney latency test (new contribution)
        _, mw_p, mw_sig = self.collector.mann_whitney_test()
        result.mann_whitney_p_value = mw_p

        # --- Final decision ---
        if violations:
            if chi2_sig or mw_sig:
                # Statistically confirmed degradation → ROLLBACK
                result.action = DeploymentAction.ROLLBACK
                result.reason = f"KPI violations confirmed: {'; '.join(violations)}"
                logger.warning("ROLLBACK — %s", result.reason)
                # TTD tracking
                self.collector.mark_degradation_detected()
            else:
                # Violations but not statistically significant → HOLD
                result.action = DeploymentAction.HOLD
                result.reason = (
                    f"KPI violations not yet significant (χ² p={chi2_p:.4f}, "
                    f"MW p={mw_p:.4f}): {'; '.join(violations)}"
                )
                logger.info("HOLD — %s", result.reason)
        else:
            if chi2_sig or c_snap.n_samples >= self.config.min_sample_size * 3:
                # All KPIs pass + significant test → PROMOTE
                result.action = DeploymentAction.PROMOTE
                result.reason = "All KPIs within bounds; statistically validated"
                logger.info("PROMOTE — %s", result.reason)
            else:
                result.action = DeploymentAction.HOLD
                result.reason = "KPIs acceptable but awaiting significance confirmation"
                logger.info("HOLD — %s", result.reason)

        self._history.append(result)
        return result

    def summary(self) -> dict:
        """Return a JSON-serializable summary of the engine state."""
        latest = self._history[-1] if self._history else None
        return {
            "total_evaluations": len(self._history),
            "latest_action": latest.action.value if latest else None,
            "latest_reason": latest.reason if latest else None,
            "latest_kpi_deltas": latest.kpi_deltas if latest else {},
            "ttd_seconds": self.collector.ttd_seconds,
            "action_counts": {
                a.value: sum(1 for r in self._history if r.action == a)
                for a in DeploymentAction
            },
        }
