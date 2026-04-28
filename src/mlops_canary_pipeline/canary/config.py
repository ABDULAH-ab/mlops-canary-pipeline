"""Canary deployment configuration.

Centralizes KPI thresholds (from Gaddam 2023, Table 2), traffic stages,
and the adaptive-weight λ parameter that constitutes one of this project's
improvements over the base paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeploymentAction(str, Enum):
    """Possible outcomes of the Decision Engine."""
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


class RoutingMode(str, Enum):
    """Traffic routing strategy (paper §3.1.1)."""
    AB = "ab"           # Both models serve real traffic
    SHADOW = "shadow"   # Candidate runs but output is discarded


@dataclass
class KPIThreshold:
    """Defines acceptance bounds for a single KPI.

    Attributes
    ----------
    baseline_mean : float
        μ from the baseline distribution.
    sigma : float
        Standard deviation observed during baseline window.
    n_sigma : float
        Number of standard deviations for the threshold band (default 2).
    direction : str
        ``"higher_better"`` for precision / recall,
        ``"lower_better"``  for latency.
    """
    baseline_mean: float
    sigma: float
    n_sigma: float = 2.0
    direction: str = "higher_better"

    @property
    def lower_bound(self) -> float:
        return self.baseline_mean - self.n_sigma * self.sigma

    @property
    def upper_bound(self) -> float:
        return self.baseline_mean + self.n_sigma * self.sigma

    def is_acceptable(self, value: float) -> bool:
        """Return True when *value* falls within the ±nσ band."""
        if self.direction == "lower_better":
            return value <= self.upper_bound
        return value >= self.lower_bound


@dataclass
class CanaryConfig:
    """Full configuration for a canary rollout cycle.

    Improvement over Gaddam (2023)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``min_sample_size`` — prevents chi-squared false alarms at early
      traffic stages (addresses §5.3.1 limitation).
    * ``ewma_alpha`` — EWMA smoothing factor to dampen metric oscillations
      before the Decision Engine evaluates, reducing alert fatigue (§5.3.4).
    * ``adaptive_lambda`` — weight parameter for blending historical and
      recent thresholds, eliminating manual recalibration (§5.3.3).
    * ``significance_level`` — α for statistical tests (chi-squared,
      Mann-Whitney) that were absent in the original paper.
    """

    # Traffic stages (% of requests to candidate)
    traffic_stages: list[int] = field(default_factory=lambda: [5, 25, 50, 100])

    # KPI threshold configuration (Table 2)
    kpi_thresholds: dict[str, KPIThreshold] = field(default_factory=lambda: {
        "precision": KPIThreshold(baseline_mean=0.918, sigma=0.006, direction="higher_better"),
        "recall": KPIThreshold(baseline_mean=0.876, sigma=0.008, direction="higher_better"),
        "latency_ms": KPIThreshold(baseline_mean=85.0, sigma=5.0, direction="lower_better"),
        "fairness_index": KPIThreshold(baseline_mean=0.91, sigma=0.02, direction="higher_better"),
    })

    # --- YOUR improvements over the paper ---
    min_sample_size: int = 30           # §5.3.1 fix: avoid small-sample noise
    ewma_alpha: float = 0.3            # §5.3.4 fix: smoothing factor
    adaptive_lambda: float = 0.7       # §5.3.3 fix: blend weight for thresholds
    significance_level: float = 0.05   # stat-test α (chi-squared / Mann-Whitney)

    # Timing
    evaluation_interval_sec: float = 30.0   # how often the Decision Engine runs
    stage_hold_sec: float = 120.0           # min time at each stage before promote

    # Routing
    routing_mode: RoutingMode = RoutingMode.AB

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / MLflow logging."""
        return {
            "traffic_stages": self.traffic_stages,
            "min_sample_size": self.min_sample_size,
            "ewma_alpha": self.ewma_alpha,
            "adaptive_lambda": self.adaptive_lambda,
            "significance_level": self.significance_level,
            "evaluation_interval_sec": self.evaluation_interval_sec,
            "stage_hold_sec": self.stage_hold_sec,
            "routing_mode": self.routing_mode.value,
            "kpi_thresholds": {
                name: {
                    "baseline_mean": t.baseline_mean,
                    "sigma": t.sigma,
                    "lower_bound": t.lower_bound,
                    "upper_bound": t.upper_bound,
                    "direction": t.direction,
                }
                for name, t in self.kpi_thresholds.items()
            },
        }
