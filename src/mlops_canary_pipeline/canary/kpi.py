"""KPI computation and statistical testing.

This module contains the **core research contributions** that improve on
Gaddam (2023):

1. **Chi-squared significance test** (§5.3.1 fix)
   — prevents premature promote/rollback when sample size is small.
2. **Mann-Whitney U test** for latency comparison (continuous metric).
3. **EWMA smoothing** (§5.3.4 fix) — dampens metric oscillations before
   the Decision Engine evaluates, reducing alert fatigue.
4. **Percentile-based adaptive thresholds** (§5.3.3 fix) — dynamically
   recomputes acceptance bands from the recent Prometheus window instead
   of relying on static ±2σ.
5. **Time-to-Detect (TTD)** measurement — new research metric absent from
   the paper.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mlops_canary_pipeline.canary.config import CanaryConfig, KPIThreshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KPISample:
    """A single observation pushed by the app after each prediction."""
    model_version: str       # "baseline" | "candidate"
    prediction: int
    ground_truth: int | None = None   # arrives via /feedback
    probability: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class KPISnapshot:
    """Aggregated KPIs for one model over a collection window."""
    n_samples: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    fairness_index: float = 0.0


# ---------------------------------------------------------------------------
# KPI Collector
# ---------------------------------------------------------------------------

class KPICollector:
    """Collects per-request samples and computes aggregated KPI snapshots.

    Maintains separate ring-buffers for baseline and candidate.
    """

    def __init__(self, config: CanaryConfig | None = None, window: int = 2000):
        self.config = config or CanaryConfig()
        self._window = window
        self._baseline_samples: deque[KPISample] = deque(maxlen=window)
        self._candidate_samples: deque[KPISample] = deque(maxlen=window)

        # EWMA state (one per KPI per model)
        self._ewma: dict[str, dict[str, float]] = {
            "baseline": {},
            "candidate": {},
        }

        # TTD tracking
        self._degradation_injected_at: float | None = None
        self._degradation_detected_at: float | None = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def record(self, sample: KPISample) -> None:
        """Push a new observation."""
        buf = self._candidate_samples if sample.model_version == "candidate" else self._baseline_samples
        buf.append(sample)

    def record_ground_truth(self, request_idx: int, label: int, model_version: str) -> None:
        """Attach a ground-truth label to the most-recent matching sample."""
        buf = self._candidate_samples if model_version == "candidate" else self._baseline_samples
        # Walk backwards — ground truth usually arrives shortly after prediction
        for s in reversed(buf):
            if s.ground_truth is None:
                s.ground_truth = label
                return

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def snapshot(self, model_version: str) -> KPISnapshot:
        """Compute the current KPI snapshot for *model_version*."""
        buf = self._candidate_samples if model_version == "candidate" else self._baseline_samples
        labelled = [s for s in buf if s.ground_truth is not None]

        if not labelled:
            return KPISnapshot()

        y_true = np.array([s.ground_truth for s in labelled])
        y_pred = np.array([s.prediction for s in labelled])
        latencies = np.array([s.latency_ms for s in buf])

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        return KPISnapshot(
            n_samples=len(labelled),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            latency_ms_mean=round(float(latencies.mean()), 3) if len(latencies) else 0.0,
            latency_ms_p95=round(float(np.percentile(latencies, 95)), 3) if len(latencies) else 0.0,
            fairness_index=1.0,  # simplified — full fairness requires subgroup labels
        )

    # ------------------------------------------------------------------
    # EWMA smoothing  (YOUR improvement — §5.3.4 fix)
    # ------------------------------------------------------------------

    def ewma_smooth(self, model_version: str, kpi_name: str, raw_value: float) -> float:
        """Apply Exponentially Weighted Moving Average to a KPI value.

        This addresses the paper's limitation §5.3.4: Prometheus fired too
        many alerts during normal oscillations.  EWMA smooths out transient
        spikes so the Decision Engine evaluates *trends* not noise.
        """
        alpha = self.config.ewma_alpha
        prev = self._ewma[model_version].get(kpi_name)
        if prev is None:
            smoothed = raw_value
        else:
            smoothed = alpha * raw_value + (1 - alpha) * prev
        self._ewma[model_version][kpi_name] = smoothed
        return round(smoothed, 6)

    # ------------------------------------------------------------------
    # Statistical tests  (YOUR improvement — absent in paper)
    # ------------------------------------------------------------------

    def chi_squared_test(
        self,
        baseline_snap: KPISnapshot,
        candidate_snap: KPISnapshot,
    ) -> tuple[float, float, bool]:
        """Chi-squared test for categorical KPI comparison.

        Improvement over Gaddam (2023) §5.3.1
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The paper used raw KPI values at 5-10% traffic, causing false
        alarms from small-sample noise.  We require ``min_sample_size``
        labelled samples *and* a statistically significant χ² result
        before allowing a promotion or rollback decision.

        Returns
        -------
        chi2 : float
        p_value : float
        significant : bool
            True when p < significance_level.
        """
        n_b = baseline_snap.n_samples
        n_c = candidate_snap.n_samples

        if n_b < self.config.min_sample_size or n_c < self.config.min_sample_size:
            logger.info(
                "Chi-squared skipped: samples too low (baseline=%d, candidate=%d, min=%d)",
                n_b, n_c, self.config.min_sample_size,
            )
            return 0.0, 1.0, False

        # Build 2x2 contingency table from precision (TP vs FP)
        tp_b = int(round(baseline_snap.precision * n_b))
        fp_b = n_b - tp_b
        tp_c = int(round(candidate_snap.precision * n_c))
        fp_c = n_c - tp_c

        table = np.array([[tp_b, fp_b], [tp_c, fp_c]])
        # Ensure no zero columns (add small correction)
        if table.min() == 0:
            table = table + 1

        chi2, p, _, _ = stats.chi2_contingency(table)
        significant = p < self.config.significance_level
        logger.info("Chi-squared: χ²=%.4f  p=%.6f  significant=%s", chi2, p, significant)
        return float(chi2), float(p), significant

    def mann_whitney_test(self, model_a: str = "baseline", model_b: str = "candidate") -> tuple[float, float, bool]:
        """Mann-Whitney U test for latency distributions.

        Improvement over Gaddam (2023)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The paper compared latency via simple mean comparison.  We use
        Mann-Whitney U — a non-parametric test that handles skewed latency
        distributions correctly.

        Returns
        -------
        u_stat : float
        p_value : float
        significant : bool
        """
        lat_a = np.array([s.latency_ms for s in (self._baseline_samples if model_a == "baseline" else self._candidate_samples)])
        lat_b = np.array([s.latency_ms for s in (self._candidate_samples if model_b == "candidate" else self._baseline_samples)])

        if len(lat_a) < self.config.min_sample_size or len(lat_b) < self.config.min_sample_size:
            return 0.0, 1.0, False

        u_stat, p = stats.mannwhitneyu(lat_a, lat_b, alternative="two-sided")
        significant = p < self.config.significance_level
        logger.info("Mann-Whitney U: U=%.2f  p=%.6f  significant=%s", u_stat, p, significant)
        return float(u_stat), float(p), significant

    # ------------------------------------------------------------------
    # Adaptive thresholds  (YOUR improvement — §5.3.3 fix)
    # ------------------------------------------------------------------

    def adaptive_threshold(self, kpi_name: str) -> KPIThreshold:
        """Compute an adaptive threshold from the recent baseline window.

        Improvement over Gaddam (2023) §5.3.3
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The paper used static ±2σ thresholds that required manual
        recalibration.  We blend the static baseline with percentile-based
        bounds from recent observations using λ:

            threshold = λ * recent_percentile  +  (1-λ) * static_baseline
        """
        static = self.config.kpi_thresholds.get(kpi_name)
        if static is None:
            raise KeyError(f"Unknown KPI: {kpi_name}")

        baseline_buf = self._baseline_samples
        if len(baseline_buf) < self.config.min_sample_size:
            return static  # not enough data — fall back to static

        # Gather the KPI values from baseline buffer
        if kpi_name == "latency_ms":
            values = np.array([s.latency_ms for s in baseline_buf])
        elif kpi_name == "precision":
            labelled = [s for s in baseline_buf if s.ground_truth is not None]
            if len(labelled) < self.config.min_sample_size:
                return static
            y_t = np.array([s.ground_truth for s in labelled])
            y_p = np.array([s.prediction for s in labelled])
            # Rolling precision in windows of 50
            window = 50
            values = []
            for i in range(0, len(y_t) - window + 1, window):
                tp = ((y_p[i:i+window] == 1) & (y_t[i:i+window] == 1)).sum()
                fp = ((y_p[i:i+window] == 1) & (y_t[i:i+window] == 0)).sum()
                values.append(tp / max(tp + fp, 1))
            values = np.array(values) if values else np.array([static.baseline_mean])
        else:
            return static  # unsupported KPI for adaptive

        lam = self.config.adaptive_lambda
        recent_mean = float(np.mean(values))
        recent_sigma = float(np.std(values)) if len(values) > 1 else static.sigma

        blended_mean = lam * recent_mean + (1 - lam) * static.baseline_mean
        blended_sigma = lam * recent_sigma + (1 - lam) * static.sigma

        return KPIThreshold(
            baseline_mean=round(blended_mean, 6),
            sigma=round(blended_sigma, 6),
            n_sigma=static.n_sigma,
            direction=static.direction,
        )

    # ------------------------------------------------------------------
    # TTD — Time to Detect  (YOUR improvement — new research metric)
    # ------------------------------------------------------------------

    def mark_degradation_injected(self) -> None:
        """Call when a known degradation is injected (for experiments)."""
        self._degradation_injected_at = time.time()
        self._degradation_detected_at = None
        logger.info("TTD clock started — degradation injected at %.2f", self._degradation_injected_at)

    def mark_degradation_detected(self) -> None:
        """Call when the Decision Engine detects the degradation."""
        if self._degradation_injected_at and self._degradation_detected_at is None:
            self._degradation_detected_at = time.time()
            logger.info("TTD clock stopped — degradation detected at %.2f", self._degradation_detected_at)

    @property
    def ttd_seconds(self) -> float | None:
        """Time-to-Detect in seconds, or None if not measured yet."""
        if self._degradation_injected_at and self._degradation_detected_at:
            return round(self._degradation_detected_at - self._degradation_injected_at, 3)
        return None

    # ------------------------------------------------------------------
    # Counts for Prometheus
    # ------------------------------------------------------------------

    @property
    def baseline_count(self) -> int:
        return len(self._baseline_samples)

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_samples)
