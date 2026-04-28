"""Tests for the canary Decision Engine and KPI module."""

import pytest

from mlops_canary_pipeline.canary.config import CanaryConfig, DeploymentAction, KPIThreshold
from mlops_canary_pipeline.canary.decision_engine import DecisionEngine
from mlops_canary_pipeline.canary.kpi import KPICollector, KPISample
from mlops_canary_pipeline.canary.rollback import RollbackController


@pytest.fixture
def config():
    return CanaryConfig(min_sample_size=5)  # lower for testing


@pytest.fixture
def collector(config):
    return KPICollector(config=config, window=500)


@pytest.fixture
def engine(collector, config):
    return DecisionEngine(collector=collector, config=config)


# ─── KPIThreshold ──────────────────────────────────────────────────────

class TestKPIThreshold:
    def test_bounds(self):
        t = KPIThreshold(baseline_mean=0.918, sigma=0.006)
        assert t.lower_bound == pytest.approx(0.906)
        assert t.upper_bound == pytest.approx(0.930)

    def test_higher_better_acceptable(self):
        t = KPIThreshold(baseline_mean=0.918, sigma=0.006, direction="higher_better")
        assert t.is_acceptable(0.920) is True
        assert t.is_acceptable(0.900) is False

    def test_lower_better_acceptable(self):
        t = KPIThreshold(baseline_mean=85.0, sigma=5.0, direction="lower_better")
        assert t.is_acceptable(80.0) is True
        assert t.is_acceptable(100.0) is False


# ─── KPICollector ──────────────────────────────────────────────────────

class TestKPICollector:
    def test_record_and_snapshot(self, collector):
        for i in range(20):
            sample = KPISample(
                model_version="baseline",
                prediction=1 if i % 3 == 0 else 0,
                ground_truth=1 if i % 4 == 0 else 0,
                latency_ms=10.0 + i * 0.5,
            )
            collector.record(sample)

        snap = collector.snapshot("baseline")
        assert snap.n_samples == 20
        assert 0 <= snap.precision <= 1
        assert 0 <= snap.recall <= 1

    def test_ewma_smoothing(self, collector):
        v1 = collector.ewma_smooth("baseline", "precision", 0.9)
        assert v1 == 0.9  # first value = raw

        v2 = collector.ewma_smooth("baseline", "precision", 0.8)
        # alpha=0.3: 0.3*0.8 + 0.7*0.9 = 0.87
        assert v2 == pytest.approx(0.87, abs=0.01)

    def test_chi_squared_insufficient_samples(self, collector):
        from mlops_canary_pipeline.canary.kpi import KPISnapshot
        b = KPISnapshot(n_samples=3, precision=0.9)
        c = KPISnapshot(n_samples=3, precision=0.8)
        _, p, sig = collector.chi_squared_test(b, c)
        assert sig is False  # too few samples

    def test_chi_squared_with_data(self, collector):
        from mlops_canary_pipeline.canary.kpi import KPISnapshot
        b = KPISnapshot(n_samples=100, precision=0.9)
        c = KPISnapshot(n_samples=100, precision=0.5)  # dramatically different
        chi2, p, sig = collector.chi_squared_test(b, c)
        assert chi2 > 0
        assert sig == True  # should be significant  # noqa: E712

    def test_ttd_measurement(self, collector):
        assert collector.ttd_seconds is None

        collector.mark_degradation_injected()
        import time
        time.sleep(0.05)
        collector.mark_degradation_detected()

        ttd = collector.ttd_seconds
        assert ttd is not None
        assert ttd >= 0.04


# ─── DecisionEngine ───────────────────────────────────────────────────

class TestDecisionEngine:
    def test_hold_when_no_data(self, engine):
        result = engine.evaluate()
        assert result.action == DeploymentAction.HOLD

    def test_hold_when_insufficient_samples(self, engine, collector):
        # Add only 3 candidate samples (below min_sample_size=5)
        for i in range(3):
            collector.record(KPISample(
                model_version="candidate", prediction=1,
                ground_truth=1, latency_ms=10.0,
            ))
        result = engine.evaluate()
        assert result.action == DeploymentAction.HOLD
        assert "Insufficient" in result.reason


# ─── RollbackController ──────────────────────────────────────────────

class TestRollbackController:
    def test_manual_rollback(self):
        from mlops_canary_pipeline.canary.router import ModelRouter
        router = ModelRouter()
        router._canary_pct = 25
        controller = RollbackController()

        event = controller.manual_rollback(router, reason="test rollback")
        assert router.canary_pct == 0
        assert controller.total_rollbacks == 1
        assert event.reason == "test rollback"
        assert event.canary_pct_before == 25
