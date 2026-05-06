"""Interactive Live Demo — Watch the canary pipeline in real-time.

Uses REAL test data from the fraud dataset (not random noise) so
both models produce meaningful predictions and KPI comparisons.

HOW TO USE:
  1. Open 3 browser tabs:
     - Grafana:    http://localhost:3000  (admin/admin)
     - Prometheus: http://localhost:9090
     - MLflow:     http://localhost:5000
  2. Run this script:
     python -m mlops_canary_pipeline.live_demo
  3. Follow the prompts — press Enter to advance stages
"""

from __future__ import annotations

import sys
import threading
import time

import requests

BASE_URL = "http://localhost:8000"

# ── Shared state ─────────────────────────────────────────────────────────
_stop_traffic = threading.Event()
_traffic_stats = {"total": 0, "baseline": 0, "candidate": 0, "errors": 0}
_lock = threading.Lock()
_test_features: list[list[float]] = []
_test_labels: list[int] = []


def log(msg: str) -> None:
    print(f"\033[36m[{time.strftime('%H:%M:%S')}]\033[0m {msg}")


def log_header(msg: str) -> None:
    print(f"\n\033[1;33m{'═' * 60}\033[0m")
    print(f"\033[1;33m  {msg}\033[0m")
    print(f"\033[1;33m{'═' * 60}\033[0m\n")


def api_get(path: str) -> dict:
    r = requests.get(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str) -> dict:
    r = requests.post(f"{BASE_URL}{path}", timeout=300)
    r.raise_for_status()
    return r.json()


def load_real_test_data():
    """Load the REAL preprocessed test data so models get meaningful input."""
    log("Loading real fraud detection test data...")
    from mlops_canary_pipeline.data.ingest import load_dataset
    from mlops_canary_pipeline.data.preprocess import preprocess

    df = load_dataset()
    data = preprocess(df, apply_smote=False)

    # Use test set features and labels
    global _test_features, _test_labels
    _test_features = data.X_test.tolist()
    _test_labels = data.y_test.tolist()

    n_fraud = sum(_test_labels)
    log(f"  Loaded {len(_test_features)} test samples ({n_fraud} fraud, "
        f"{len(_test_features) - n_fraud} legit)")


# ── Background traffic sender ───────────────────────────────────────────
def _traffic_worker(requests_per_sec: float = 5.0):
    """Sends REAL test data continuously in the background."""
    interval = 1.0 / requests_per_sec
    idx = 0
    n = len(_test_features)

    while not _stop_traffic.is_set():
        try:
            # Pick a real test sample (cycle through dataset)
            features = _test_features[idx % n]
            gt = _test_labels[idx % n]
            idx += 1

            r = requests.post(
                f"{BASE_URL}/predict",
                json={"features": features},
                timeout=5,
            )
            result = r.json()

            # Send REAL ground truth (not random)
            requests.post(
                f"{BASE_URL}/feedback",
                json={"ground_truth": gt, "model_version": result["model_version"]},
                timeout=5,
            )
            with _lock:
                _traffic_stats["total"] += 1
                _traffic_stats[result["model_version"]] += 1
        except Exception:
            with _lock:
                _traffic_stats["errors"] += 1
        time.sleep(interval)


def start_traffic(rps: float = 5.0) -> None:
    _stop_traffic.clear()
    t = threading.Thread(target=_traffic_worker, args=(rps,), daemon=True)
    t.start()
    log(f"📡 Background traffic started ({rps} req/sec, using real test data)")


def stop_traffic():
    _stop_traffic.set()
    time.sleep(0.5)
    log("🛑 Background traffic stopped")


def show_traffic_stats():
    with _lock:
        stats = dict(_traffic_stats)
    total = stats["total"]
    if total == 0:
        return
    b_pct = stats["baseline"] / total * 100
    c_pct = stats["candidate"] / total * 100
    log(f"📊 Traffic: {total} total — baseline: {stats['baseline']} ({b_pct:.0f}%) | "
        f"candidate: {stats['candidate']} ({c_pct:.0f}%) | errors: {stats['errors']}")


def show_kpi_comparison():
    """Fetch and display current KPIs side-by-side."""
    try:
        status = api_get("/canary/status")
    except Exception as e:
        log(f"  ⚠️  Could not fetch status: {e}")
        return

    b = status["kpi_snapshots"]["baseline"]
    c = status["kpi_snapshots"]["candidate"]
    drift = status.get("drift", {})

    print()
    print(f"  \033[1m{'Metric':<22} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}\033[0m")
    print(f"  {'─' * 52}")
    for metric in ["precision", "recall", "f1", "latency_ms_mean", "fairness_index", "drift_score"]:
        bv = b.get(metric, 0)
        cv = c.get(metric, 0)
        delta = round(cv - bv, 4) if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) else "—"
        if metric in ("precision", "recall", "f1", "fairness_index"):
            color = "32" if cv >= bv else "31"
        elif metric == "latency_ms_mean":
            color = "32" if cv <= bv else "31"
        else:
            color = "37"
        print(f"  {metric:<22} {bv:>10} \033[{color}m{cv:>10}\033[0m {delta:>10}")

    print(f"\n  Samples:  baseline={b['n_samples']}  candidate={c['n_samples']}")
    if drift:
        print(f"  Drift:    baseline={drift.get('baseline_drift_score', 0):.4f}  "
              f"candidate={drift.get('candidate_drift_score', 0):.4f}  "
              f"detected={'⚠️  YES' if drift.get('candidate_drift_detected') else '✅ No'}")
    print()


def show_decision():
    """Trigger evaluation and show the result."""
    try:
        result = api_post("/canary/evaluate")
    except Exception as e:
        log(f"  ⚠️  Evaluation failed: {e}")
        return {}

    action = result["action"]
    if action == "promote":
        color = "32"
        icon = "🟢"
    elif action == "rollback":
        color = "31"
        icon = "🔴"
    else:
        color = "33"
        icon = "🟡"

    log(f"{icon} Decision: \033[1;{color}m{action.upper()}\033[0m")
    log(f"   Reason: {result['reason']}")
    log(f"   Chi² p-value: {result.get('chi2_p_value')}")
    log(f"   Mann-Whitney p: {result.get('mann_whitney_p_value')}")
    log(f"   Canary traffic: {result['canary_pct']}%")
    return result


def wait_for_enter(msg: str = "Press Enter to continue..."):
    print(f"\n\033[1;35m  ⏸  {msg}\033[0m", end="", flush=True)
    input()


def wait_and_collect(seconds: int = 18, interval: int = 3):
    """Send traffic for a period, printing stats every interval."""
    rounds = seconds // interval
    for _ in range(rounds):
        time.sleep(interval)
        show_traffic_stats()


# ── Main demo flow ──────────────────────────────────────────────────────
def main():
    log_header("LIVE CANARY DEPLOYMENT DEMO")
    log("Open these in your browser NOW:")
    log("  📊 Grafana:    http://localhost:3000  (admin / admin)")
    log("  📈 Prometheus: http://localhost:9090")
    log("  🧪 MLflow:     http://localhost:5000")

    # Health check
    try:
        health = api_get("/health")
        log(f"\n✅ App is healthy: {health['status']}")
    except Exception as e:
        log(f"\n❌ App not running: {e}")
        log("   Run: docker compose up --build -d")
        sys.exit(1)

    # Reset all state from any previous run
    reset = api_post("/canary/reset")
    log(f"🔄 {reset['message']}")

    # Load REAL test data
    load_real_test_data()

    # ── Phase 1: Baseline traffic ────────────────────────────────────
    log_header("PHASE 1: Baseline Traffic (no canary)")
    log("Sending REAL test data to baseline model only...")
    log("👀 Watch Grafana — you'll see request rate increase")

    start_traffic(rps=10)
    wait_and_collect(seconds=15)

    show_kpi_comparison()
    wait_for_enter("Ready to start canary? Press Enter...")

    # ── Phase 2: Start canary at 5% ─────────────────────────────────
    log_header("PHASE 2: Canary Started — 5% Traffic")
    result = api_post("/canary/start")
    log(f"🚀 Canary started! {result['canary_pct']}% → candidate model")
    log("👀 Watch Grafana — Traffic Split gauge should show 5%")
    log("⏳ Collecting data for candidate model...")

    wait_and_collect(seconds=24)

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to increase to 25%? Press Enter...")

    # ── Phase 3: Advance to 25% ─────────────────────────────────────
    log_header("PHASE 3: Canary at 25% Traffic")
    result = api_post("/canary/start")
    log(f"📈 Traffic increased to {result['canary_pct']}%")
    log("👀 Watch Grafana — more candidate data appearing")

    wait_and_collect(seconds=24)

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to increase to 50%? Press Enter...")

    # ── Phase 4: Advance to 50% ─────────────────────────────────────
    log_header("PHASE 4: Canary at 50% Traffic")
    result = api_post("/canary/start")
    log(f"📈 Traffic increased to {result['canary_pct']}%")

    wait_and_collect(seconds=24)

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to go to 100% (full promotion)? Press Enter...")

    # ── Phase 5: Full promotion or rollback ──────────────────────────
    log_header("PHASE 5: Final Decision — 100% or Rollback")
    result = api_post("/canary/start")
    log(f"📈 Traffic at {result['canary_pct']}%")

    wait_and_collect(seconds=24)

    show_kpi_comparison()
    log("\n⚖️  Final KPI evaluation...")
    show_decision()

    # ── Phase 6: Drift check ─────────────────────────────────────────
    log_header("PHASE 6: Drift Detection Check")
    drift = api_get("/drift/status")
    log(f"Baseline drift score:  {drift['baseline_drift_score']:.4f}")
    log(f"Candidate drift score: {drift['candidate_drift_score']:.4f}")
    log(f"Drift detected: {'⚠️  YES' if drift['baseline_drift_detected'] or drift['candidate_drift_detected'] else '✅ No'}")

    wait_for_enter("Check drift-and-retrain? Press Enter...")

    retrain = api_post("/drift/check-and-retrain")
    log(f"Drift action: {retrain['action']}")
    if retrain['drift_detected']:
        log("🔄 System rolled back and retrained automatically!")
    else:
        log("✅ No drift — models are stable")

    # ── Cleanup ──────────────────────────────────────────────────────
    stop_traffic()

    log_header("DEMO COMPLETE")
    show_traffic_stats()
    final_status = api_get("/canary/status")
    engine = final_status["decision_engine"]
    log(f"Total evaluations: {engine['total_evaluations']}")
    log(f"Decisions: {engine['action_counts']}")
    log(f"Final canary %:   {final_status['router']['canary_pct']}%")
    if final_status.get("ttd_seconds"):
        log(f"Time-to-Detect:   {final_status['ttd_seconds']}s")
    log("\n✅ All done! Check your Grafana dashboard for the full picture.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_traffic()
        log("\n\n👋 Demo stopped by user")
