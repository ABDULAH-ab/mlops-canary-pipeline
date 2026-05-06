"""Interactive Live Demo — Watch the canary pipeline in real-time.

This script lets you:
  1. Send continuous traffic while watching Grafana/Prometheus
  2. Press Enter to advance canary stages (5% → 25% → 50% → 100%)
  3. See KPI comparisons update live in the terminal
  4. Watch the Decision Engine promote or rollback automatically

HOW TO USE:
  1. Open 3 browser tabs:
     - Grafana:    http://localhost:3000  (admin/admin)
     - Prometheus: http://localhost:9090
     - MLflow:     http://localhost:5000
  2. Run this script:
     python -m mlops_canary_pipeline.live_demo
  3. Follow the prompts — press Enter to advance stages

The script sends traffic continuously so you can watch
dashboards update in real-time.
"""

from __future__ import annotations

import random
import sys
import threading
import time

import requests

BASE_URL = "http://localhost:8000"

# ── Shared state ─────────────────────────────────────────────────────────
_stop_traffic = threading.Event()
_traffic_stats = {"total": 0, "baseline": 0, "candidate": 0, "errors": 0}
_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"\033[36m[{time.strftime('%H:%M:%S')}]\033[0m {msg}")


def log_header(msg: str) -> None:
    print(f"\n\033[1;33m{'═' * 60}\033[0m")
    print(f"\033[1;33m  {msg}\033[0m")
    print(f"\033[1;33m{'═' * 60}\033[0m\n")


def log_kpi(label: str, value, color: str = "32") -> None:
    print(f"  \033[{color}m{label:<25}\033[0m {value}")


def random_features() -> list[float]:
    return [round(random.gauss(0, 1), 3) for _ in range(28)] + [round(random.uniform(1, 500), 2)]


def api_get(path: str) -> dict:
    r = requests.get(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str) -> dict:
    r = requests.post(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


# ── Background traffic sender ───────────────────────────────────────────
def _traffic_worker(requests_per_sec: float = 5.0):
    """Sends traffic continuously in the background."""
    interval = 1.0 / requests_per_sec
    while not _stop_traffic.is_set():
        try:
            features = random_features()
            r = requests.post(
                f"{BASE_URL}/predict",
                json={"features": features},
                timeout=5,
            )
            data = r.json()
            # Send ground truth feedback (2% fraud rate)
            gt = 1 if random.random() < 0.02 else 0
            requests.post(
                f"{BASE_URL}/feedback",
                json={"ground_truth": gt, "model_version": data["model_version"]},
                timeout=5,
            )
            with _lock:
                _traffic_stats["total"] += 1
                _traffic_stats[data["model_version"]] += 1
        except Exception:
            with _lock:
                _traffic_stats["errors"] += 1
        time.sleep(interval)


def start_traffic(rps: float = 5.0) -> threading.Thread:
    _stop_traffic.clear()
    t = threading.Thread(target=_traffic_worker, args=(rps,), daemon=True)
    t.start()
    log(f"📡 Background traffic started ({rps} req/sec)")
    return t


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
        # Color: green if candidate better, red if worse (for higher_better metrics)
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
        return result if 'result' in dir() else {}

    action = result["action"]
    if action == "promote":
        color = "32"  # green
        icon = "🟢"
    elif action == "rollback":
        color = "31"  # red
        icon = "🔴"
    else:
        color = "33"  # yellow
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

    # ── Phase 1: Baseline traffic ────────────────────────────────────
    log_header("PHASE 1: Baseline Traffic (no canary)")
    log("Sending traffic to baseline model only...")
    log("👀 Watch Grafana — you'll see request rate increase")

    start_traffic(rps=8)
    for i in range(5):
        time.sleep(3)
        show_traffic_stats()

    show_kpi_comparison()
    wait_for_enter("Ready to start canary? Press Enter...")

    # ── Phase 2: Start canary at 5% ─────────────────────────────────
    log_header("PHASE 2: Canary Started — 5% Traffic")
    result = api_post("/canary/start")
    log(f"🚀 Canary started! {result['canary_pct']}% → candidate model")
    log("👀 Watch Grafana — Traffic Split gauge should show 5%")

    for i in range(5):
        time.sleep(3)
        show_traffic_stats()

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to increase to 25%? Press Enter...")

    # ── Phase 3: Advance to 25% ─────────────────────────────────────
    log_header("PHASE 3: Canary at 25% Traffic")
    result = api_post("/canary/start")
    log(f"📈 Traffic increased to {result['canary_pct']}%")
    log("👀 Watch Grafana — more candidate data appearing")

    for i in range(6):
        time.sleep(3)
        show_traffic_stats()

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to increase to 50%? Press Enter...")

    # ── Phase 4: Advance to 50% ─────────────────────────────────────
    log_header("PHASE 4: Canary at 50% Traffic")
    result = api_post("/canary/start")
    log(f"📈 Traffic increased to {result['canary_pct']}%")

    for i in range(6):
        time.sleep(3)
        show_traffic_stats()

    show_kpi_comparison()
    log("\n⚖️  Running KPI evaluation...")
    show_decision()
    wait_for_enter("Ready to go to 100% (full promotion)? Press Enter...")

    # ── Phase 5: Full promotion or rollback ──────────────────────────
    log_header("PHASE 5: Final Decision — 100% or Rollback")
    result = api_post("/canary/start")
    log(f"📈 Traffic at {result['canary_pct']}%")

    for i in range(6):
        time.sleep(3)
        show_traffic_stats()

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
