"""Degradation Experiment — End-to-End Canary Demo Script.

This script demonstrates the FULL MLOps canary pipeline by:
1. Sending normal traffic → baseline handles everything
2. Starting canary → candidate gets 5% traffic
3. Sending predictions + ground truth feedback → KPIs are computed
4. Triggering evaluation → Decision Engine decides promote/hold/rollback
5. Checking drift detection status
6. Measuring Time-to-Detect (TTD)

Run:
    python -m mlops_canary_pipeline.experiment

Requires the app to be running:
    uvicorn mlops_canary_pipeline.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import random
import sys
import time

import requests

BASE_URL = "http://localhost:8000"
N_NORMAL_REQUESTS = 60      # normal traffic phase
N_CANARY_REQUESTS = 100     # canary traffic phase


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def random_features() -> list[float]:
    """Generate a random 29-feature vector."""
    return [round(random.gauss(0, 1), 3) for _ in range(28)] + [round(random.uniform(1, 500), 2)]


def send_prediction(features: list[float] | None = None) -> dict:
    features = features or random_features()
    r = requests.post(f"{BASE_URL}/predict", json={"features": features}, timeout=10)
    r.raise_for_status()
    return r.json()


def send_feedback(ground_truth: int, model_version: str) -> None:
    requests.post(
        f"{BASE_URL}/feedback",
        json={"ground_truth": ground_truth, "model_version": model_version},
        timeout=10,
    )


def api_get(path: str) -> dict:
    r = requests.get(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str) -> dict:
    r = requests.post(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    log("=" * 60)
    log("MLOps Canary Pipeline — End-to-End Experiment")
    log("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Health check
    # ------------------------------------------------------------------
    log("\n📡 Step 0: Health check")
    try:
        health = api_get("/health")
        log(f"  App status: {health['status']}")
    except Exception as e:
        log("  ❌ App not running! Start it first: uvicorn mlops_canary_pipeline.main:app")
        log(f"  Error: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Normal traffic (baseline only)
    # ------------------------------------------------------------------
    log(f"\n📊 Step 1: Sending {N_NORMAL_REQUESTS} normal requests (baseline only)")
    for i in range(N_NORMAL_REQUESTS):
        result = send_prediction()
        # Send ground truth (simulate: 2% fraud rate)
        gt = 1 if random.random() < 0.02 else 0
        send_feedback(gt, result["model_version"])

        if (i + 1) % 20 == 0:
            log(f"  Sent {i+1}/{N_NORMAL_REQUESTS} requests")

    log("  ✅ Baseline traffic complete")

    # ------------------------------------------------------------------
    # Step 2: Check initial status
    # ------------------------------------------------------------------
    log("\n📈 Step 2: Initial KPI status")
    status = api_get("/canary/status")
    b_snap = status["kpi_snapshots"]["baseline"]
    log(f"  Baseline — precision: {b_snap['precision']}, recall: {b_snap['recall']}, "
        f"f1: {b_snap['f1']}, fairness: {b_snap['fairness_index']}")
    log(f"  Drift score: {b_snap['drift_score']}, detected: {b_snap['drift_detected']}")

    # ------------------------------------------------------------------
    # Step 3: Start canary deployment
    # ------------------------------------------------------------------
    log("\n🚀 Step 3: Starting canary deployment")
    start_result = api_post("/canary/start")
    log(f"  Canary started at {start_result['canary_pct']}% traffic")

    # ------------------------------------------------------------------
    # Step 4: Canary traffic (mixed baseline + candidate)
    # ------------------------------------------------------------------
    log(f"\n🔄 Step 4: Sending {N_CANARY_REQUESTS} canary requests (mixed traffic)")
    model_counts = {"baseline": 0, "candidate": 0}
    for i in range(N_CANARY_REQUESTS):
        result = send_prediction()
        model_counts[result["model_version"]] = model_counts.get(result["model_version"], 0) + 1

        # Send ground truth
        gt = 1 if random.random() < 0.02 else 0
        send_feedback(gt, result["model_version"])

        if (i + 1) % 25 == 0:
            log(f"  Sent {i+1}/{N_CANARY_REQUESTS} — "
                f"baseline: {model_counts['baseline']}, candidate: {model_counts['candidate']}")

    log(f"  ✅ Canary traffic complete — {model_counts}")

    # ------------------------------------------------------------------
    # Step 5: KPI Evaluation
    # ------------------------------------------------------------------
    log("\n⚖️  Step 5: Triggering KPI evaluation")
    eval_result = api_post("/canary/evaluate")
    log(f"  Decision: {eval_result['action']}")
    log(f"  Reason: {eval_result['reason']}")
    log(f"  KPI deltas: {eval_result['kpi_deltas']}")
    log(f"  Chi² p-value: {eval_result['chi2_p_value']}")
    log(f"  Mann-Whitney p-value: {eval_result['mann_whitney_p_value']}")
    log(f"  Canary traffic: {eval_result['canary_pct']}%")

    # ------------------------------------------------------------------
    # Step 6: Drift detection
    # ------------------------------------------------------------------
    log("\n🔍 Step 6: Drift detection status")
    drift = api_get("/drift/status")
    log(f"  Baseline drift:  score={drift['baseline_drift_score']}, detected={drift['baseline_drift_detected']}")
    log(f"  Candidate drift: score={drift['candidate_drift_score']}, detected={drift['candidate_drift_detected']}")

    # ------------------------------------------------------------------
    # Step 7: Full status report
    # ------------------------------------------------------------------
    log("\n📋 Step 7: Full canary status")
    status = api_get("/canary/status")
    c_snap = status["kpi_snapshots"]["candidate"]
    b_snap = status["kpi_snapshots"]["baseline"]

    log(f"\n  {'Metric':<20} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}")
    log(f"  {'-'*50}")
    for metric in ["precision", "recall", "f1", "latency_ms_mean", "fairness_index", "drift_score"]:
        b_val = b_snap.get(metric, 0)
        c_val = c_snap.get(metric, 0)
        delta = round(c_val - b_val, 4) if isinstance(b_val, (int, float)) else "N/A"
        log(f"  {metric:<20} {b_val:>10} {c_val:>10} {delta:>10}")

    log(f"\n  Decision engine: {status['decision_engine']}")
    log(f"  Rollback controller: {status['rollback_controller']}")
    if status.get("ttd_seconds"):
        log(f"  Time-to-Detect: {status['ttd_seconds']}s")

    # ------------------------------------------------------------------
    # Step 8: Check drift-and-retrain endpoint
    # ------------------------------------------------------------------
    log("\n🔄 Step 8: Check drift-and-retrain")
    retrain_result = api_post("/drift/check-and-retrain")
    log(f"  Drift detected: {retrain_result['drift_detected']}")
    log(f"  Action taken: {retrain_result['action']}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("EXPERIMENT COMPLETE")
    log("=" * 60)
    log(f"  Total predictions:  {N_NORMAL_REQUESTS + N_CANARY_REQUESTS}")
    log(f"  Decision:           {eval_result['action']}")
    log(f"  Final canary %:     {eval_result['canary_pct']}%")
    log(f"  Drift detected:     {retrain_result['drift_detected']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
