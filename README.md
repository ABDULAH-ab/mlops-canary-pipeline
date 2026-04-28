# MLOps Canary Pipeline

**KPI-Gated Progressive Delivery for ML Models**

An end-to-end MLOps pipeline that implements and extends the framework from
[Gaddam (2023)](https://doi.org/10.63282/3117-5481/AIJCST-V5I4P104) —
*"Progressive Delivery for Models with Quality KPIs"*.

## Research Contributions

This project improves on the base paper by addressing four specific limitations:

| Paper Limitation | Our Improvement |
|---|---|
| **§5.3.1** Statistical noise at early canary stages | Chi-squared significance test + minimum sample size gate |
| **§5.3.3** Static ±2σ thresholds need manual tuning | Adaptive thresholds via λ-weighted blending of recent window |
| **§5.3.4** Alert fatigue from Prometheus oscillations | EWMA smoothing + tiered Alertmanager rules (critical/warning/info) |
| **§5.3.2** Requires Kubernetes + cloud infrastructure | Runs entirely on Docker Compose — any 8GB RAM laptop |

**Additional contributions:**
- Mann-Whitney U test for latency distribution comparison
- Time-to-Detect (TTD) as a new research metric
- Full open-source reproducible codebase (paper describes concepts only)
- Grafana A vs B comparison dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD (GitHub Actions)                   │
│  Push → Lint → Test → Train → Evaluate → Build Docker → Deploy │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                        Serving Layer                            │
│  Client → Model Router ──┬── Baseline (XGBoost)                │
│                           └── Candidate (LightGBM)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                     Decision Engine                             │
│  KPI Collector → EWMA Smoothing → Adaptive Thresholds          │
│  → Chi-squared / Mann-Whitney → Promote / Hold / Rollback      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      Observability                              │
│  Prometheus (metrics + alert rules) → Grafana (KPI dashboards) │
│  MLflow (experiment tracking + model registry)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** | Model serving API + canary management endpoints |
| **MLflow** | Experiment tracking + model registry |
| **Prometheus** | Metrics collection + tiered alerting rules |
| **Grafana** | Real-time KPI dashboards (baseline vs candidate) |
| **Docker Compose** | Full local observability stack |
| **GitHub Actions** | CI/CD with train → evaluate → deploy pipeline |
| **XGBoost** | Baseline model (fraud detection) |
| **LightGBM** | Candidate model (fraud detection) |

## Quick Start

### 1. Setup environment

```bash
python -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev]"
```

### 2. Generate dataset & train models

```bash
python -m mlops_canary_pipeline.data.ingest
python -m mlops_canary_pipeline.training.train
python -m mlops_canary_pipeline.training.evaluate
```

### 3. Run locally

```bash
uvicorn mlops_canary_pipeline.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Or start the full observability stack

```bash
docker compose up --build
```

### 5. Run a canary rollout

```bash
# Start canary at 5% traffic
curl -X POST http://localhost:8000/canary/start

# Send predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.5, 0.3, 1.2, -0.8, 0.6, -1.1, 0.4, 0.9, -0.2, 0.7, -0.3, 1.5, -0.6, 0.2, 0.8, -1.0, 0.5, -0.4, 1.1, 0.3, -0.7, 0.6, 0.1, -0.9, 0.4, 1.3, -0.5, 50.0]}'

# Send ground truth feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"ground_truth": 0, "model_version": "baseline"}'

# Trigger KPI evaluation
curl -X POST http://localhost:8000/canary/evaluate

# Check status
curl http://localhost:8000/canary/status | python -m json.tool
```

## Service Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Model inference (routed through canary) |
| `/feedback` | POST | Ground truth label for KPI computation |
| `/metrics` | GET | Prometheus metrics |
| `/canary/start` | POST | Start canary rollout at first stage |
| `/canary/configure` | POST | Set traffic split % |
| `/canary/status` | GET | Current canary state + KPI snapshots |
| `/canary/evaluate` | POST | Trigger KPI evaluation cycle |
| `/canary/rollback` | POST | Manual rollback to baseline |

## Dashboards

| URL | Dashboard |
|---|---|
| http://localhost:8000 | FastAPI service |
| http://localhost:5000 | MLflow experiment tracking |
| http://localhost:9090 | Prometheus metrics |
| http://localhost:3000 | Grafana (admin/admin) |

## Running Tests

```bash
pytest -v
```

## Project Structure

```
├── .github/workflows/ci.yml       # CI/CD pipeline
├── data/                           # Generated dataset (gitignored)
├── models/                         # Trained model artifacts (gitignored)
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── mlops-overview.json
│   │   │   └── canary-deployment.json
│   │   └── provisioning/
│   └── prometheus/
│       ├── prometheus.yml
│       └── alert_rules.yml
├── src/mlops_canary_pipeline/
│   ├── main.py                     # FastAPI app
│   ├── canary/
│   │   ├── config.py               # KPI thresholds + canary config
│   │   ├── decision_engine.py      # KPI-gated promotion logic
│   │   ├── kpi.py                  # Statistical tests + EWMA + adaptive thresholds
│   │   ├── rollback.py             # Rollback controller
│   │   └── router.py               # Traffic splitting
│   ├── data/
│   │   ├── ingest.py               # Synthetic fraud dataset
│   │   └── preprocess.py           # Scaling + SMOTE
│   └── training/
│       ├── train.py                # XGBoost + LightGBM training
│       ├── evaluate.py             # KPI evaluation
│       └── register.py             # MLflow Model Registry
├── tests/
│   ├── test_app.py
│   ├── test_canary.py
│   └── test_data.py
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Reference

Gaddam, R. R. (2023). *Progressive Delivery for Models with Quality KPIs*.
American International Journal of Computer Science and Technology, 5(4), 33–47.
https://doi.org/10.63282/3117-5481/AIJCST-V5I4P104
