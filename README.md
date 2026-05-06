# KPI-Gated Progressive Delivery MLOps Pipeline

![CI/CD Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps pipeline implementing and extending the research from Gaddam (2023) on KPI-gated canary deployments. This project provides a fully automated, closed-loop system for safely deploying, monitoring, and rolling back Machine Learning models in production.

---

## 🌟 Research Novelty & Contributions

This project builds upon the foundational concepts of KPI-gated deployments but actively implements solutions for the limitations identified in the original paper:

1. **Statistical Significance Gates:** Added Chi-squared (for categorical metrics like precision) and Mann-Whitney U (for continuous metrics like latency) tests to prevent premature rollbacks caused by small-sample noise (§5.3.1 fix).
2. **Adaptive Thresholds:** Implemented dynamic KPI acceptance bands blending static baseline metrics with recent percentiles using a tunable λ weight (§5.3.3 fix).
3. **Alert Fatigue Reduction:** Applied Exponentially Weighted Moving Averages (EWMA) to smooth KPI volatility before Decision Engine evaluation (§5.3.4 fix).
4. **Drift Detection & Fairness:** Fully implemented Kolmogorov-Smirnov (KS) distribution drift detection and an Equality of Opportunity fairness index—concepts discussed but not implemented in the base research.
5. **Closed MLOps Loop:** Automated the cycle of Drift Detection → Rollback → Retrain → Redeploy.

## 🏗️ Architecture

The system is containerized via Docker and orchestrated using Docker Compose.

* **FastAPI Application:** Serves the ML models, handles routing (Canary vs. Baseline), and computes real-time KPIs.
* **Decision Engine:** Evaluates live traffic KPIs against adaptive thresholds to automate `PROMOTE`, `HOLD`, or `ROLLBACK` actions.
* **MLflow:** Tracks experiments, hyperparameters, and evaluation metrics during the model training phase.
* **Prometheus:** Scrapes real-time application metrics (latency, traffic split, KPI snapshots, drift scores).
* **Grafana:** Visualizes the Prometheus metrics on a pre-provisioned dashboard for instant observability.
* **GitHub Actions:** Provides Continuous Integration (CI), running `ruff` linting and `pytest` test suites on every push.

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* Docker & Docker Compose
* Git

### 1. Installation & Training

First, clone the repository and set up your virtual environment.

```bash
# Set up virtual environment
python3 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline (Generates synthetic data, trains XGBoost & LightGBM)
python -m mlops_canary_pipeline.training.train
```

*Note: The training script automatically generates a realistic, highly-imbalanced synthetic FinTech fraud dataset (~100k rows, 2% fraud).*

### 2. Start the Infrastructure

Launch the FastAPI application alongside the observability stack (Prometheus, Grafana, MLflow).

```bash
docker compose up --build -d
```

**Access Points:**
* **FastAPI App:** `http://localhost:8000`
* **Grafana Dashboard:** `http://localhost:3000` *(Login: admin / admin)*
* **Prometheus:** `http://localhost:9090`
* **MLflow Tracking:** `http://localhost:5000`

### 3. Run the Live Demo

The project includes an interactive CLI demo that simulates background traffic (using the actual test dataset) and walks you through the canary deployment phases.

```bash
python -m mlops_canary_pipeline.live_demo
```

**What the demo does:**
1. **Phase 1 (Baseline):** Sends traffic exclusively to the XGBoost baseline.
2. **Phase 2-4 (Canary):** Gradually increases traffic to the LightGBM candidate (5% → 25% → 50%).
3. **Phase 5 (Decision):** Evaluates EWMA-smoothed KPIs. If the candidate degrades performance significantly, it triggers an immediate rollback. If it performs well, it promotes to 100%.
4. **Phase 6 (Drift & Retrain):** Checks for data distribution drift via KS testing. If drift is detected, the system automatically rolls back to a safe state and triggers the training pipeline to fit new models.

## 📁 Repository Structure

```text
.
├── .github/workflows/       # CI/CD pipelines
├── data/                    # Generated datasets
├── docker/                  # Dockerfiles
├── grafana/                 # Pre-provisioned dashboards & datasources
├── models/                  # Serialized baseline/candidate models
├── prometheus/              # Prometheus scraping config
├── src/mlops_canary_pipeline/
│   ├── canary/              # KPI collector, Router, Decision Engine, Rollback
│   ├── data/                # Data ingestion & preprocessing (SMOTE)
│   ├── training/            # MLflow training & evaluation scripts
│   ├── experiment.py        # Automated end-to-end experiment script
│   ├── live_demo.py         # Interactive presentation script
│   └── main.py              # FastAPI application
├── tests/                   # Pytest unit tests
├── docker-compose.yml       # Infrastructure orchestration
└── requirements.txt         # Python dependencies
```

## 🧪 Evaluation Metrics

The Decision Engine evaluates models based on the following KPIs:
* **Precision & Recall** (Higher is better)
* **Latency (ms)** (Lower is better; evaluated via Mann-Whitney U test)
* **Fairness Index** (Equality of Opportunity ratio between feature subgroups)
* **Drift Score** (Kolmogorov-Smirnov statistic against reference distribution)

## 📄 License
This project is licensed under the MIT License.
