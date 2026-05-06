# Prepare Notes

- Project track: Track 2
- Current focus: local environment and infrastructure scaffold
- Integrated baseline: FastAPI app, MLflow logging, Prometheus metrics, Grafana provisioning, GitHub Actions CI
- Next step: replace the demo scoring logic with the paper's actual model pipeline and dataset flow
- Paper note: progressive delivery for ML models focuses on the problem of safe rollout under drift, fairness, and rollback risk.
- Paper note: proposed solution is KPI-gated canary/shadow rollout with router, metrics collector, decision engine, and rollback controller.
- Paper gap: mostly conceptual; little concrete implementation detail, limited experimental validation, and no full end-to-end automated MLOps system.
