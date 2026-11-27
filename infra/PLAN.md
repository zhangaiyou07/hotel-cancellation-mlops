# Plan: dashboards, CI/CD and integration (`feature/dashboards-cicd`)

**Owner:** Zijia Zhang (@zhangaiyou07)

## Goal
Make the whole system start with one `docker compose up`, make its health visible in Grafana, and ship
Docker images automatically on every merge to `main`.

## Tasks
1. **docker-compose skeleton**: `docker-compose.yml` with the shared services (PostgreSQL, MLflow server,
   Prometheus, Grafana) plus a `.env.example`. Each teammate then adds their own service (API, frontend,
   monitoring job, Prefect) to the same file. Final end-to-end integration run.
2. **Prometheus**: `infra/prometheus/prometheus.yml` scrapes the model API's `/metrics` endpoint.
3. **Grafana dashboards as code**: provisioned data sources (Prometheus and PostgreSQL) and a dashboard JSON
   in `infra/grafana/` with request rate, latency, error rate and drift-score panels.
4. **Alerts**: a drift-score threshold alert that fires a webhook to trigger the Prefect retraining flow.
5. **CD**: a GitHub Actions workflow that, on merge to `main`, builds the Docker images and publishes them
   to GitHub Container Registry (GHCR) using `GITHUB_TOKEN` (no extra accounts).
6. **README**: architecture diagram, setup and teardown, and dashboard screenshots under normal traffic and
   after drift injection.

## Interfaces with teammates
| With | What we need to agree on |
|---|---|
| Model API (Mo Zhou) | URL and port of `/metrics` |
| Monitoring (Cassie Yang) | Schema of the drift-metrics table in PostgreSQL |
| Training pipeline (Zexi Liu) | How a Grafana alert triggers the retraining flow |
| Frontend (Siyu Liu) | Container name and port in docker-compose |

## Definition of done
- `docker compose up` starts the full stack and `docker compose down` stops it.
- The Grafana dashboard shows traffic and drift; injecting drift fires the alert.
- Merging a PR to `main` publishes the images to GHCR.
