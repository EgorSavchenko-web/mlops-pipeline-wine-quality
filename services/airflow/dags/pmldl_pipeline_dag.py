"""
PMLDL Assignment 1 - the complete automated pipeline.

    data_engineering -> model_engineering -> deploy -> smoke_test -> prune

The DAG is scheduled every five minutes. Each run re-reads the raw data,
retrains the model from scratch, rebuilds the API image around the fresh
artefact and restarts the two deployment containers, then proves the result
works by asking the running app container to call the running API container.

Design notes
------------
* Stages 1 and 2 are plain Python scripts under ``code/``, and the DAG runs
  each one as a **separate process** rather than importing it. Two reasons.
  First, the repository layout required by the assignment puts them inside a
  top-level directory called ``code``, which is also the name of a Python
  standard-library module - importing it as a package would shadow the stdlib.
  Second, and more importantly, the Airflow worker has ``dill`` imported, and
  dill patches the standard pickler for the whole process: a model trained
  in-process picks up ``dill._dill`` references and then cannot be loaded in
  the slim API image. A clean subprocess makes the artefact depend only on what
  the API image installs.
* A stage crash therefore cannot take the worker down with it, and each stage
  stays runnable by hand with byte-identical behaviour.
* The deployment stage shells out to ``docker compose``. The Docker socket is
  bind mounted into this container, so the images are built and the containers
  started on the host daemon: they are siblings of Airflow, and their ports are
  published straight to localhost.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

LOG = logging.getLogger(__name__)

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/project"))
DEPLOY_DIR = PROJECT_DIR / "code" / "deployment"
COMPOSE_FILE = DEPLOY_DIR / "docker-compose.yml"


def _run_stage(relative_path: str) -> None:
    """Execute a stage script in a fresh Python process.

    The isolation is load-bearing, not stylistic. Airflow's worker imports
    ``dill``, and dill patches the standard pickler's dispatch table for the
    whole process; a model dumped in that process carries ``dill._dill``
    references and then fails to load in the slim API image, which installs no
    dill. A clean subprocess inherits none of the scheduler's import state, so
    the artefact depends only on what the API image actually has.

    It also means a stage crash cannot take the worker down with it, and the
    stage stays runnable by hand (``python code/models/train_model.py``) with
    byte-identical behaviour.
    """
    script = PROJECT_DIR / relative_path
    if not script.exists():
        raise FileNotFoundError(f"{script} not found - is the repository mounted at {PROJECT_DIR}?")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        env={**os.environ, "PROJECT_DIR": str(PROJECT_DIR)},
        capture_output=True,
        text=True,
    )
    # The stages log through the logging module, which writes to stderr.
    for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        if text.strip():
            LOG.info("%s %s:\n%s", relative_path, stream, text.strip())
    if result.returncode != 0:
        raise RuntimeError(f"{relative_path} exited with code {result.returncode}")


def _read_json(relative_path: str) -> dict:
    return json.loads((PROJECT_DIR / relative_path).read_text())


# ---------------------------------------------------------------------------
# Stage 1 - data engineering
# ---------------------------------------------------------------------------
def run_data_engineering(**context) -> dict:
    _run_stage("code/datasets/prepare_data.py")
    report = _read_json("data/processed/data_report.json")
    LOG.info("data report:\n%s", json.dumps(report, indent=2))
    context["ti"].xcom_push(key="data_report", value=report)
    return report


# ---------------------------------------------------------------------------
# Stage 2 - model engineering
# ---------------------------------------------------------------------------
def run_model_engineering(**context) -> dict:
    _run_stage("code/models/train_model.py")
    metrics = _read_json("models/metrics.json")
    LOG.info("testing metrics:\n%s", json.dumps(metrics, indent=2))

    # A guard rail rather than a hard gate: a forest that scores at chance level
    # means something upstream is broken, and shipping it would be worse than
    # failing the run loudly.
    if metrics["roc_auc"] < 0.6:
        raise ValueError(f"model quality collapsed: roc_auc={metrics['roc_auc']:.3f}")

    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


with DAG(
    dag_id="pmldl_wine_quality_pipeline",
    description="Data engineering -> model engineering -> Dockerised deployment",
    # Every five minutes, as required. max_active_runs=1 means a slow run
    # delays the next one instead of stacking concurrent docker builds on top
    # of each other.
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    default_args={
        "owner": "pmldl",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["pmldl", "mlops", "assignment-1"],
) as dag:

    data_engineering = PythonOperator(
        task_id="data_engineering",
        python_callable=run_data_engineering,
        doc_md="Stage 1: load -> clean (duplicates, missing values, outliers) -> stratified split.",
    )

    model_engineering = PythonOperator(
        task_id="model_engineering",
        python_callable=run_model_engineering,
        doc_md="Stage 2: feature engineering -> train -> evaluate -> log to MLflow -> package.",
    )

    # Stage 3. `--build` rebuilds the API image because models/model.pkl, which
    # it COPYs, has just changed; the dependency layers above it stay cached, so
    # in practice this takes a few seconds after the first run.
    # The compose invocation is wrapped rather than run under `set -e`, so that
    # a failure still prints container state and logs before the task gives up.
    # Without this, a container that dies on startup surfaces in Airflow as the
    # single word "unhealthy" and tells you nothing about why.
    deploy = BashOperator(
        task_id="deploy_api_and_app",
        bash_command=f"""
cd {DEPLOY_DIR}

status=0
docker compose -f {COMPOSE_FILE} up -d --build --force-recreate --remove-orphans || status=$?

echo "===== containers ====="
docker compose -f {COMPOSE_FILE} ps -a || true

echo "===== pmldl-api: last 120 log lines ====="
docker logs --tail 120 pmldl-api 2>&1 || echo "(no api container)"

echo "===== pmldl-api: state ====="
# grep rather than 'docker inspect --format', whose Go template braces collide
# with Airflow's Jinja templating of bash_command.
docker inspect pmldl-api 2>/dev/null \
  | grep -E '"(Status|Running|Restarting|OOMKilled|Dead|ExitCode|Error|FinishedAt|ExitCode|Output)"' \
  | head -40 || true

echo "===== pmldl-app: last 40 log lines ====="
docker logs --tail 40 pmldl-app 2>&1 || echo "(no app container)"

if [ "$status" != "0" ]; then
  echo "compose exited with status $status - see the logs above"
  exit "$status"
fi
echo "deployment finished; app on http://localhost:8501, API docs on http://localhost:8000/docs"
""",
        doc_md="Stage 3: build the API image around the fresh model and (re)start both containers.",
    )

    # Proves the two containers are genuinely separate and genuinely talking:
    # the request is issued from inside the app container and addressed to the
    # api container by its service name on the shared Docker network.
    smoke_test = BashOperator(
        task_id="smoke_test",
        bash_command=r"""
set -e
echo "waiting for the API container to answer..."
ready=0
for attempt in $(seq 1 40); do
  if docker exec pmldl-api python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" >/dev/null 2>&1; then
    ready=1
    echo "API answered after ${attempt} attempt(s)"
    break
  fi
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "API never answered. Health endpoint as last seen:"
  docker exec pmldl-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())" || true
  echo "Last 120 log lines of pmldl-api:"
  docker logs --tail 120 pmldl-api 2>&1 || true
  exit 1
fi

# A container that is up but could not load its model answers /health with
# status "degraded" and an "error" field. Fail loudly on that rather than
# letting the prediction below produce a confusing 503.
echo "API health:"
docker exec pmldl-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())"

echo "calling the API from inside the app container..."
docker exec -i pmldl-app python - <<'PY'
import json
import requests

health = requests.get("http://api:8000/health", timeout=10).json()
assert health["model_loaded"], health

sample = {
    "fixed_acidity": 7.9, "volatile_acidity": 0.35, "citric_acid": 0.46,
    "residual_sugar": 3.6, "chlorides": 0.078, "free_sulfur_dioxide": 15.0,
    "total_sulfur_dioxide": 37.0, "density": 0.9973, "ph": 3.35,
    "sulphates": 0.86, "alcohol": 12.8,
}
response = requests.post("http://api:8000/predict", json=sample, timeout=10)
assert response.status_code == 200, (response.status_code, response.text)
result = response.json()
assert result["prediction"] in (0, 1), result
print("smoke test OK:", json.dumps(result))
PY
echo "app is reachable at http://localhost:8501, API docs at http://localhost:8000/docs"
""",
        doc_md="End-to-end check: app container -> API container -> prediction.",
    )

    # Every run produces a new API image and orphans the previous one. Without
    # this the untagged layers accumulate by gigabytes over a day of 5-minute
    # runs. Only dangling (untagged, unreferenced) images are touched.
    prune = BashOperator(
        task_id="prune_dangling_images",
        bash_command="docker image prune -f",
        doc_md="Housekeeping: drop the untagged images left behind by the rebuild.",
    )

    data_engineering >> model_engineering >> deploy >> smoke_test >> prune
