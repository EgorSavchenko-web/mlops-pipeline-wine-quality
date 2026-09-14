"""
Stage 2 - Model Engineering.

Input  : data/processed/train.csv
         data/processed/test.csv
Output : models/model.pkl        (packaged estimator, consumed by the API)
         models/model_card.json  (feature order + provenance for the API)
         models/metrics.json     (testing metrics, also logged to MLflow)
         mlruns/                 (MLflow file-based tracking store)

Operations, in the order the assignment prescribes:

  1. FEATURE ENGINEERING - delegated to ``features.build_features`` so that the
     API applies byte-identical logic at serving time.
  2. TRAINING            - a RandomForestClassifier on the training features.
  3. EVALUATION          - accuracy / precision / recall / F1 / ROC-AUC on the
     held-out testing features, logged to MLflow and to a JSON file.
  4. PACKAGING           - joblib dump of the fitted estimator plus a model
     card describing the exact input contract.

Run standalone with:  python code/models/train_model.py
"""

from __future__ import annotations

import json
import logging
import os
import pickletools
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2]))
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
MLRUNS_DIR = PROJECT_DIR / "mlruns"

RANDOM_STATE = 42
EXPERIMENT_NAME = "wine-quality-classification"

# Whether to store the 4 MB model binary as an MLflow artefact on every run.
# On, because the assignment asks for the model itself to be logged in MLflow
# and because it makes each run reproducible from the tracking store alone.
# The cost is real: one run every five minutes is roughly 1.2 GB of tracking
# store per day, so set this to False for a long unattended deployment. The
# lightweight model card is logged either way, so even then every run records
# exactly which model it produced.
LOG_MODEL_BINARY = True

# Hyper-parameters. Deliberately modest: the assignment rewards a working
# pipeline, not a leaderboard score, and a 200-tree forest on ~1k rows trains
# in under a second, which keeps the 5-minute schedule comfortable.
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,      # mild regularisation against memorising single rows
    max_features="sqrt",
    class_weight="balanced", # the classes are near-even, but this keeps the
                             # model honest if the balance drifts in future data
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

LOG = logging.getLogger("stage2.model")

sys.path.insert(0, str(PROJECT_DIR / "code" / "models"))
from features import FEATURE_ORDER, TARGET, build_features  # noqa: E402


# Modules that must never appear inside the serialised model. The API image
# installs only the libraries in code/deployment/api/requirements.txt; a
# reference to anything else means the artefact will not load there.
#
# `dill` is the concrete case this guard was written for. Airflow's worker
# process imports dill, and dill patches the standard pickler's dispatch table
# globally, so a model dumped inside that process picks up `dill._dill`
# references and then dies on startup in the API container with a bare
# ModuleNotFoundError. Training runs in its own process precisely to avoid
# that, and this check proves it worked instead of trusting that it did.
FORBIDDEN_PICKLE_MODULES = (b"dill", b"cloudpickle", b"__main__")


def verify_artifact(model_path: Path, model, X_sample) -> None:
    """Fail the stage here rather than in production if the pickle is unusable."""
    payload = model_path.read_bytes()
    found = [m.decode() for m in FORBIDDEN_PICKLE_MODULES if m in payload]
    if found:
        raise RuntimeError(
            f"{model_path.name} references {found}, which the API image does not "
            f"install. Was training run inside a process that imported one of them?"
        )

    # Round-trip: the artefact must reproduce the in-memory model exactly.
    reloaded = joblib.load(model_path)
    if not (reloaded.predict(X_sample) == model.predict(X_sample)).all():
        raise RuntimeError("reloaded model disagrees with the in-memory model")

    # Report which third-party packages the pickle does depend on, so a future
    # version bump that adds one is visible in the task log.
    try:
        modules = sorted({
            arg.split(".")[0]
            for op, arg, _ in pickletools.genops(payload)
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE")
            and isinstance(arg, str) and "." in arg
            and arg.split(".")[0].isidentifier()
        })
        LOG.info("artefact verified; pickle references: %s", modules)
    except Exception:  # the byte scan above is the check that matters
        LOG.info("artefact verified (round-trip ok)")


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = PROCESSED_DIR / "train.csv", PROCESSED_DIR / "test.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run Stage 1 first")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def evaluate(model, X, y) -> dict:
    """Testing metrics. ROC-AUC uses probabilities, the rest use hard labels."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_split()

    # --- 1. feature engineering -------------------------------------------
    X_train, y_train = build_features(train_df), train_df[TARGET]
    X_test, y_test = build_features(test_df), test_df[TARGET]
    LOG.info("features: %s columns, train=%s test=%s",
             X_train.shape[1], len(X_train), len(X_test))

    # --- 2. training -------------------------------------------------------
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # --- 3. evaluation -----------------------------------------------------
    metrics = evaluate(model, X_test, y_test)
    # 5-fold CV on the training half is a cheap guard against a lucky split:
    # if the test F1 and the CV F1 diverge sharply, the split is the reason.
    metrics["cv_f1_mean"] = float(
        cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
    )
    metrics["train_accuracy"] = float(model.score(X_train, y_train))
    LOG.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    importances = dict(
        sorted(
            zip(FEATURE_ORDER, (float(v) for v in model.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    LOG.info("top features: %s", list(importances)[:5])

    # --- MLflow tracking ---------------------------------------------------
    # A local file store is used instead of a tracking server: it needs no extra
    # container to produce a run history, and `mlflow ui` (or the optional
    # mlflow service in the Airflow compose file) reads the same directory.
    try:
        import mlflow

        mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")):
            mlflow.log_params(MODEL_PARAMS)
            mlflow.log_params(
                {
                    "n_train_rows": len(X_train),
                    "n_test_rows": len(X_test),
                    "n_features": X_train.shape[1],
                    "sklearn_version": sklearn.__version__,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"stage": "model-engineering", "pipeline": "airflow"})
            mlflow_ok = True
    except Exception as exc:  # tracking must never break the pipeline
        LOG.warning("MLflow logging skipped: %s", exc)
        mlflow_ok = False

    # --- 4. packaging ------------------------------------------------------
    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model, model_path)
    verify_artifact(model_path, model, X_test)

    card = {
        "model_type": type(model).__name__,
        "params": MODEL_PARAMS,
        "feature_order": FEATURE_ORDER,
        "target": TARGET,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "feature_importances": importances,
        "mlflow_logged": mlflow_ok,
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Attach the model card - and optionally the binary - to the run just
    # created, so the tracking store says what each set of metrics was produced
    # by rather than only what the numbers were.
    if mlflow_ok:
        try:
            import mlflow

            with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
                mlflow.log_artifact(str(MODELS_DIR / "model_card.json"), artifact_path="model")
                if LOG_MODEL_BINARY:
                    mlflow.log_artifact(str(model_path), artifact_path="model")
        except Exception as exc:
            LOG.warning("MLflow artifact logging skipped: %s", exc)

    LOG.info("stage 2 finished: %s (%.1f KB)", model_path,
             model_path.stat().st_size / 1024)
    return metrics


if __name__ == "__main__":
    main()
