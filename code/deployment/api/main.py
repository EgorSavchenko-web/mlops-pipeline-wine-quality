"""
Stage 3a - Model API (FastAPI).

The service loads the artefact produced by Stage 2 and exposes it over HTTP.
It runs in its own container and knows nothing about the web application; the
only contract between them is this JSON API.

Endpoints
---------
GET  /health   liveness + whether a model is loaded (used by the Docker
               healthcheck and by the Airflow smoke test)
GET  /model    model card: feature order, training timestamp, importances
GET  /metrics  testing metrics recorded by Stage 2
POST /predict  single-sample prediction
POST /predict/batch  many samples in one call
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import FEATURE_ORDER, RAW_FEATURES, build_features

MODEL_DIR = Path("/srv/models")
LOG = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Wine Quality Model API",
    description="PMLDL Assignment 1 - model serving layer.",
    version="1.0.0",
)

# Module-level state, populated on startup. Keeping the model in memory means a
# prediction costs one forward pass and no disk I/O.
STATE: dict[str, Any] = {"model": None, "card": None, "metrics": None, "error": None}


def _load_json(name: str) -> dict | None:
    path = MODEL_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@app.on_event("startup")
def load_artifacts() -> None:
    """Load the model baked into the image by the pipeline's deployment stage.

    Every failure is caught and recorded rather than raised. An exception here
    would abort uvicorn's startup and kill the container, leaving nothing but
    "unhealthy" to debug; instead the service stays up, /health reports
    ``degraded`` and names the reason, and the Airflow smoke test can print it.
    """
    model_path = MODEL_DIR / "model.pkl"
    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"no model at {model_path} - the image was built without Stage 2 output"
            )
        STATE["model"] = joblib.load(model_path)
        STATE["card"] = _load_json("model_card.json")
        STATE["metrics"] = _load_json("metrics.json")
        LOG.info("model loaded, trained at %s",
                 (STATE["card"] or {}).get("trained_at_utc", "unknown"))
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        LOG.exception("failed to load the model artefacts")


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class WineSample(BaseModel):
    """One wine's physico-chemical measurements.

    Field names mirror the processed dataset exactly, and the bounds are the
    observed ranges of the training data widened a little, so that a typo such
    as an alcohol level of 940 is rejected at the edge instead of producing a
    confident nonsense answer.
    """

    fixed_acidity: float = Field(7.4, ge=0, le=20, description="g(tartaric acid)/dm3")
    volatile_acidity: float = Field(0.70, ge=0, le=2, description="g(acetic acid)/dm3")
    citric_acid: float = Field(0.00, ge=0, le=2, description="g/dm3")
    residual_sugar: float = Field(1.9, ge=0, le=20, description="g/dm3")
    chlorides: float = Field(0.076, ge=0, le=1, description="g(sodium chloride)/dm3")
    free_sulfur_dioxide: float = Field(11.0, ge=0, le=150, description="mg/dm3")
    total_sulfur_dioxide: float = Field(34.0, ge=0, le=400, description="mg/dm3")
    density: float = Field(0.9978, ge=0.9, le=1.1, description="g/cm3")
    ph: float = Field(3.51, ge=2, le=5)
    sulphates: float = Field(0.56, ge=0, le=3, description="g(potassium sulphate)/dm3")
    alcohol: float = Field(9.4, ge=0, le=20, description="% vol")

    model_config = {
        "json_schema_extra": {
            "example": {
                "fixed_acidity": 7.4, "volatile_acidity": 0.70, "citric_acid": 0.0,
                "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0, "density": 0.9978, "ph": 3.51,
                "sulphates": 0.56, "alcohol": 9.4,
            }
        }
    }


class Prediction(BaseModel):
    prediction: int = Field(description="1 = good wine (quality >= 6), 0 = not good")
    label: str
    probability_good: float
    model_trained_at_utc: str | None = None
    predicted_at_utc: str


class BatchRequest(BaseModel):
    samples: list[WineSample]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE["model"] is not None else "degraded",
        "model_loaded": STATE["model"] is not None,
        "model_trained_at_utc": (STATE["card"] or {}).get("trained_at_utc"),
        # Populated only when loading failed - this is what turns an opaque
        # "unhealthy" container into an actionable error message.
        "error": STATE["error"],
    }


@app.get("/model")
def model_card() -> dict:
    if STATE["card"] is None:
        raise HTTPException(503, "model card unavailable")
    return STATE["card"]


@app.get("/metrics")
def metrics() -> dict:
    if STATE["metrics"] is None:
        raise HTTPException(503, "metrics unavailable")
    return STATE["metrics"]


def _predict_frame(df: pd.DataFrame) -> list[dict]:
    model = STATE["model"]
    if model is None:
        raise HTTPException(503, "model not loaded")
    X = build_features(df)[FEATURE_ORDER]
    proba = model.predict_proba(X)[:, 1]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trained_at = (STATE["card"] or {}).get("trained_at_utc")
    out = []
    for p in proba:
        label = int(p >= 0.5)
        out.append(
            {
                "prediction": label,
                "label": "good" if label else "not good",
                "probability_good": round(float(p), 4),
                "model_trained_at_utc": trained_at,
                "predicted_at_utc": now,
            }
        )
    return out


@app.post("/predict", response_model=Prediction)
def predict(sample: WineSample) -> dict:
    df = pd.DataFrame([sample.model_dump()])[RAW_FEATURES]
    return _predict_frame(df)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest) -> list[dict]:
    if not request.samples:
        raise HTTPException(422, "samples must not be empty")
    df = pd.DataFrame([s.model_dump() for s in request.samples])[RAW_FEATURES]
    return _predict_frame(df)
