"""
Stage 3b - Web application (Streamlit).

The app owns no model. It collects the eleven physico-chemical inputs, POSTs
them to the model API over the internal Docker network, and renders whatever
comes back. That separation is the point of the assignment: the two containers
can be rebuilt, restarted or scaled independently of each other.
"""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st

# Inside docker-compose the API is reachable by service name; when the app is
# run on the host for debugging, override with API_URL=http://localhost:8000.
API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")
TIMEOUT = 10

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# Eleven inputs: (key, label, min, max, default, step, help)
FIELDS = [
    ("fixed_acidity",        "Fixed acidity (g/dm³)",        0.0, 20.0,  7.4,   0.1),
    ("volatile_acidity",     "Volatile acidity (g/dm³)",     0.0,  2.0,  0.70,  0.01),
    ("citric_acid",          "Citric acid (g/dm³)",          0.0,  2.0,  0.00,  0.01),
    ("residual_sugar",       "Residual sugar (g/dm³)",       0.0, 20.0,  1.9,   0.1),
    ("chlorides",            "Chlorides (g/dm³)",            0.0,  1.0,  0.076, 0.001),
    ("free_sulfur_dioxide",  "Free SO₂ (mg/dm³)",            0.0, 150.0, 11.0,  1.0),
    ("total_sulfur_dioxide", "Total SO₂ (mg/dm³)",           0.0, 400.0, 34.0,  1.0),
    ("density",              "Density (g/cm³)",              0.900, 1.100, 0.9978, 0.0001),
    ("ph",                   "pH",                           2.0,  5.0,  3.51,  0.01),
    ("sulphates",            "Sulphates (g/dm³)",            0.0,  3.0,  0.56,  0.01),
    ("alcohol",              "Alcohol (% vol)",              0.0, 20.0,  9.4,   0.1),
]

# Two real rows from the dataset, handy for a live demonstration.
PRESETS = {
    "Typical low-quality red": dict(
        fixed_acidity=7.4, volatile_acidity=0.70, citric_acid=0.00, residual_sugar=1.9,
        chlorides=0.076, free_sulfur_dioxide=11.0, total_sulfur_dioxide=34.0,
        density=0.9978, ph=3.51, sulphates=0.56, alcohol=9.4),
    "Typical high-quality red": dict(
        fixed_acidity=7.9, volatile_acidity=0.35, citric_acid=0.46, residual_sugar=3.6,
        chlorides=0.078, free_sulfur_dioxide=15.0, total_sulfur_dioxide=37.0,
        density=0.9973, ph=3.35, sulphates=0.86, alcohol=12.8),
}


def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Sidebar: proof that the app really is talking to a separate service
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Model service")
    st.caption(f"API endpoint: `{API_URL}`")
    health = api_get("/health")
    if health and health.get("model_loaded"):
        st.success("API reachable, model loaded")
        st.caption(f"Model trained at {health.get('model_trained_at_utc')} UTC")
    elif health:
        st.warning("API reachable, but no model is loaded")
    else:
        st.error("API unreachable - is the `api` container running?")

    metrics = api_get("/metrics")
    if metrics:
        st.subheader("Test metrics")
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("F1", f"{metrics['f1']:.3f}")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    card = api_get("/model")
    if card:
        with st.expander("Top features"):
            for name, value in list(card.get("feature_importances", {}).items())[:6]:
                st.write(f"`{name}` — {value:.3f}")

# --------------------------------------------------------------------------
# Main form
# --------------------------------------------------------------------------
st.title("🍷 Wine Quality Predictor")
st.write(
    "Enter the physico-chemical measurements of a red wine and the model will "
    "predict whether a taster panel would score it **6 or higher**."
)

preset = st.selectbox("Load a preset", ["— custom —", *PRESETS])
defaults = PRESETS.get(preset, {})

with st.form("wine_form"):
    values = {}
    columns = st.columns(3)
    for index, (key, label, low, high, default, step) in enumerate(FIELDS):
        with columns[index % 3]:
            values[key] = st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=float(defaults.get(key, default)),
                step=step,
                format="%.4f" if step < 0.01 else "%.2f",
                key=f"{preset}:{key}",   # re-seed the widgets when a preset changes
            )
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=values, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        probability = result["probability_good"]
        st.subheader("Prediction")
        if result["prediction"] == 1:
            st.success(f"### Good wine  ·  P(good) = {probability:.1%}")
        else:
            st.warning(f"### Not a good wine  ·  P(good) = {probability:.1%}")
        st.progress(probability)
        st.caption(
            f"Served by the model trained at {result.get('model_trained_at_utc')} UTC · "
            f"response received {datetime.now().strftime('%H:%M:%S')}"
        )
        with st.expander("Raw API response"):
            st.json(result)
