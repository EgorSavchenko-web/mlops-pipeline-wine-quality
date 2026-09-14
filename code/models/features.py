"""
Shared feature-engineering contract.

This module is the single source of truth for how raw wine measurements are
turned into the numeric matrix the model consumes.  It is imported by:

  * ``code/models/train_model.py``  (Stage 2 - training)
  * ``code/deployment/api/main.py`` (Stage 3 - serving)

Keeping one implementation on both sides removes the classic MLOps failure
mode of training/serving skew, where the API silently feeds the model a
differently-shaped or differently-ordered matrix than it was trained on.

The module deliberately has no dependency other than pandas, so it can be
copied into the slim API image without dragging scikit-learn along.
"""

from __future__ import annotations

import pandas as pd

# Raw physico-chemical measurements, exactly as they appear in the processed
# CSVs after column normalisation in Stage 1.  Order matters: it defines the
# order of the columns produced below.
RAW_FEATURES: list[str] = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

# Derived features. Each one encodes a piece of oenological domain knowledge
# that a tree model would otherwise have to approximate with many splits.
DERIVED_FEATURES: list[str] = [
    "total_acidity",       # overall acid load of the wine
    "bound_sulfur_ratio",  # share of SO2 that is already bound (not protective)
    "alcohol_to_density",  # alcohol normalised by density - a sugar/alcohol proxy
    "acidity_to_alcohol",  # balance between sharpness and body
]

# Final column order fed to the estimator.
FEATURE_ORDER: list[str] = RAW_FEATURES + DERIVED_FEATURES

# Name of the binary target produced in Stage 1.
TARGET = "is_good"

# Small constant guarding the divisions below against zero denominators.
_EPS = 1e-6


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix for ``df``.

    ``df`` must contain every column in :data:`RAW_FEATURES`.  Extra columns
    (such as the target) are ignored.  The returned frame always has exactly
    the columns of :data:`FEATURE_ORDER`, in that order.
    """
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing raw feature columns: {missing}")

    out = df[RAW_FEATURES].astype("float64").copy()

    out["total_acidity"] = out["fixed_acidity"] + out["volatile_acidity"] + out["citric_acid"]
    out["bound_sulfur_ratio"] = (
        out["total_sulfur_dioxide"] - out["free_sulfur_dioxide"]
    ) / (out["total_sulfur_dioxide"] + _EPS)
    out["alcohol_to_density"] = out["alcohol"] / (out["density"] + _EPS)
    out["acidity_to_alcohol"] = out["total_acidity"] / (out["alcohol"] + _EPS)

    return out[FEATURE_ORDER]
