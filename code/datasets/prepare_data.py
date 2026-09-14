"""
Stage 1 - Data Engineering.

Input  : data/raw/winequality-red.csv          (raw artefact, versioned in git)
Output : data/processed/train.csv              (training artefact)
         data/processed/test.csv               (testing artefact)
         data/processed/data_report.json       (audit trail of what was dropped)

The stage performs the three operations required by the assignment:

  1. LOADING   - read the raw file, tolerating either the ``;`` separator used
                 by the original UCI distribution or the ``,`` separator used
                 by most mirrors.
  2. CLEANING  - normalise column names, drop duplicate rows, impute missing
                 values with the column median, and remove univariate outliers
                 with an interquartile-range rule.
  3. SPLITTING - build the binary target and produce a stratified train/test
                 split, saved as two separate CSV files.

Every step appends a record to ``data_report.json`` so the effect of the stage
is auditable after the fact instead of being buried in scheduler logs.

Run standalone with:  python code/datasets/prepare_data.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# PROJECT_DIR is injected by Airflow; when the script is run by hand we fall
# back to the repository root inferred from this file's location.
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2]))

RAW_PATH = PROJECT_DIR / "data" / "raw" / "winequality-red.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

RANDOM_STATE = 42          # fixed so that every pipeline run is reproducible
TEST_SIZE = 0.2
QUALITY_THRESHOLD = 6      # quality >= 6  ->  "good wine"  (label 1)
IQR_MULTIPLIER = 3.0       # 3.0 = "far out" fence; 1.5 would delete ~15% of a
                           # small dataset, which costs more than it cleans.

LOG = logging.getLogger("stage1.data")

# Import the shared feature contract to reuse the canonical column list.
sys.path.insert(0, str(PROJECT_DIR / "code" / "models"))
from features import RAW_FEATURES, TARGET  # noqa: E402


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, auto-detecting the field separator."""
    if not path.exists():
        raise FileNotFoundError(
            f"raw data not found at {path}. See README section 'Dataset'."
        )
    # ``utf-8-sig`` transparently swallows a UTF-8 BOM if the mirror added one.
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    LOG.info("loaded %s rows x %s cols (sep=%r)", len(df), df.shape[1], sep)
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``Free Sulfur Dioxide`` -> ``free_sulfur_dioxide``."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def drop_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Identical physico-chemical readings are re-measurements, not evidence.

    Leaving them in leaks information across the train/test split (the same
    row can land on both sides) and inflates the reported metrics.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)
    LOG.info("duplicates removed: %s", before - len(df))
    return df


def impute_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Median imputation for numeric gaps.

    The pristine UCI file has no missing values, but a scheduled pipeline must
    not assume that about tomorrow's input, so the guard stays in.  The median
    is preferred over the mean because it is unaffected by the heavy right
    tails these chemistry columns have.
    """
    na_per_column = df.isna().sum()
    report["missing_values_imputed"] = {
        col: int(n) for col, n in na_per_column.items() if n > 0
    }
    if na_per_column.sum():
        df = df.fillna(df.median(numeric_only=True))
        LOG.info("imputed %s missing cells", int(na_per_column.sum()))
    return df


def remove_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Drop rows lying outside the IQR fence on any physico-chemical feature.

    A row is kept only if *every* feature is inside
    ``[Q1 - k*IQR, Q3 + k*IQR]``.  The target column is never used for this
    decision, otherwise the cleaning step would be leaking the label.
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)
    per_column = {}
    for col in RAW_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        col_mask = df[col].between(low, high)
        per_column[col] = int((~col_mask).sum())
        mask &= col_mask
    df = df[mask].reset_index(drop=True)
    report["outliers_removed_total"] = before - len(df)
    report["outliers_flagged_per_column"] = per_column
    LOG.info("outliers removed: %s (%.1f%%)", before - len(df),
             100 * (before - len(df)) / max(before, 1))
    return df


# --------------------------------------------------------------------------
# 3. Splitting
# --------------------------------------------------------------------------
def add_target(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Binarise the 3..8 quality score into a good/not-good label.

    Binary classification is chosen over regression on purpose: the score is an
    ordinal median of three tasters rather than a true interval scale, the
    extreme grades (3, 4, 8) have too few examples to learn, and a good/bad
    verdict with a probability is what the web application can present
    meaningfully to a user.
    """
    df = df.copy()
    df[TARGET] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
    balance = df[TARGET].value_counts(normalize=True).round(4)
    report["class_balance"] = {int(k): float(v) for k, v in balance.items()}
    LOG.info("class balance: %s", report["class_balance"])
    return df


def split_and_save(df: pd.DataFrame, report: dict) -> None:
    """Stratified split, written to two separate files as the stage output."""
    columns = RAW_FEATURES + [TARGET]
    train_df, test_df = train_test_split(
        df[columns],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],   # keeps the good/bad ratio identical on both sides
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    report["train_rows"] = len(train_df)
    report["test_rows"] = len(test_df)
    LOG.info("wrote train=%s rows, test=%s rows to %s",
             len(train_df), len(test_df), PROCESSED_DIR)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report: dict = {"raw_path": str(RAW_PATH)}

    df = load_raw(RAW_PATH)
    df = normalise_columns(df)
    report["rows_raw"] = len(df)

    df = drop_duplicates(df, report)
    df = impute_missing(df, report)
    df = remove_outliers(df, report)
    df = add_target(df, report)
    report["rows_clean"] = len(df)

    split_and_save(df, report)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    LOG.info("stage 1 finished: %s -> %s rows", report["rows_raw"], report["rows_clean"])
    return report


if __name__ == "__main__":
    main()
