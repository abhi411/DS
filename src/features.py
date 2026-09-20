"""Shared cleaning + feature engineering for the churn project.

This module is imported by BOTH the notebook (training) and app.py (serving).
That is deliberate: the fitted pipeline is pickled with a reference to
`src.features.CleanAndEngineer`, so the class must live in an importable module
(a class defined inside a notebook is pickled as `__main__.X` and cannot be
loaded by the API).

Design rule to avoid data leakage: `CleanAndEngineer` is STATELESS. It learns
nothing from the data (no medians, no means), so it behaves identically on
training data, test data and a single JSON record sent to the API. Everything
that is *fitted* (one-hot encoder, decision tree) sits after it in the
sklearn Pipeline and is fitted on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------
# Column definitions (order matches the original CSV, minus customerID/Churn)
# --------------------------------------------------------------------------
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
# SeniorCitizen is already 0/1, so it is passed through as a number.
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = [c for c in RAW_FEATURE_COLUMNS if c not in RAW_NUMERIC]

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
AUTOPAY_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]

ENGINEERED_NUMERIC = ["num_addons", "avg_monthly_spend", "autopay"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


class CleanAndEngineer(BaseEstimator, TransformerMixin):
    """Clean raw Telco columns and (optionally) add engineered features.

    Parameters
    ----------
    add_features : bool, default=True
        If False only cleaning is applied. This lets the notebook run a clean
        with/without-features comparison using the very same code path.
    """

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self  # stateless: nothing is learned from the data

    def transform(self, X):
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

        # Select + order the model columns (this also drops customerID / Churn).
        df = X[RAW_FEATURE_COLUMNS].copy()

        # TotalCharges is text in the raw CSV because 11 brand-new customers
        # (tenure == 0) have a blank " ". They have not been billed yet, so the
        # correct value is 0 (a constant, not a statistic learned from data).
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        if self.add_features:
            # 1. Lifecycle stage: churn risk falls steeply with tenure.
            df["tenure_group"] = pd.cut(
                df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
            ).astype(str)
            # 2. Engagement: how many optional add-ons the customer actually has.
            df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1).astype(int)
            # 3. Historical average bill (falls back to the current bill for new customers).
            df["avg_monthly_spend"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / df["tenure"].clip(lower=1),
                df["MonthlyCharges"],
            )
            # 4. Payment friction: automatic payment = fewer chances to lapse.
            df["autopay"] = df["PaymentMethod"].isin(AUTOPAY_METHODS).astype(int)
        return df


def build_pipeline(model, add_features=True):
    """Full pipeline: clean/engineer -> one-hot encode -> model.

    Trees do not need feature scaling, so numeric columns are passed through.
    `handle_unknown="ignore"` means an unseen category at prediction time
    becomes an all-zero row instead of crashing the API.
    """
    cat_cols = RAW_CATEGORICAL + (ENGINEERED_CATEGORICAL if add_features else [])
    num_cols = RAW_NUMERIC + (ENGINEERED_NUMERIC if add_features else [])
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("clean", CleanAndEngineer(add_features=add_features)),
        ("prep", prep),
        ("clf", model),
    ])
