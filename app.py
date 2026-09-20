"""Customer churn prediction REST API (Flask).

    python app.py                 # http://127.0.0.1:5000
    POST /predict                 # JSON in, {"prediction", "churn_probability"} out
    GET  /health                  # liveness + model info

The saved model is a full sklearn Pipeline (cleaning -> feature engineering ->
one-hot encoding -> decision tree), so this file contains NO preprocessing
logic: raw customer fields go in, a prediction comes out. That guarantees the
API preprocesses new data exactly like the training notebook did.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
# The pickled pipeline references `src.features.CleanAndEngineer`; make sure the
# project root is importable no matter where the server is started from.
sys.path.insert(0, str(BASE_DIR))

from src.features import RAW_FEATURE_COLUMNS  # noqa: E402
from src.schema import validate_payload        # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model" / "churn_model.pkl"))
META_PATH = BASE_DIR / "model" / "model_metadata.json"

# Fail fast with a clear message instead of a stack trace on the first request.
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. Run notebook/churn_analysis.ipynb "
        "(Run All) to train and save it."
    )
model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

app = Flask(__name__)
app.json.sort_keys = False                      # keep "prediction" before "churn_probability"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024    # a customer record is < 1 KB; reject huge bodies


@app.get("/")
def index():
    return jsonify(
        service="Customer churn prediction API",
        endpoints={"POST /predict": "score one customer", "GET /health": "health check"},
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=metadata.get("model", type(model).__name__),
        library_versions=metadata.get("library_versions", {}),
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(
            error="Invalid input",
            details=[{"field": "body",
                      "message": "Body must be a valid JSON object with header 'Content-Type: application/json'."}],
        ), 400

    record, errors = validate_payload(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400

    customer_id = record.pop("customerID", None)
    row = pd.DataFrame([record], columns=RAW_FEATURE_COLUMNS)

    probability = float(model.predict_proba(row)[0, 1])
    label = int(model.predict(row)[0])          # same decision rule as the notebook evaluation

    response = {"prediction": "Yes" if label == 1 else "No",
                "churn_probability": round(probability, 4)}
    if customer_id is not None:
        response = {"customerID": customer_id, **response}
    return jsonify(response)


@app.errorhandler(Exception)
def handle_error(exc):
    """Always answer with JSON, never an HTML error page."""
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.name, message=exc.description), exc.code
    app.logger.exception("Unhandled error")
    return jsonify(error="Internal server error",
                   message="Something went wrong while scoring this customer."), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)),
            debug=False)
