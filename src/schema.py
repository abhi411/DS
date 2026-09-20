"""Input schema + validation for the prediction API.

Allowed values come straight from the data dictionary / observed data. Keeping
validation separate from Flask makes it easy to unit-test.
"""
import math

from src.features import ADDON_COLS, RAW_FEATURE_COLUMNS

YES_NO = ["Yes", "No"]

ALLOWED_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# Sanity ranges (training data: tenure 0-72, MonthlyCharges ~18-119).
TENURE_RANGE = (0, 120)
MONTHLY_RANGE = (0.0, 1000.0)
TOTAL_RANGE = (0.0, 100000.0)

OPTIONAL_FIELDS = ["customerID"]  # accepted and echoed back, never used by the model


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _err(errors, field, message):
    errors.append({"field": field, "message": message})


def validate_payload(payload):
    """Validate one customer record.

    Returns (record, errors). `record` is a clean dict ready for a DataFrame
    when there are no errors, otherwise None.
    """
    errors = []
    if not isinstance(payload, dict):
        return None, [{"field": "body", "message": "Request body must be a JSON object."}]

    unknown = sorted(set(payload) - set(RAW_FEATURE_COLUMNS) - set(OPTIONAL_FIELDS))
    if unknown:
        _err(errors, "body", f"Unknown field(s): {unknown}")

    record = {}

    cid = payload.get("customerID")
    if "customerID" in payload and (isinstance(cid, bool) or not isinstance(cid, (str, int))):
        _err(errors, "customerID", "Must be a string or integer if provided.")

    # --- categorical fields ------------------------------------------------
    for col, allowed in ALLOWED_VALUES.items():
        if col not in payload or payload[col] is None:
            _err(errors, col, "Field is required.")
        elif payload[col] not in allowed or not isinstance(payload[col], str):
            _err(errors, col, f"Invalid value {payload[col]!r}. Allowed: {allowed}")
        else:
            record[col] = payload[col]

    # --- SeniorCitizen (binary) -------------------------------------------
    v = payload.get("SeniorCitizen")
    if v is None:
        _err(errors, "SeniorCitizen", "Field is required.")
    elif isinstance(v, bool) or v not in (0, 1):
        _err(errors, "SeniorCitizen", f"Invalid value {v!r}. Allowed: 0 or 1.")
    else:
        record["SeniorCitizen"] = int(v)

    # --- tenure ------------------------------------------------------------
    v = payload.get("tenure")
    tenure = None
    if v is None:
        _err(errors, "tenure", "Field is required.")
    elif not _is_number(v) or not math.isfinite(v) or float(v) != int(v):
        _err(errors, "tenure", f"Must be a whole number of months, got {v!r}.")
    elif not (TENURE_RANGE[0] <= v <= TENURE_RANGE[1]):
        _err(errors, "tenure", f"Must be between {TENURE_RANGE[0]} and {TENURE_RANGE[1]} months.")
    else:
        tenure = int(v)
        record["tenure"] = tenure

    # --- MonthlyCharges ----------------------------------------------------
    v = payload.get("MonthlyCharges")
    if v is None:
        _err(errors, "MonthlyCharges", "Field is required.")
    elif not _is_number(v) or not math.isfinite(v):
        _err(errors, "MonthlyCharges", f"Must be a finite number, got {v!r}.")
    elif not (MONTHLY_RANGE[0] <= v <= MONTHLY_RANGE[1]):
        _err(errors, "MonthlyCharges", f"Must be between {MONTHLY_RANGE[0]} and {MONTHLY_RANGE[1]}.")
    else:
        record["MonthlyCharges"] = float(v)

    # --- TotalCharges ------------------------------------------------------
    # Mirrors the training data: blank/missing is only legitimate for brand-new
    # customers (tenure == 0) who have not been billed yet.
    v = payload.get("TotalCharges")
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            v = None
        else:
            try:
                v = float(v)
            except ValueError:
                _err(errors, "TotalCharges", f"Must be a number or numeric string, got {payload['TotalCharges']!r}.")
                v = "invalid"
    if v == "invalid":
        pass
    elif v is None:
        if tenure == 0:
            record["TotalCharges"] = float("nan")  # cleaned to 0.0 inside the pipeline
        elif tenure is not None:
            _err(errors, "TotalCharges", "Required unless tenure is 0 (customer not yet billed).")
    elif not _is_number(v) or not math.isfinite(v):
        _err(errors, "TotalCharges", f"Must be a finite number, got {v!r}.")
    elif not (TOTAL_RANGE[0] <= v <= TOTAL_RANGE[1]):
        _err(errors, "TotalCharges", f"Must be between {TOTAL_RANGE[0]} and {TOTAL_RANGE[1]}.")
    else:
        record["TotalCharges"] = float(v)

    # --- cross-field consistency (verified in the training data) ------------
    if record.get("PhoneService") == "No" and record.get("MultipleLines") in ("Yes", "No"):
        _err(errors, "MultipleLines", "Must be 'No phone service' when PhoneService is 'No'.")
    if record.get("PhoneService") == "Yes" and record.get("MultipleLines") == "No phone service":
        _err(errors, "MultipleLines", "Cannot be 'No phone service' when PhoneService is 'Yes'.")
    if record.get("InternetService") == "No":
        for col in ADDON_COLS:
            if col in record and record[col] != "No internet service":
                _err(errors, col, "Must be 'No internet service' when InternetService is 'No'.")
    elif record.get("InternetService") in ("DSL", "Fiber optic"):
        for col in ADDON_COLS:
            if record.get(col) == "No internet service":
                _err(errors, col, "Cannot be 'No internet service' when the customer has internet.")

    if errors:
        return None, errors
    ordered = {c: record[c] for c in RAW_FEATURE_COLUMNS}
    if "customerID" in payload:
        ordered["customerID"] = payload["customerID"]
    return ordered, []
