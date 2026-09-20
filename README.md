# Customer Churn Prediction (IBM Telco)

An end-to-end machine-learning project: a **Decision Tree** that flags telecom customers likely to churn, wrapped in a **Flask REST API** so the retention team can score customers on demand.

**Workflow:** Business problem → Data → Preparation → EDA → Feature engineering → Model → Evaluation → Interpretation → Saved model → API

## Results at a glance

Final model: Decision Tree (`gini`, `max_depth=5`, `min_samples_leaf=100`, `class_weight="balanced"`, 24 leaves), evaluated once on a held-out 30% test set (2,113 customers, 561 churners).

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.711 | 0.474 | 0.822 | 0.601 | 0.832 |

- **Recall is the priority.** Missing a churner (lost customer) costs far more than a false alarm (an unneeded retention offer). The model catches **82% of churners** by contacting 46% of customers (1.8× lift over random).
- Accuracy is deliberately *not* the headline: always predicting "No" already scores 73.5%.
- Main drivers: contract type (month-to-month), tenure (first year), fiber-optic internet, higher monthly charges, no online-security / tech-support add-ons.

The full reasoning (decisions, charts, business interpretation, limitations) is in the notebook.

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn_-_Data_Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb      # complete analysis and modelling (outputs included)
├── src/
│   ├── features.py               # cleaning + feature engineering + pipeline builder (shared)
│   └── schema.py                 # API input validation rules
├── model/
│   ├── churn_model.pkl           # saved full pipeline (preprocessing + tree)
│   └── model_metadata.json       # parameters, test metrics, library versions
├── tests/
│   └── test_api.py               # 19 automated API tests
├── app.py                        # Flask REST API
├── requirements.txt
├── README.md
├── sample_request.json
└── sample_response.json
```

`src/features.py` is imported by **both** the notebook and the API. The saved pipeline references its `CleanAndEngineer` class, so the API must be able to import `src` (it does this automatically when started from anywhere).

## Setup

Requires **Python 3.11 or newer** (built and tested on 3.12).

```bash
cd customer_churn_project
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> The pinned versions are the ones the shipped model was trained with. If `pip` cannot install them (for example on an older Python), relax the pins and simply re-run the notebook: it retrains and overwrites `model/churn_model.pkl`, so model and libraries always match.

## Run the notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Use **Kernel → Restart & Run All**. It takes about a minute, reproduces every table and chart, and (re)writes `model/churn_model.pkl`, `model/model_metadata.json` and `sample_request.json`. It works whether Jupyter is started from the project root or from `notebook/`.

## Run the API

```bash
python app.py
```

The server listens on `http://127.0.0.1:5000`. Optional environment variables: `PORT`, `HOST`, `MODEL_PATH`.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one customer (JSON in, prediction + probability out) |
| `GET /health` | Liveness check plus model info |

### Sample request and response

**Request** (`sample_request.json`)

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.7, "TotalCharges": 190.1
}
```

**Response** (`200 OK`, `sample_response.json`)

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8821
}
```

### Ways to call it

```bash
# macOS / Linux / Git Bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

```powershell
# Windows PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -InFile sample_request.json
```

```python
# Python (pip install requests)
import json, requests
r = requests.post("http://127.0.0.1:5000/predict", json=json.load(open("sample_request.json")))
print(r.status_code, r.json())
```

### Input fields

All 19 fields are required, using the values from the data dictionary:

| Field | Allowed values |
|---|---|
| `gender` | Female, Male |
| `SeniorCitizen` | 0 or 1 |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | Yes, No |
| `tenure` | whole number of months, 0-120 |
| `MultipleLines` | Yes, No, No phone service |
| `InternetService` | DSL, Fiber optic, No |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Yes, No, No internet service |
| `Contract` | Month-to-month, One year, Two year |
| `PaymentMethod` | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | number, 0-1000 |
| `TotalCharges` | number or numeric string. May be blank or omitted **only** when `tenure` is 0 (customer not yet billed) |
| `customerID` | optional string/integer; echoed back in the response, never used by the model |

### Handling invalid input

Bad input never reaches the model and never produces a stack trace. The API returns **HTTP 400** with every problem it found:

```json
{
  "error": "Invalid input",
  "details": [
    {"field": "Contract", "message": "Invalid value 'Three year'. Allowed: ['Month-to-month', 'One year', 'Two year']"},
    {"field": "tenure", "message": "Must be between 0 and 120 months."}
  ]
}
```

What is checked: body is a JSON object; no unknown fields; all required fields present; correct types (booleans, `NaN` and `Infinity` rejected); allowed categories; numeric ranges; and cross-field consistency (e.g. `PhoneService = "No"` requires `MultipleLines = "No phone service"`). Wrong method → `405`, unknown URL → `404`, oversized body → `413`; all answers are JSON.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

19 tests cover valid requests, risk ordering (a high-risk customer must score above a low-risk one), every validation rule, malformed bodies and the JSON error handlers.

## Key design decisions

| Decision | Reason |
|---|---|
| Split 70:30 (`random_state=42`, stratified) **before** any analysis | Prevents leakage; keeps the 26.5% churn rate in both parts |
| One sklearn `Pipeline` saved as a single file | API contains no preprocessing code, so training and serving cannot drift apart |
| `TotalCharges` blank → 0 | The 11 blanks are all brand-new customers (tenure 0) who have not been billed; a constant fill cannot leak information |
| Keep the 22 same-profile rows | They are different customers (different IDs); their churn labels agree |
| `class_weight="balanced"` | Lifts recall from ~0.50 to ~0.80 on the churn class, the business priority |
| Models compared with 5-fold CV on the training set; final choice by CV F1 | Test set touched once, at the end |
| Shallowest tree among near-ties | Grid results differ by <0.002 F1 (noise is ~0.02), so the simpler tree wins |

## Limitations

- `churn_probability` is a **risk score for ranking customers**, not a calibrated probability. With balanced class weights the average score on the test set (0.41) sits above the true churn rate (0.27).
- The four engineered features do not measurably improve a decision tree (differences are within noise); they are kept for interpretability. See section 4 of the notebook.
- Customers with `tenure = 0` were never observed to churn (only 11 exist), so predictions for them are extrapolations.
- The results describe associations in one historical snapshot, not causes. Retention offers should be validated with an A/B test, and the model retrained periodically.
