# Customer Churn Predictin (IBM Telco)

This is a proJect I made to predict if telecom customers are gonna leave (churn). I used a **Decision Tree** model and put it inside a **Flask API** so it's easy to use.

**What I did:** Basically looked at the data, did some cleaning, picked a modle, and made sure it works with a simple web server.

## Quick results

I used a Decision Tree (depth of 5). Tested it on 2,113 customers it haven't seen before.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.711 | 0.474 | 0.822 | 0.601 | 0.832 |

- **I cared most about Recall.** It's better to catch most of teh people who might leave, even if we're wrong sometimes. This model catches **82% of churners**.
- Accuracy isn't everything here because most people stay anyway (73%).
- People usually leave if they have a month-to-month contract, are new, or have fiber-optic internet without extra security.

The full details and charts are in the notebook if you wanna see them.

## File stuff

```
customer_churn_project/
├── data/                         # the csv files
├── notebook/
│   └── churn_analysis.ipynb      # where I did all the work
├── src/
│   ├── features.py               # cleaning code (shared with the api)
│   └── schema.py                 # checks if input is good
├── model/
│   ├── churn_model.pkl           # the saved model
│   └── model_metadata.json       # some info about the model
├── tests/
│   └── test_api.py               # tests to make sure I didn't break things
├── app.py                        # the Flask app
├── requirements.txt              # stuff you need to install
├── README.md
├── sample_request.json
└── sample_response.json
```

## How to setup

You need **Python 3.11+**.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

*Note: if the install fails, you can just remove the versions from requirements.txt and re-run the notebook to make a new model.*

## Running things

### The Notebook
```bash
jupyter notebook notebook/churn_analysis.ipynb
```
Just hit **Restart & Run All**. It takes about a minute and recreates everything.

### The API
```bash
python app.py
```
It runs on `http://127.0.0.1:5000`.

| Endpoint | What it does |
|---|---|
| `POST /predict` | Send customer data, get back if they'll churn |
| `GET /health` | Check if its alive |

### Example request
Send this to `/predict`:
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

And you get this:
```json
{
  "prediction": "Yes",
  "churn_probability": 0.8821
}
```

You can test it with `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

## Validation (making sure input is okay)

The API checks all 19 fields. If you send something wrong (like a negative tenure), it'll give you a `400` error with a message telling you what's wrong. 

## Tests
I wrote 19 tests to make sure everything works.
```bash
python -m unittest discover -s tests -v
```

## Why I did it this way

- **Used a Pipeline:** This makes sure the cleaning code in the API is exactly teh same as in the notebook.
- **TotalCharges blanks:** Fixed the 11 missing values by setting them to 0 (they were just new customers).
- **Balanced weights:** Used this to make teh model better at finding people who leave.
- **Simple model:** I picked a simpler tree because it worked almost as well as the complicated ones but is easier to explain.

## Some things to know
- The probability isn't perfect, it's more for ranking who's most likely to leave.
- Brand new customers (tenure 0) are hard to predict since there weren't many in the data.
- This is just based on old data, it doesn't mean these things *cause* people to leave.
