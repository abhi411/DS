"""Automated checks for the prediction API.  Run from the project root:

    python -m unittest discover -s tests -v
"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402

HIGH_RISK = json.loads((ROOT / "sample_request.json").read_text())

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 64.5, "TotalCharges": 3870.0,
}


class PredictApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post("/predict", json=payload)

    def assert_400_mentions(self, resp, field):
        self.assertEqual(resp.status_code, 400, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertEqual(body["error"], "Invalid input")
        self.assertIn(field, [d["field"] for d in body["details"]])

    # ---- happy paths -------------------------------------------------------
    def test_valid_request_returns_prediction_and_probability(self):
        resp = self.post(HIGH_RISK)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(list(body), ["prediction", "churn_probability"])
        self.assertIn(body["prediction"], ("Yes", "No"))
        self.assertTrue(0.0 <= body["churn_probability"] <= 1.0)

    def test_risk_ordering_is_sensible(self):
        high = self.post(HIGH_RISK).get_json()
        low = self.post(LOW_RISK).get_json()
        self.assertEqual(high["prediction"], "Yes")
        self.assertEqual(low["prediction"], "No")
        self.assertGreater(high["churn_probability"], low["churn_probability"])

    def test_customer_id_is_echoed(self):
        body = self.post({**HIGH_RISK, "customerID": "7590-VHVEG"}).get_json()
        self.assertEqual(body["customerID"], "7590-VHVEG")

    def test_total_charges_as_numeric_string_is_accepted(self):
        self.assertEqual(self.post({**HIGH_RISK, "TotalCharges": "190.1"}).status_code, 200)

    def test_blank_total_charges_ok_for_new_customer(self):
        for blank in (" ", "", None):
            resp = self.post({**HIGH_RISK, "tenure": 0, "TotalCharges": blank})
            self.assertEqual(resp.status_code, 200, blank)

    def test_total_charges_may_be_omitted_for_new_customer(self):
        payload = {k: v for k, v in HIGH_RISK.items() if k != "TotalCharges"}
        self.assertEqual(self.post({**payload, "tenure": 0}).status_code, 200)

    def test_invalid_category_rejected(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "Contract": "Three year"}), "Contract")

    # ---- invalid input -----------------------------------------------------
    def test_missing_required_field(self):
        payload = copy.deepcopy(HIGH_RISK); del payload["Contract"]
        self.assert_400_mentions(self.post(payload), "Contract")

    def test_multiple_errors_are_all_reported(self):
        payload = {**HIGH_RISK, "Contract": "bad", "tenure": "abc"}
        resp = self.post(payload)
        fields = [d["field"] for d in resp.get_json()["details"]]
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Contract", fields); self.assertIn("tenure", fields)

    def test_wrong_types(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "tenure": "abc"}), "tenure")
        self.assert_400_mentions(self.post({**HIGH_RISK, "tenure": 2.5}), "tenure")
        self.assert_400_mentions(self.post({**HIGH_RISK, "tenure": True}), "tenure")
        self.assert_400_mentions(self.post({**HIGH_RISK, "MonthlyCharges": "95.7"}), "MonthlyCharges")
        self.assert_400_mentions(self.post({**HIGH_RISK, "SeniorCitizen": "yes"}), "SeniorCitizen")
        self.assert_400_mentions(self.post({**HIGH_RISK, "gender": 5}), "gender")

    def test_out_of_range_values(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "tenure": -3}), "tenure")
        self.assert_400_mentions(self.post({**HIGH_RISK, "MonthlyCharges": -10}), "MonthlyCharges")

    def test_nan_and_infinity_rejected(self):
        for bad in ("NaN", "Infinity"):
            raw = json.dumps(HIGH_RISK).replace('"MonthlyCharges": 95.7', f'"MonthlyCharges": {bad}')
            resp = self.client.post("/predict", data=raw, content_type="application/json")
            self.assert_400_mentions(resp, "MonthlyCharges")

    def test_total_charges_required_when_tenure_positive(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "TotalCharges": " "}), "TotalCharges")
        self.assert_400_mentions(self.post({**HIGH_RISK, "TotalCharges": "abc"}), "TotalCharges")

    def test_cross_field_consistency(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "PhoneService": "No", "MultipleLines": "Yes"}), "MultipleLines")
        self.assert_400_mentions(self.post({**HIGH_RISK, "InternetService": "No"}), "OnlineSecurity")
        self.assert_400_mentions(self.post({**LOW_RISK, "OnlineSecurity": "No internet service"}), "OnlineSecurity")

    def test_unknown_field_rejected(self):
        self.assert_400_mentions(self.post({**HIGH_RISK, "favourite_colour": "blue"}), "body")

    def test_body_must_be_json_object(self):
        resp = self.client.post("/predict", data="this is not json", content_type="application/json")
        self.assert_400_mentions(resp, "body")
        resp = self.client.post("/predict", data="{}", content_type="text/plain")
        self.assert_400_mentions(resp, "body")
        resp = self.post([HIGH_RISK])
        self.assert_400_mentions(resp, "body")
        resp = self.client.post("/predict")
        self.assert_400_mentions(resp, "body")

    # ---- other endpoints ---------------------------------------------------
    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_errors_are_json_not_html(self):
        resp = self.client.get("/predict")           # wrong method
        self.assertEqual(resp.status_code, 405)
        self.assertEqual(resp.get_json()["error"], "Method Not Allowed")
        resp = self.client.get("/nope")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "Not Found")

    def test_oversized_body_rejected(self):
        resp = self.client.post("/predict", data="x" * 200_000, content_type="application/json")
        self.assertEqual(resp.status_code, 413)


if __name__ == "__main__":
    unittest.main(verbosity=2)
