import pytest
from pydantic import ValidationError
from litestar.testing import TestClient
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST
from app.main import predict_churn, app
from app.schemas import CustomerFeatures

VALID = {"CreditScore" : 740,
        "Geography": "France",
        "Gender": "Male",
        "Age": 25,
        "Tenure": 3,
        "Balance": 25000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 7500}

# Function Test
def test_predict_churn():
    prediction, label =predict_churn(VALID)
    assert prediction in (0,1)
    assert label in ("Churn", "Stay")

# Endpoints test - home
def test_home():
    with TestClient(app= app) as client:
        resp= client.get("/")
        assert resp.status_code == HTTP_200_OK
        assert resp.json() == {"message": "Welcome to Churn Prediction API", "docs": "/schema/swagger"}
        
# Endpoints test - health
def test_health():
    with TestClient(app= app) as client:
        resp= client.get("/health")
        assert resp.status_code == HTTP_200_OK
        assert resp.json() == {"status": "ok"}

# Endpoint Test - predict happy path
def test_predict():
    with TestClient(app= app ) as client:
        resp= client.post("/predict", json= VALID)
        assert resp.status_code == HTTP_200_OK
        assert resp.json()["Label"] in ("Churn", "Stay")

# Endpoint Test predict -> invalid input
def test_predict_invalid():
    bad = {**VALID, "Geography": "Mars"}
    with TestClient(app = app) as client:
        resp = client.post("/predict", json= bad)
        assert resp.status_code == HTTP_400_BAD_REQUEST

# pytest raises
def test_schema():
    bad = {**VALID, "Geography": "Mars"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)