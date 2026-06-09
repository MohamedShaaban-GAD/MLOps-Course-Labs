import pickle, pandas as pd
from pathlib import Path
from litestar import Litestar, get, post
from app.schemas import CustomerFeatures, PredictionResponse
from app.logging_conf import get_logger
from litestar.status_codes import HTTP_200_OK

ARTIFACTS= Path(__file__).parent / "artifacts"

logger = get_logger()

try:
    with open(ARTIFACTS / "model.pkl", "rb") as f:
        model= pickle.load(f)
    with open(ARTIFACTS / "preprocessor.pkl", "rb") as f:   
        preprocessor= pickle.load(f)
    logger.info("Loaded model and preprocessor")
except Exception as e:
    logger.error(f"Failed to load model/preprocessor: {e}")
    raise

feats = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary"
    ]

def predict_churn(features: dict) -> tuple[int, str]:
    df = pd.DataFrame([features])[feats]
    X = pd.DataFrame(preprocessor.transform(df), columns=preprocessor.get_feature_names_out())
    prediction = int(model.predict(X)[0])
    label = "Churn" if prediction == 1 else "Stay"
    return prediction, label

@get("/")
async def home() -> dict[str, str]:
    return {"message": "Welcome to Churn Prediction API", "docs": "/schema/swagger"}

@get("/health")
async def app_helth() -> dict[str, str]:
    return {"status": "ok"}

@post("/predict", status_code= HTTP_200_OK)
async def predict(data : CustomerFeatures) -> PredictionResponse:
    logger.info(f"New request | input: {data}")
    prediction, label= predict_churn(data.model_dump())
    response= PredictionResponse(Prediction= prediction, Label= label)
    logger.info(f"Response: {response}")
    return response

app = Litestar([home, app_helth, predict])