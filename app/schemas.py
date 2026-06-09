from typing import Annotated, Literal
from pydantic import BaseModel, Field



class CustomerFeatures(BaseModel):
    CreditScore: Annotated[int, Field(ge=350, le= 850)]
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: Annotated[int, Field(ge=18, le=100)]
    Tenure: Annotated[int, Field(ge=0)]
    Balance: Annotated[float, Field(ge=0)]
    NumOfProducts: Annotated[int, Field(ge=1)]
    HasCrCard: Literal[0,1]
    IsActiveMember: Literal[0,1]
    EstimatedSalary: Annotated[float, Field(ge=0)]

class PredictionResponse(BaseModel):
    Prediction: int
    Label: str # churn or stay