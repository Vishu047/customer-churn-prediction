# src/schemas.py

from typing import Literal
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    # Demographics
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1, description="0: No, 1: Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # Account / Tenure
    tenure: int = Field(ge=0, le=100, description="Months with company")

    # Services
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    OnlineBackup: Literal["No internet service", "No", "Yes"]
    DeviceProtection: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]
    StreamingTV: Literal["No internet service", "No", "Yes"]
    StreamingMovies: Literal["No internet service", "No", "Yes"]

    # Contract & Billing
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, description="Current monthly bill amount")
    TotalCharges: float = Field(ge=0.0, description="Total historical charges")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    recommendation: str
    