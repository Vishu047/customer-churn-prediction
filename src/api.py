# src/api.py

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.schemas import CustomerData, PredictionResponse

# Global container for serialized artifacts
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts into memory once
    model_path = Path("models/best_random_forest.joblib")
    prep_path = Path("models/preprocessor.joblib")

    if not model_path.exists() or not prep_path.exists():
        raise RuntimeError(
            "Missing serialized artifacts! Ensure 'models/best_random_forest.joblib' "
            "and 'models/preprocessor.joblib' exist."
        )

    artifacts["model"] = joblib.load(model_path)
    artifacts["preprocessor"] = joblib.load(prep_path)
    print("\n[FastAPI] Preprocessor and Random Forest artifacts loaded successfully.")
    yield
    # Shutdown: Clean up resources
    artifacts.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API serving the tuned Random Forest churn model.",
    version="1.0.0",
    lifespan=lifespan,
)


def assign_risk_profile(probability: float) -> tuple[str, str]:
    """Maps continuous churn probability to actionable business recommendations."""
    if probability >= 0.70:
        return (
            "High",
            "Urgent: Assign retention specialist immediately. Offer 1-year contract discount or bundled tech support.",
        )
    elif probability >= 0.40:
        return (
            "Medium",
            "Moderate Risk: Trigger automated satisfaction survey and loyalty perks.",
        )
    else:
        return (
            "Low",
            "Healthy Account: Maintain standard operational communication.",
        )


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        # 1. Convert validated Pydantic model to a single-row DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # 2. Transform raw features using the exact saved ColumnTransformer
        transformed_features = artifacts["preprocessor"].transform(input_data)

        # 3. Model inference
        prob_churn = float(artifacts["model"].predict_proba(transformed_features)[0, 1])
        prediction = int(artifacts["model"].predict(transformed_features)[0])

        risk_level, recommendation = assign_risk_profile(prob_churn)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(prob_churn, 4),
            risk_level=risk_level,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")