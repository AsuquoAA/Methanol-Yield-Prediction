from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib

from app.target_transformation import load_boxcox_transformer  # Adjust if needed

# Initialize FastAPI app
app = FastAPI()

# Load pipeline and transformer
model = joblib.load('model.joblib')
boxcox = load_boxcox_transformer()

# Root route
@app.get("/")
def read_root():
    return {"message": "Methanol Yield Prediction API is live!"}

# Input schema
class YieldFeatures(BaseModel):
    Temperature: float = Field(..., alias="Temperature (K)")
    Pressure_Bar: float = Field(..., alias="Pressure (bar)")
    Residence_Time_1: float = Field(..., alias="Residence Time (s)_1")
    Residence_Time_2: float = Field(..., alias="Residence Time (s)_2")

    class Config:
        allow_population_by_field_name = True

# Prediction route
@app.post("/predict")
def predict_yield(features: YieldFeatures):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict(by_alias=True)])

        # Basic input validation
        if (input_df <= 0).any().any():
            raise ValueError("All input values must be greater than zero.")

        # Predict transformed yield
        transformed_pred = model.predict(input_df)

        # Inverse Box-Cox transform to get real yield
        final_pred = boxcox.inverse_transform(np.array(transformed_pred).reshape(-1, 1))

        # Return prediction
        return {"Predicted Percentage yield": f"{float(final_pred.flatten()[0]*100):.2f}%"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Catch all unexpected errors
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )
