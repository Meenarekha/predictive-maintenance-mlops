import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "models/final_gradient_boosting.pkl"

FEATURE_COLUMNS = [
    "cycle",
    "setting_1",
    "setting_2",
    "lpc_outlet_temperature",
    "hpc_outlet_temperature",
    "lpt_outlet_temperature",
    "bypass_duct_pressure",
    "hpc_outlet_pressure",
    "physical_fan_speed",
    "physical_core_speed",
    "hpc_outlet_static_pressure",
    "fuel_flow_to_ps30_ratio",
    "corrected_fan_speed",
    "corrected_core_speed",
    "bypass_ratio",
    "bleed_enthalpy",
    "high_pressure_turbine_cool_air_flow",
    "low_pressure_turbine_cool_air_flow",
]


app = FastAPI(
    title="Aircraft Predictive Maintenance API",
    description="API for predicting Remaining Useful Life (RUL) of aircraft engines.",
    version="1.0.0",
)

model = None


def load_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model = joblib.load(MODEL_PATH)

    return model

class PredictionRequest(BaseModel):
    cycle: float
    setting_1: float
    setting_2: float
    lpc_outlet_temperature: float
    hpc_outlet_temperature: float
    lpt_outlet_temperature: float
    bypass_duct_pressure: float
    hpc_outlet_pressure: float
    physical_fan_speed: float
    physical_core_speed: float
    hpc_outlet_static_pressure: float
    fuel_flow_to_ps30_ratio: float
    corrected_fan_speed: float
    corrected_core_speed: float
    bypass_ratio: float
    bleed_enthalpy: float
    high_pressure_turbine_cool_air_flow: float
    low_pressure_turbine_cool_air_flow: float


@app.get("/")
def home():
    return {
        "message": "Aircraft Predictive Maintenance API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = np.array(
         [[getattr(request, feature) for feature in FEATURE_COLUMNS]]
    )

    prediction = load_model().predict(input_data)[0]

    return {
        "predicted_RUL": round(float(prediction), 2)
    }