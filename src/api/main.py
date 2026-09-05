from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import os


app = FastAPI(
    title="NYC Taxi Duration Prediction API",
    description="Predict NYC taxi trip duration using an ONNX model",
    version="1.0.0",
)


# =========================
# Load Model & Preprocessor
# =========================

MODEL_PATH = os.path.join("models", "model.onnx")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# =========================
# Request / Response Models
# =========================


class TaxiRequest(BaseModel):
    PU_DO: str
    trip_distance: float


class TaxiResponse(BaseModel):
    predicted_duration_minutes: float


# =========================
# Routes
# =========================


@app.get("/")
def root():
    return {
        "message": "NYC Taxi Duration Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
    }


@app.post("/predict", response_model=TaxiResponse)
def predict(request: TaxiRequest):
    # Create DataFrame with the same columns
    # used during training
    input_df = pd.DataFrame(
        [
            {
                "PU_DO": request.PU_DO,
                "trip_distance": request.trip_distance,
            }
        ]
    )

    # Apply the SAME preprocessing used during training
    processed_input = preprocessor.transform(input_df)

    # Convert to dense float32 array
    input_data = processed_input.toarray().astype(np.float32)

    # ONNX prediction
    prediction = session.run(
        [output_name],
        {input_name: input_data},
    )

    duration = float(prediction[0][0][0])

    return {"predicted_duration_minutes": duration}
