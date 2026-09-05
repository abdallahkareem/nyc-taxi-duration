import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="NYC Taxi Duration Prediction API",
)


# Load model and vectorizer
model = joblib.load("models/model.pkl")
dv = joblib.load("models/dv.pkl")


class TaxiRequest(BaseModel):
    PU_DO: str
    trip_distance: float


@app.get("/")
def root():
    return {"message": "NYC Taxi Duration Prediction API"}


@app.post("/predict")
def predict(request: TaxiRequest):
    # Convert request to DataFrame
    df = pd.DataFrame(
        [
            {
                "PU_DO": request.PU_DO,
                "trip_distance": request.trip_distance,
            }
        ]
    )

    # Convert features to dictionary
    features = df.to_dict(orient="records")

    # Vectorize
    X = dv.transform(features)

    # Prediction
    prediction = model.predict(X)[0]

    return {"predicted_duration_minutes": float(prediction)}
