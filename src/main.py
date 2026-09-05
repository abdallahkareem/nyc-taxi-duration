import os

import joblib
from dotenv import load_dotenv

from src.loader import load_data
from src.features import create_features
from src.vectorizer import vectorize_features
from src.split import split_data
from src.train import train_model
from src.evaluation import evaluate_model

import src.tracking as tracking


load_dotenv()


def main():
    # Load
    data_path = os.getenv("DATA_PATH")
    df = load_data(
        data_path,
        columns=[
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
        ],
    )

    # Feature engineering
    df = create_features(df)

    # Target
    y = df["duration"]

    # Vectorization
    X_vec, dv = vectorize_features(df)

    # Split
    (X_train, X_test, y_train, y_test) = split_data(X_vec, y)

    # MLflow
    with tracking.start_run():
        tracking.log_params(
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
        )

        # Training
        model = train_model(
            X_train,
            y_train,
        )

        # Evaluation
        rmse, mae = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        tracking.log_metrics(
            rmse,
            mae,
        )

        # Save
        os.makedirs(
            "models",
            exist_ok=True,
        )

        model_path = "models/model.pkl"
        dv_path = "models/dv.pkl"

        joblib.dump(
            model,
            model_path,
        )

        joblib.dump(
            dv,
            dv_path,
        )

        tracking.log_artifact(
            model_path,
            "model",
        )

        tracking.log_artifact(
            dv_path,
            "vectorizer",
        )


if __name__ == "__main__":
    main()
