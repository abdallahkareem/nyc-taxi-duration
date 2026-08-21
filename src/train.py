import os

import numpy as np
import onnx
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from src.loader import load_data
from src.preprocessing import preprocess
from src.features import create_features
from src.evaluation import evaluate_model


load_dotenv()


def main():

 

    # Load Data
    data_path = os.getenv("DATA_PATH")

    df = load_data(data_path)

    print(f"Loaded {len(df)} rows")


  
    # Preprocessing
    df = preprocess(df)

    print(f"After preprocessing: {len(df)} rows")


    
    # Feature Engineering
    df = create_features(df)


    

    # Features / Target
    X = df[
        [
            "PU_DO",
            "trip_distance",
        ]
    ]

    y = df["trip_duration"] / 60  # Duration in minutes


    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")


    # =========================
    # Preprocessing Pipeline
    # =========================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "PU_DO",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["PU_DO"],
            ),
            (
                "trip_distance",
                "passthrough",
                ["trip_distance"],
            ),
        ]
    )


    # =========================
    # Transform Data
    # =========================

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_test_transformed = preprocessor.transform(X_test)


    # =========================
    # Train Model
    # =========================

    model = LinearRegression()

    model.fit(
        X_train_transformed,
        y_train,
    )


    # =========================
    # Evaluation
    # =========================

    rmse = evaluate_model(
        model,
        X_test_transformed,
        y_test,
    )

    print(f"RMSE: {rmse:.2f} minutes")


    # =========================
    # Convert to ONNX
    # =========================

    print("Converting model to ONNX...")

    n_features = X_train_transformed.shape[1]

    initial_type = [
        (
            "float_input",
            FloatTensorType(
                [None, n_features]
            ),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
    )


    # =========================
    # Save ONNX Model
    # =========================

    os.makedirs("models", exist_ok=True)

    model_path = "models/model.onnx"

    with open(model_path, "wb") as f:
        f.write(
            onnx_model.SerializeToString()
        )

    print(f"Model saved successfully: {model_path}")


if __name__ == "__main__":
    main()