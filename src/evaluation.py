import numpy as np

from sklearn.metrics import mean_squared_error


def evaluate_model(model, X_test, y_test) -> float:
    """Evaluate model using RMSE."""

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    return rmse