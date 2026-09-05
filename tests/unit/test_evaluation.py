import numpy as np

from src.evaluation import calculate_rmse


def test_calculate_rmse():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    rmse = calculate_rmse(y_true, y_pred)

    assert rmse == 0
