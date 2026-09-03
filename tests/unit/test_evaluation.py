import numpy as np

from src.evaluation import evaluate_model


class _FakeModel:
    """A tiny stand-in for a fitted sklearn model, no real ML needed here."""

    def __init__(self, predictions):
        self._predictions = np.array(predictions)

    def predict(self, X):
        return self._predictions


def test_returns_zero_rmse_for_perfect_predictions():
    y_test = np.array([10, 20, 30])
    model = _FakeModel(predictions=[10, 20, 30])

    rmse = evaluate_model(model, X_test=None, y_test=y_test)

    assert rmse == 0.0


def test_computes_expected_rmse():
    y_test = np.array([0, 0, 0, 0])
    model = _FakeModel(predictions=[1, 1, 1, 1])  # constant error of 1

    rmse = evaluate_model(model, X_test=None, y_test=y_test)

    assert rmse == 1.0


def test_rmse_is_always_non_negative():
    y_test = np.array([5, -3, 8])
    model = _FakeModel(predictions=[2, -10, 100])

    rmse = evaluate_model(model, X_test=None, y_test=y_test)

    assert rmse >= 0
