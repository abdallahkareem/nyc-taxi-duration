import pandas as pd

from src.train import split_data


def test_split_data():
    df = pd.DataFrame(
        {
            "PU_DO": ["1_2", "2_3", "3_4", "4_5", "5_6"],
            "trip_distance": [1, 2, 3, 4, 5],
            "duration": [10, 20, 30, 40, 50],
        }
    )

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
