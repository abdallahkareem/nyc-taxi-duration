import pandas as pd

from src.features import create_features


def test_creates_pu_do_column():
    df = pd.DataFrame(
        {
            "PULocationID": [10, 25],
            "DOLocationID": [7, 100],
        }
    )

    result = create_features(df)

    assert "PU_DO" in result.columns
    assert list(result["PU_DO"]) == ["10_7", "25_100"]


def test_pu_do_values_are_strings():
    df = pd.DataFrame(
        {
            "PULocationID": [1],
            "DOLocationID": [2],
        }
    )

    result = create_features(df)

    assert isinstance(result.iloc[0]["PU_DO"], str)


def test_does_not_mutate_original_dataframe():
    df = pd.DataFrame(
        {
            "PULocationID": [1],
            "DOLocationID": [2],
        }
    )

    create_features(df)

    assert "PU_DO" not in df.columns
