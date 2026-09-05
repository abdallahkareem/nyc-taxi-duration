import pandas as pd

from src.features import create_features


def test_create_features():
    df = pd.DataFrame(
        {
            "PULocationID": [65, 66],
            "DOLocationID": [233, 188],
            "trip_distance": [6.2, 5.36],
        }
    )

    result = create_features(df)

    assert "PU_DO" in result.columns
    assert result["PU_DO"].iloc[0] == "65_233"
    assert result["PU_DO"].iloc[1] == "66_188"
