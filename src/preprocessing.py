import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw taxi dataset."""

    df = df.copy()

    df["trip_duration"] = (
    df["lpep_dropoff_datetime"]
    - df["lpep_pickup_datetime"]
).dt.total_seconds()

    # Remove invalid durations
    df = df[df["trip_duration"] > 0]

    # Remove invalid distances
    df = df[df["trip_distance"] > 0]

    # Remove missing values
    df = df.dropna(
        subset=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "trip_duration",
        ]
    )
    df = df[
    (df["trip_duration"] >= 60) &
    (df["trip_duration"] <= 7200)
]

    return df