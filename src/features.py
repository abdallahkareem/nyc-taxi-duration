import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features used by the ML model."""

    df = df.copy()

    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["duration"] = (dropoff - pickup).dt.total_seconds() / 60

    df = df.rename(
        columns={
            "PULocationID": "PUlocationID",
            "DOLocationID": "DOlocationID",
        }
    )

    df["PU_DO"] = (
        df["PUlocationID"].astype("Int64").astype(str)
        + "_"
        + df["DOlocationID"].astype("Int64").astype(str)
    )

    df = df[
        df["duration"].between(1, 60)
        & df["trip_distance"].between(0.01, 100)
        & df["PUlocationID"].notna()
        & df["DOlocationID"].notna()
    ].copy()

    return df
