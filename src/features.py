import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features used by the ML model."""

    df = df.copy()

    df["PU_DO"] = (
        df["PULocationID"].astype(str)
        + "_"
        + df["DOLocationID"].astype(str)
    )

    return df