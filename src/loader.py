import os

import pandas as pd


def load_data(path: str, columns: list) -> pd.DataFrame:
    """Load dataset from the given parquet path."""

    if not path:
        raise ValueError("DATA_PATH is not set.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_parquet(path, columns=columns)
