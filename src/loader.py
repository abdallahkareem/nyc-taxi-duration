import os

import pandas as pd
from dotenv import load_dotenv


load_dotenv()


def load_data() -> pd.DataFrame:
    """Load the dataset from the path defined in the environment variables."""

    data_path = os.getenv("DATA_PATH")

    if not data_path:
        raise ValueError("DATA_PATH is not set in the environment variables.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_parquet(data_path)