import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

data_path = os.getenv("DATA_PATH")

df = pd.read_parquet(data_path)

df.info()

