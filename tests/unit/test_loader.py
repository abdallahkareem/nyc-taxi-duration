import pandas as pd
import pytest

from src.loader import load_data


def test_load_data_raises_when_path_is_empty():
    with pytest.raises(ValueError, match="DATA_PATH is not set"):
        load_data("")


def test_load_data_raises_when_path_is_none():
    with pytest.raises(ValueError, match="DATA_PATH is not set"):
        load_data(None)


def test_load_data_raises_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(str(missing_path))


def test_load_data_returns_dataframe(tmp_path):
    file_path = tmp_path / "sample.parquet"

    df = pd.DataFrame(
        {
            "PULocationID": [1, 2],
            "DOLocationID": [3, 4],
            "trip_distance": [1.5, 2.0],
        }
    )
    df.to_parquet(file_path)

    result = load_data(str(file_path))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["PULocationID", "DOLocationID", "trip_distance"]
