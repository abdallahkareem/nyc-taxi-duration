import pandas as pd
import pytest

from src.preprocessing import preprocess


def _base_df(rows):
    """Helper to build a raw taxi dataframe from a list of dicts."""
    return pd.DataFrame(rows)


def test_computes_trip_duration_in_seconds():
    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:10:00"),
                "trip_distance": 2.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            }
        ]
    )

    result = preprocess(df)

    assert result.iloc[0]["trip_duration"] == 600  # 10 minutes


def test_removes_zero_or_negative_duration():
    df = _base_df(
        [
            {  # valid: 5 minutes
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:05:00"),
                "trip_distance": 1.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            },
            {  # invalid: dropoff before pickup -> negative duration
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:10:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "trip_distance": 1.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            },
        ]
    )

    result = preprocess(df)

    assert len(result) == 1


def test_removes_invalid_distance():
    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:05:00"),
                "trip_distance": 0.0,  # invalid
                "PULocationID": 1,
                "DOLocationID": 2,
            }
        ]
    )

    result = preprocess(df)

    assert len(result) == 0


def test_removes_missing_values():
    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:05:00"),
                "trip_distance": 1.5,
                "PULocationID": None,
                "DOLocationID": 2,
            }
        ]
    )

    result = preprocess(df)

    assert len(result) == 0


@pytest.mark.parametrize("duration_seconds", [59, 7201])
def test_removes_durations_outside_bounds(duration_seconds):
    pickup = pd.Timestamp("2024-01-01 08:00:00")
    dropoff = pickup + pd.Timedelta(seconds=duration_seconds)

    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pickup,
                "lpep_dropoff_datetime": dropoff,
                "trip_distance": 1.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            }
        ]
    )

    result = preprocess(df)

    assert len(result) == 0


@pytest.mark.parametrize("duration_seconds", [60, 3600, 7200])
def test_keeps_durations_within_bounds(duration_seconds):
    pickup = pd.Timestamp("2024-01-01 08:00:00")
    dropoff = pickup + pd.Timedelta(seconds=duration_seconds)

    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pickup,
                "lpep_dropoff_datetime": dropoff,
                "trip_distance": 1.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            }
        ]
    )

    result = preprocess(df)

    assert len(result) == 1


def test_does_not_mutate_original_dataframe():
    df = _base_df(
        [
            {
                "lpep_pickup_datetime": pd.Timestamp("2024-01-01 08:00:00"),
                "lpep_dropoff_datetime": pd.Timestamp("2024-01-01 08:05:00"),
                "trip_distance": 1.0,
                "PULocationID": 1,
                "DOLocationID": 2,
            }
        ]
    )
    original_columns = list(df.columns)

    preprocess(df)

    assert list(df.columns) == original_columns
    assert "trip_duration" not in df.columns
