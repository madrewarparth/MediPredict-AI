"""
Tests for Data Preprocessing, Quality Checks, and Point-in-Time Feature Discipline.
"""

import pytest
import pandas as pd

from src.data_preprocessing import (
    clean_appointment_data,
    validate_raw_schema,
    DataQualityError,
)
from src.feature_engineering import (
    derive_point_in_time_features,
    verify_feature_discipline,
    LeakageError,
    split_chronologically,
    PROHIBITED_COLUMNS,
)
from src.config import ALL_MODEL_FEATURES, TARGET_COL


@pytest.fixture
def sample_raw_dataframe():
    """Generates a valid raw DataFrame matching the Kaggle schema."""
    return pd.DataFrame({
        "PatientId": [101, 102, 103, 104, 105],
        "AppointmentID": [5001, 5002, 5003, 5004, 5005],
        "Gender": ["F", "M", "F", "M", "F"],
        "ScheduledDay": [
            "2016-04-10T08:00:00Z",
            "2016-04-11T09:00:00Z",
            "2016-04-12T10:00:00Z",
            "2016-04-13T11:00:00Z",
            "2016-04-14T12:00:00Z",
        ],
        "AppointmentDay": [
            "2016-04-10T00:00:00Z",
            "2016-04-18T00:00:00Z",
            "2016-04-20T00:00:00Z",
            "2016-04-25T00:00:00Z",
            "2016-05-01T00:00:00Z",
        ],
        "Age": [25, 45, 62, 18, 33],
        "Neighbourhood": [
            "JARDIM DA PENHA",
            "MARIA ORTIZ",
            "RESISTÊNCIA",
            "CENTRO",
            "TABUAZEIRO",
        ],
        "Scholarship": [0, 1, 0, 0, 1],
        "Hipertension": [0, 1, 1, 0, 0],
        "Diabetes": [0, 0, 1, 0, 0],
        "Alcoholism": [0, 0, 0, 0, 0],
        "Handcap": [0, 0, 0, 0, 0],
        "SMS_received": [0, 1, 1, 1, 1],
        "No-show": ["No", "Yes", "No", "Yes", "No"],
    })


def test_validate_raw_schema_missing_columns():
    """Missing required raw columns must raise DataQualityError."""
    df_incomplete = pd.DataFrame({"Age": [30], "Gender": ["F"]})
    with pytest.raises(DataQualityError):
        validate_raw_schema(df_incomplete)


def test_clean_filters_invalid_age(sample_raw_dataframe):
    """Ages < 0 or > 115 must be filtered out."""
    df = sample_raw_dataframe.copy()
    df.loc[0, "Age"] = -5
    df.loc[1, "Age"] = 130

    cleaned, audit = clean_appointment_data(df)
    assert audit["dropped_reasons"]["invalid_age"] == 2
    assert len(cleaned) == 3


def test_clean_filters_negative_waiting_days(sample_raw_dataframe):
    """Negative waiting days (AppointmentDay before ScheduledDay) must be filtered out."""
    df = sample_raw_dataframe.copy()
    # Schedule on April 15, appointment on April 10 -> negative waiting days
    df.loc[0, "ScheduledDay"] = "2016-04-15T08:00:00Z"
    df.loc[0, "AppointmentDay"] = "2016-04-10T00:00:00Z"

    cleaned, audit = clean_appointment_data(df)
    assert audit["dropped_reasons"]["negative_waiting_days"] == 1
    assert len(cleaned) == 4


def test_clean_target_encoding(sample_raw_dataframe):
    """Target variable 'No-show' must be correctly mapped to integer 0 and 1."""
    cleaned, _ = clean_appointment_data(sample_raw_dataframe)
    assert TARGET_COL in cleaned.columns
    assert set(cleaned[TARGET_COL].unique()).issubset({0, 1})
    # Row 0 was "No" -> 0; Row 1 was "Yes" -> 1
    assert cleaned.iloc[0][TARGET_COL] == 0
    assert cleaned.iloc[1][TARGET_COL] == 1


def test_feature_discipline_rejects_prohibited_columns():
    """Verify that including prohibited features raises LeakageError."""
    for prohibited in PROHIBITED_COLUMNS:
        with pytest.raises(LeakageError):
            verify_feature_discipline(["age", "waiting_days", prohibited])


def test_point_in_time_feature_matrix_columns(sample_raw_dataframe):
    """Engineered feature matrix must contain exactly the approved model features."""
    cleaned, _ = clean_appointment_data(sample_raw_dataframe)
    X, _ = derive_point_in_time_features(cleaned)

    # Must match ALL_MODEL_FEATURES
    assert list(X.columns) == ALL_MODEL_FEATURES

    # Ensure prohibited columns are strictly absent
    for prohibited in PROHIBITED_COLUMNS:
        assert prohibited not in X.columns


def test_chronological_split_integrity(sample_raw_dataframe):
    """Test that chronological split strictly maintains temporal ordering with no overlap."""
    # Expand sample to 30 rows
    records = []
    base_sched = pd.Timestamp("2016-04-01T08:00:00Z")
    for i in range(30):
        sched = base_sched + pd.Timedelta(days=i)
        app = sched + pd.Timedelta(days=5)
        records.append({
            "PatientId": 1000 + i,
            "AppointmentID": 5000 + i,
            "Gender": "F" if i % 2 == 0 else "M",
            "ScheduledDay": sched.isoformat(),
            "AppointmentDay": app.isoformat(),
            "Age": 20 + i,
            "Neighbourhood": "JARDIM DA PENHA",
            "Scholarship": 0,
            "Hipertension": 0,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 1,
            "No-show": "Yes" if i % 4 == 0 else "No",
        })
    df_large = pd.DataFrame(records)
    cleaned, _ = clean_appointment_data(df_large)

    X_train, y_train, X_val, y_val, X_test, y_test, meta, _ = split_chronologically(cleaned)

    # Check sum of partitions equals total
    assert len(X_train) + len(X_val) + len(X_test) == len(cleaned)

    # Check temporal ordering
    t_end = pd.Timestamp(meta["train"]["end_time"])
    v_start = pd.Timestamp(meta["validation"]["start_time"])
    v_end = pd.Timestamp(meta["validation"]["end_time"])
    test_start = pd.Timestamp(meta["test"]["start_time"])

    assert t_end <= v_start, "Train partition must precede Validation partition"
    assert v_end <= test_start, "Validation partition must precede Test partition"

