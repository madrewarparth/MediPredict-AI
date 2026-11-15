"""
Tests for Inference Pipeline, Validation Guards, and Batch Integrity.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.predict import (
    InferenceEngine,
    get_inference_engine,
    ValidationError,
    ModelArtifactError,
)
from src.config import RISK_BANDS


@pytest.fixture
def engine():
    """Returns singleton inference engine instance."""
    return get_inference_engine()


@pytest.fixture
def valid_single_appointment():
    return {
        "Age": 34,
        "Gender": "F",
        "Scholarship": 0,
        "Neighbourhood": "JARDIM DA PENHA",
        "ScheduledDay": "2016-05-01T08:30:00Z",
        "AppointmentDay": "2016-05-15T00:00:00Z",
    }


def test_inference_engine_initialization(engine):
    """Engine must load pipeline, metadata, and non-empty top neighbourhoods."""
    assert engine.pipeline is not None
    assert engine.metadata is not None
    assert len(engine.top_neighbourhoods) > 0
    assert 0.0 < engine.threshold < 1.0


def test_predict_single_returns_valid_probability(engine, valid_single_appointment):
    """Prediction probability must be a float between 0.0 and 1.0."""
    result = engine.predict_single(valid_single_appointment)
    prob = result["estimated_probability"]

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert result["risk_band"] in [
        RISK_BANDS["LOWER"]["name"],
        RISK_BANDS["MODERATE"]["name"],
        RISK_BANDS["HIGHER"]["name"],
    ]
    assert isinstance(result["review_flag"], bool)
    assert isinstance(result["factors"], list)


def test_predict_single_invalid_age_raises(engine, valid_single_appointment):
    """Invalid age values must raise ValidationError."""
    bad_rec = valid_single_appointment.copy()
    bad_rec["Age"] = -5
    with pytest.raises(ValidationError):
        engine.predict_single(bad_rec)

    bad_rec["Age"] = 150
    with pytest.raises(ValidationError):
        engine.predict_single(bad_rec)


def test_predict_single_negative_waiting_days_raises(engine, valid_single_appointment):
    """Negative waiting days (AppointmentDay before ScheduledDay) must raise ValidationError."""
    bad_rec = valid_single_appointment.copy()
    bad_rec["ScheduledDay"] = "2016-05-20T10:00:00Z"
    bad_rec["AppointmentDay"] = "2016-05-10T00:00:00Z"

    with pytest.raises(ValidationError, match="Temporal anomaly"):
        engine.predict_single(bad_rec)


def test_batch_prediction_row_count_and_order_preserved(engine):
    """Batch prediction must strictly preserve input row count, order, and columns."""
    sample_df = pd.DataFrame([
        {
            "AppointmentID": 1,
            "Age": 22,
            "Gender": "M",
            "Scholarship": 1,
            "Neighbourhood": "CENTRO",
            "ScheduledDay": "2016-05-01T09:00:00Z",
            "AppointmentDay": "2016-05-01T00:00:00Z",  # Same-day
        },
        {
            "AppointmentID": 2,
            "Age": 70,
            "Gender": "F",
            "Scholarship": 0,
            "Neighbourhood": "JARDIM DA PENHA",
            "ScheduledDay": "2016-05-01T10:00:00Z",
            "AppointmentDay": "2016-05-20T00:00:00Z",  # 19 days wait
        },
        {
            "AppointmentID": 3,
            "Age": 45,
            "Gender": "F",
            "Scholarship": 0,
            "Neighbourhood": "RESISTÊNCIA",
            "ScheduledDay": "2016-05-02T11:00:00Z",
            "AppointmentDay": "2016-05-05T00:00:00Z",
        },
    ])

    scored_df = engine.predict_batch(sample_df)

    # 1. Same row count
    assert len(scored_df) == len(sample_df)
    # 2. Same row order
    assert list(scored_df["AppointmentID"]) == [1, 2, 3]
    # 3. New score columns exist
    assert "estimated_no_show_prob" in scored_df.columns
    assert "risk_band" in scored_df.columns
    assert "review_priority" in scored_df.columns
    # 4. Valid probability values
    for val in scored_df["estimated_no_show_prob"]:
        assert 0.0 <= val <= 1.0


def test_missing_model_artifact_raises():
    """Initializing engine with non-existent path raises ModelArtifactError."""
    with pytest.raises(ModelArtifactError):
        InferenceEngine(pipeline_path=Path("non_existent_model.joblib"))



def test_missing_gender_is_rejected_not_fabricated(engine, valid_single_appointment):
    rec = {k: v for k, v in valid_single_appointment.items() if k != "Gender"}
    with pytest.raises(ValidationError, match="Gender"):
        engine.predict_single(rec)


def test_explanations_come_from_the_model(engine, valid_single_appointment):
    long_wait = {**valid_single_appointment, "AppointmentDay": "2016-06-20T00:00:00Z"}
    res = engine.predict_single(long_wait)
    lead = [f for f in res["factors"] if f["feature"] == "waiting_days"]
    assert lead and lead[0]["delta"] > 0, "A 50-day lead time should raise risk versus a typical booking"

    same_day = {**valid_single_appointment, "AppointmentDay": "2016-05-01T00:00:00Z"}
    res = engine.predict_single(same_day)
    lead = [f for f in res["factors"] if f["feature"] == "waiting_days"]
    assert lead and lead[0]["delta"] < 0, "Same-day booking should lower risk"


def test_metadata_matches_feature_config(engine):
    from src.config import ALL_MODEL_FEATURES

    assert engine.metadata["features"]["all_model_features"] == ALL_MODEL_FEATURES
    cut = engine.band_cutoffs
    assert 0 < cut["moderate"] < cut["higher"] < 1


def test_unseen_weekday_is_warned(engine, valid_single_appointment):
    # 2016-05-15 is a Sunday; the training data has no Sunday appointments
    res = engine.predict_single(valid_single_appointment)
    assert any("Sunday" in w for w in res["warnings"])
    weekday_rec = {**valid_single_appointment, "AppointmentDay": "2016-05-16T00:00:00Z"}
    assert engine.predict_single(weekday_rec)["warnings"] == []
