"""
Tests for bulk CSV schema handling and batch scoring of a review queue.
"""

import pandas as pd
import pytest

from src.batch_schema import (
    missing_model_columns,
    normalize_dataframe_columns,
    sample_booking_template,
)
from src.config import RISK_BANDS
from src.predict import get_inference_engine


@pytest.fixture
def engine():
    return get_inference_engine()


def test_aliases_map_to_canonical_names():
    df = pd.DataFrame(columns=["appointment_id", "patient_age", "Sex", "booking_date", "appointment_date",
                               "Neighborhood", "welfare"])
    out = normalize_dataframe_columns(df)
    assert list(out.columns) == ["AppointmentRef", "Age", "Gender", "ScheduledDay", "AppointmentDay",
                                 "Neighbourhood", "Scholarship"]


def test_neighbourhood_is_not_renamed_to_clinic_specialty():
    """Regression: Neighbourhood used to be renamed away, so every row scored as 'Other'."""
    out = normalize_dataframe_columns(pd.DataFrame(columns=["Neighbourhood", "Age"]))
    assert "Neighbourhood" in out.columns
    assert "ClinicSpecialty" not in out.columns


def test_duplicate_aliases_do_not_overwrite():
    out = normalize_dataframe_columns(pd.DataFrame(columns=["age", "patient_age"]))
    assert list(out.columns).count("Age") == 1


def test_missing_columns_reported():
    df = normalize_dataframe_columns(pd.DataFrame(columns=["AppointmentRef", "NoShowProb", "LeadDays"]))
    assert set(missing_model_columns(df)) == {"Age", "Gender", "ScheduledDay", "AppointmentDay"}


def test_sample_template_scores_with_one_rejected_row(engine):
    tmpl = normalize_dataframe_columns(sample_booking_template())
    assert missing_model_columns(tmpl) == []
    scored = engine.predict_batch(tmpl)
    assert len(scored) == len(tmpl)
    assert list(scored["AppointmentRef"]) == list(tmpl["AppointmentRef"])
    bad = scored["estimated_no_show_prob"].isna()
    assert bad.sum() == 1
    assert "Temporal anomaly" in scored.loc[bad, "validation_error"].iloc[0]
    assert scored.loc[~bad, "estimated_no_show_prob"].nunique() > 1


def test_neighbourhood_value_reaches_model(engine):
    base = {"Age": 30, "Gender": "F", "Scholarship": 0, "ScheduledDay": "2016-05-02T09:00:00Z",
            "AppointmentDay": "2016-05-20T00:00:00Z"}
    known = engine.top_neighbourhoods[0]
    rec = engine.validate_record_dict({**base, "Neighbourhood": known.lower()})
    assert engine.prepare_feature_row(rec)["neighbourhood_grouped"].iloc[0] == known


def test_band_matches_review_flag(engine):
    """Anything flagged must be Moderate or Higher; anything not flagged must be Lower."""
    df = pd.DataFrame([
        {"Age": a, "Gender": "F", "Scholarship": s, "Neighbourhood": "CENTRO",
         "ScheduledDay": "2016-05-02T09:00:00Z", "AppointmentDay": app}
        for a in (5, 25, 45, 80) for s in (0, 1)
        for app in ("2016-05-02T00:00:00Z", "2016-05-10T00:00:00Z", "2016-06-20T00:00:00Z")
    ])
    for thr in (None, 0.10, 0.30, 0.60):
        scored = engine.predict_batch(df, custom_threshold=thr)
        flagged = scored["flagged_for_review"] == "Yes"
        assert (scored.loc[~flagged, "risk_band"] == RISK_BANDS["LOWER"]["name"]).all()
        assert (scored.loc[flagged, "risk_band"] != RISK_BANDS["LOWER"]["name"]).all()


def test_batch_matches_single(engine):
    tmpl = sample_booking_template().iloc[:6]
    scored = engine.predict_batch(tmpl)
    for rec, p in zip(tmpl.to_dict(orient="records"), scored["estimated_no_show_prob"], strict=True):
        assert abs(engine.predict_single(rec)["estimated_probability"] - p) < 1e-4
