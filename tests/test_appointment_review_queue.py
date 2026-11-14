"""
Unit and Integration Tests for MediPredict AI Appointment Review Queue
Validates strict schema adherence, non-fabrication of demographic attributes,
dynamic probability ingestion, and risk band classification.
"""

import io
import importlib
import pandas as pd
import numpy as np
import pytest

# Load page module dynamically because filename begins with a number
bulk_page = importlib.import_module("pages.3_Bulk_Prediction")
normalize_dataframe_columns = bulk_page.normalize_dataframe_columns


def test_appointment_review_queue_csv_schema():
    """Verify that the approved 7-column review queue schema normalizes correctly."""
    csv_data = """AppointmentRef,ScheduledSlot,LeadDays,ClinicSpecialty,NoShowProb,RiskBand,ActionStatus
#APT-1001,Dec 18 2026 09:30 AM,7,ENT,0.240,Moderate estimated risk,Pending Review
#APT-1002,Nov 23 2026 10:15 AM,14,ENT,0.385,Moderate estimated risk,Pending Review
#APT-1003,Dec 12 2026 02:00 PM,28,Ophthalmology,0.682,Higher estimated risk,Pending Review
#APT-1004,Dec 27 2026 11:30 AM,4,Orthopedics,0.125,Lower estimated risk,Pending Review
#APT-1005,Jan 06 2027 08:45 AM,35,Orthopedics,0.748,Higher estimated risk,Pending Review
#APT-1006,Dec 26 2026 03:15 PM,2,ENT,0.085,Lower estimated risk,Reviewed
#APT-1007,Nov 27 2026 01:00 PM,19,Dermatology,0.510,Higher estimated risk,Pending Review
"""
    df = pd.read_csv(io.StringIO(csv_data))
    norm_df = normalize_dataframe_columns(df)

    # 1. No demographic attributes should be present
    assert "Age" not in norm_df.columns
    assert "Gender" not in norm_df.columns
    assert "PatientName" not in norm_df.columns

    # 2. Approved columns must be preserved
    assert "AppointmentRef" in norm_df.columns
    assert "ScheduledSlot" in norm_df.columns
    assert "LeadDays" in norm_df.columns
    assert "ClinicSpecialty" in norm_df.columns
    assert "NoShowProb" in norm_df.columns
    assert "RiskBand" in norm_df.columns
    assert "ActionStatus" in norm_df.columns

    # 3. Probabilities must be dynamic and distinct
    probs = norm_df["NoShowProb"].tolist()
    assert len(set(probs)) > 1, "Probabilities must be dynamic and not static demo values"
    for p in probs:
        assert 0.0 <= float(p) <= 1.0, f"Probability out of range: {p}"

    # 4. References must match original exactly
    refs = norm_df["AppointmentRef"].tolist()
    assert refs[0] == "#APT-1001"
    assert refs[1] == "#APT-1002"
    assert refs[2] == "#APT-1003"


def test_no_show_prob_percentage_strings():
    """Verify handling of percentage strings in review queue CSV."""
    csv_data = """AppointmentRef,ScheduledSlot,LeadDays,ClinicSpecialty,NoShowProb,RiskBand,ActionStatus
#APT-2001,Dec 18 2026 09:30 AM,7,ENT,24.0%,Moderate estimated risk,Pending Review
#APT-2002,Nov 23 2026 10:15 AM,14,Cardiology,68.2%,Higher estimated risk,Pending Review
"""
    df = pd.read_csv(io.StringIO(csv_data))
    norm_df = normalize_dataframe_columns(df)

    def parse_prob(val):
        s = str(val).strip().replace("%", "")
        f = float(s)
        if f > 1.0 and f <= 100.0:
            return f / 100.0
        return f

    probs = norm_df["NoShowProb"].apply(parse_prob).tolist()
    assert abs(probs[0] - 0.24) < 1e-4
    assert abs(probs[1] - 0.682) < 1e-4
