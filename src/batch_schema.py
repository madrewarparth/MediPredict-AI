"""
MediPredict AI - Batch CSV schema handling
Maps common column spellings in hospital scheduling exports onto the raw booking
schema the inference engine expects, and provides a downloadable template.
"""

from typing import Dict, List

import pandas as pd

# canonical column -> accepted spellings (compared lower-case, without _ - or spaces)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "AppointmentRef": ["appointmentref", "appointmentid", "aptid", "refid", "referenceid", "ref", "bookingid"],
    "ScheduledSlot": ["scheduledslot", "slot", "scheduledtime", "appointmenttime"],
    "LeadDays": ["leaddays", "leadtime", "waitingdays", "waitdays"],
    "ClinicSpecialty": ["clinicspecialty", "specialty", "speciality", "department", "clinic"],
    "NoShowProb": ["noshowprob", "noshowprobability", "probability", "prob", "riskprob"],
    "RiskBand": ["riskband", "riskcategory", "risklevel", "risk"],
    "ActionStatus": ["actionstatus", "status", "triagestatus", "reviewstatus"],
    "Age": ["age", "patientage"],
    "Gender": ["gender", "sex"],
    "ScheduledDay": ["scheduledday", "scheduleddate", "schedday", "bookingdate", "bookedat", "scheduled"],
    "AppointmentDay": ["appointmentday", "appointmentdate", "visitdate", "clinicdate", "appointment"],
    "Scholarship": ["scholarship", "welfare", "bolsafamilia"],
    "Neighbourhood": ["neighbourhood", "neighborhood", "location", "catchment", "area"],
}

_LOOKUP = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}

# The columns the model needs (Neighbourhood and Scholarship are optional).
MODEL_INPUT_COLUMNS = ["Age", "Gender", "ScheduledDay", "AppointmentDay"]
OPTIONAL_MODEL_COLUMNS = ["Neighbourhood", "Scholarship"]


def _key(col: str) -> str:
    return str(col).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames recognised columns to their canonical names (case/separator-insensitive).
    Unrecognised columns are kept as-is. If two source columns map to the same
    canonical name, only the first is renamed so no data is silently overwritten.
    """
    col_map, used = {}, set()
    for col in df.columns:
        canon = _LOOKUP.get(_key(col))
        if canon and canon not in used and canon not in df.columns.drop(col, errors="ignore"):
            col_map[col] = canon
            used.add(canon)
    return df.rename(columns=col_map)


def missing_model_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in MODEL_INPUT_COLUMNS if c not in df.columns]


def sample_booking_template() -> pd.DataFrame:
    """A small raw booking export the model can score (one row is deliberately invalid)."""
    return pd.DataFrame([
        {"AppointmentRef": "APT-1001", "Age": 34, "Gender": "F", "Neighbourhood": "JARDIM DA PENHA", "Scholarship": 0,
         "ScheduledDay": "2026-11-16 09:30", "AppointmentDay": "2026-11-16"},
        {"AppointmentRef": "APT-1002", "Age": 22, "Gender": "M", "Neighbourhood": "CENTRO", "Scholarship": 1,
         "ScheduledDay": "2026-11-16 10:05", "AppointmentDay": "2026-12-14"},
        {"AppointmentRef": "APT-1003", "Age": 67, "Gender": "F", "Neighbourhood": "MARIA ORTIZ", "Scholarship": 0,
         "ScheduledDay": "2026-11-16 11:40", "AppointmentDay": "2026-11-20"},
        {"AppointmentRef": "APT-1004", "Age": 8, "Gender": "M", "Neighbourhood": "RESISTÊNCIA", "Scholarship": 1,
         "ScheduledDay": "2026-11-17 08:15", "AppointmentDay": "2026-12-29"},
        {"AppointmentRef": "APT-1005", "Age": 45, "Gender": "F", "Neighbourhood": "ITARARÉ", "Scholarship": 0,
         "ScheduledDay": "2026-11-17 14:20", "AppointmentDay": "2026-11-27"},
        {"AppointmentRef": "APT-1006", "Age": 29, "Gender": "F", "Neighbourhood": "BONFIM", "Scholarship": 1,
         "ScheduledDay": "2026-11-18 16:45", "AppointmentDay": "2027-01-08"},
        {"AppointmentRef": "APT-1007", "Age": 51, "Gender": "M", "Neighbourhood": "SANTA MARTHA", "Scholarship": 0,
         "ScheduledDay": "2026-11-20 09:00", "AppointmentDay": "2026-11-18"},
    ])
