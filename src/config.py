"""
MediPredict AI - Configuration & Policy Constants
Defines paths, feature dictionaries, point-in-time policies,
risk thresholds, and theme tokens.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory & File Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ASSETS_DIR = BASE_DIR / "assets"

RAW_DATA_FILE = DATA_RAW_DIR / "appointments.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "appointments_clean.parquet"
PROCESSED_CSV_FILE = DATA_PROCESSED_DIR / "appointments_clean.csv"
MODEL_PIPELINE_FILE = MODELS_DIR / "appointment_pipeline.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Raw Data Schema & Column Definitions
# ---------------------------------------------------------------------------
RAW_TARGET_COL = "No-show"
TARGET_COL = "no_show"

RAW_REQUIRED_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
]

# ---------------------------------------------------------------------------
# Feature Dictionary & Leakage Policy
# ---------------------------------------------------------------------------
FEATURE_POLICY = {
    "PatientId": {
        "status": "EXCLUDED",
        "reason": "Direct patient identifier. Prone to data leakage and memorization; does not generalize.",
        "category": "Identifier",
    },
    "AppointmentID": {
        "status": "EXCLUDED",
        "reason": "Internal primary key. Arbitrary sequential counter with zero predictive validity.",
        "category": "Identifier",
    },
    "Hipertension": {
        "status": "EXCLUDED",
        "reason": "Clinical diagnosis variable. Excluded per non-clinical operations governance policy.",
        "category": "Clinical",
    },
    "Diabetes": {
        "status": "EXCLUDED",
        "reason": "Clinical diagnosis variable. Excluded per non-clinical operations governance policy.",
        "category": "Clinical",
    },
    "Alcoholism": {
        "status": "EXCLUDED",
        "reason": "Sensitive clinical indicator. Excluded to prevent clinical bias and respect scope.",
        "category": "Clinical",
    },
    "Handcap": {
        "status": "EXCLUDED",
        "reason": "Physical condition/disability indicator. Excluded under strict non-clinical mandate.",
        "category": "Clinical",
    },
    "SMS_received": {
        "status": "EXCLUDED",
        "reason": "Post-booking event. Occurs after the scheduling decision; violates booking-time discipline.",
        "category": "Post-Booking Event",
    },
    "Age": {
        "status": "INCLUDED",
        "engineered_name": "age",
        "reason": "Available at booking. Cleaned to valid physiological bounds (0-115).",
        "category": "Demographic",
    },
    "Gender": {
        "status": "INCLUDED",
        "engineered_name": "gender",
        "reason": "Standard administrative demographic known at booking time.",
        "category": "Demographic",
    },
    "Neighbourhood": {
        "status": "INCLUDED",
        "engineered_name": "neighbourhood_grouped",
        "reason": "Clinic / geographic catchment area. Top 25 retained, low-frequency grouped into 'Other'.",
        "category": "Geographic",
    },
    "Scholarship": {
        "status": "INCLUDED",
        "engineered_name": "scholarship",
        "reason": "Social welfare program enrollment (Bolsa Família). Operational factor known at booking.",
        "category": "Socioeconomic Indicator",
    },
    "ScheduledDay": {
        "status": "INCLUDED_AS_DERIVED",
        "engineered_name": ["scheduled_hour", "scheduled_day_of_week"],
        "reason": "Timestamp of booking creation. Used to extract temporal scheduling behavioral patterns.",
        "category": "Temporal",
    },
    "AppointmentDay": {
        "status": "INCLUDED_AS_DERIVED",
        "engineered_name": ["appointment_day_of_week", "waiting_days", "is_same_day"],
        "reason": "Scheduled appointment date. Combined with ScheduledDay to calculate lead/waiting time.",
        "category": "Temporal",
    },
}

# Features passed to model matrix
# `log_waiting_days` is a monotone transform of lead time: attendance decays roughly
# logarithmically with lead time, so this lets linear models capture the curve.
MODEL_NUMERIC_FEATURES = ["age", "waiting_days", "log_waiting_days", "scheduled_hour"]
MODEL_CATEGORICAL_FEATURES = [
    "gender",
    "neighbourhood_grouped",
    "scheduled_day_of_week",
    "appointment_day_of_week",
]
MODEL_BINARY_FEATURES = ["scholarship", "is_same_day"]

ALL_MODEL_FEATURES = (
    MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES + MODEL_BINARY_FEATURES
)

# Human-readable names for the raw booking-time inputs (used in explanations)
FEATURE_LABELS = {
    "age": "Patient age",
    "waiting_days": "Lead time (days)",
    "log_waiting_days": "Lead time (log scale)",
    "scheduled_hour": "Booking hour",
    "gender": "Gender",
    "neighbourhood_grouped": "Neighbourhood",
    "scheduled_day_of_week": "Booking weekday",
    "appointment_day_of_week": "Appointment weekday",
    "scholarship": "Welfare enrolment",
    "is_same_day": "Same-day booking",
}

TOP_NEIGHBOURHOODS_COUNT = 25

# The bundled dataset in data/raw is produced by src/generate_data.py (synthetic,
# Kaggle-schema). Set to False when you replace it with real hospital data.
DATA_IS_SYNTHETIC = True

# ---------------------------------------------------------------------------
# Risk Band & Operational Thresholds
# ---------------------------------------------------------------------------
# Fallback only; the trained threshold lives in models/model_metadata.json.
DEFAULT_CLASSIFICATION_THRESHOLD = 0.30

# Band cut-offs are learned at training time and stored in metadata:
#   Lower    : p <  operating threshold           (not flagged)
#   Moderate : threshold <= p < higher cut-off    (flagged, routine review)
#   Higher   : p >= higher cut-off                (flagged, priority call)
# The higher cut-off is the validation-set quantile below, i.e. the top
# (1 - quantile) share of appointments by estimated risk.
HIGHER_BAND_QUANTILE = 0.85
DEFAULT_BAND_CUTOFFS = {"moderate": 0.25, "higher": 0.40}

RISK_BANDS = {
    "LOWER": {
        "name": "Lower estimated risk",
        "short": "Lower",
        "color": "#16A34A",
        "priority": "Standard operations",
        "action": "Standard automated reminder protocol.",
    },
    "MODERATE": {
        "name": "Moderate estimated risk",
        "short": "Moderate",
        "color": "#D97706",
        "priority": "Routine review",
        "action": "Flagged: add a secondary reminder if staff capacity allows.",
    },
    "HIGHER": {
        "name": "Higher estimated risk",
        "short": "Higher",
        "color": "#DC2626",
        "priority": "Priority review",
        "action": "Flagged: priority candidate for a staff reminder call ~48h before. Never auto-cancel.",
    },
}

# ---------------------------------------------------------------------------
# Visual Theme Tokens (Clinical Light)
# ---------------------------------------------------------------------------
THEME = {
    "app_bg": "#F6F8F7",
    "sidebar_bg": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "surface_elevated": "#EEF6F2",
    "primary_teal": "#0F766E",
    "teal_hover": "#115E59",
    "secondary_blue": "#2563EB",
    "text_main": "#0F172A",
    "text_muted": "#5B6B7F",
    "border": "#E2E8F0",
    "success": "#16A34A",
    "warning": "#D97706",
    "error": "#DC2626",
    "neutral": "#94A3B8",
    "attended": "#0F766E",
    "no_show": "#E11D48",
}

# ---------------------------------------------------------------------------
# Chronological Split Policy
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

