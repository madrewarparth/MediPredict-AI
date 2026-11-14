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
MODEL_NUMERIC_FEATURES = ["age", "waiting_days", "scheduled_hour"]
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

TOP_NEIGHBOURHOODS_COUNT = 25

# ---------------------------------------------------------------------------
# Risk Band & Operational Thresholds
# ---------------------------------------------------------------------------
DEFAULT_CLASSIFICATION_THRESHOLD = 0.30

RISK_BANDS = {
    "LOWER": {
        "name": "Lower estimated risk",
        "color": "#22C55E",
        "max_prob": 0.25,
        "action": "Standard operational reminder protocol",
    },
    "MODERATE": {
        "name": "Moderate estimated risk",
        "color": "#F59E0B",
        "min_prob": 0.25,
        "max_prob": 0.50,
        "action": "Consider secondary reminder review if capacity permits",
    },
    "HIGHER": {
        "name": "Higher estimated risk",
        "color": "#F87171",
        "min_prob": 0.50,
        "action": "Priority candidate for staff outreach review (No auto-cancellation)",
    },
}

# ---------------------------------------------------------------------------
# Visual Theme Tokens (Clinical Intelligence - Clinical Light)
# ---------------------------------------------------------------------------
THEME = {
    "app_bg": "#f8fafc",
    "sidebar_bg": "#ffffff",
    "card_bg": "#ffffff",
    "surface_elevated": "#f1f5f9",
    "primary_teal": "#0284c7",  # Sky blue 600
    "teal_hover": "#0369a1",  # Sky blue 700
    "secondary_blue": "#0284c7",
    "text_main": "#0f172a",  # Slate 900
    "text_muted": "#64748b",  # Slate 500
    "border": "#e2e8f0",  # Slate 200
    "success": "#10b981",  # Emerald 500
    "warning": "#f59e0b",  # Amber 500
    "error": "#ef4444",  # Red 500
    "neutral": "#94a3b8",  # Slate 400
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

