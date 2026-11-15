"""
MediPredict AI - Data Preprocessing & Validation
Handles raw data ingestion, schema validation, data quality checks,
temporal parsing, anomaly filtering, and target transformation.
"""

from typing import Dict, Tuple, Any
import pandas as pd
from src.config import (
    RAW_DATA_FILE,
    RAW_REQUIRED_COLUMNS,
    RAW_TARGET_COL,
    TARGET_COL,
    PROCESSED_DATA_FILE,
    PROCESSED_CSV_FILE,
    DATA_PROCESSED_DIR,
)


class DataQualityError(ValueError):
    """Raised when data fails critical schema or quality validation checks."""
    pass


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Validates that incoming raw DataFrame contains all required columns.
    Raises DataQualityError if any required column is missing.
    """
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataQualityError(
            f"Raw dataset is missing required schema columns: {missing}"
        )


def clean_appointment_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw appointment data and conducts data-quality filtering:
      1. Schema verification
      2. Duplicate removal
      3. Date parsing and UTC normalization
      4. Lead/waiting time calculation in days
      5. Filtering anomalous records:
         - Invalid age (< 0 or > 115)
         - Negative waiting time (AppointmentDay scheduled before ScheduledDay)
      6. Target variable binary mapping: 'Yes' -> 1, 'No' -> 0

    Returns:
      cleaned_df: Cleaned pandas DataFrame
      audit_report: Dictionary with data-quality metrics and filtering counts
    """
    validate_raw_schema(df)
    initial_rows = len(df)
    audit: Dict[str, Any] = {"initial_rows": initial_rows, "dropped_reasons": {}}

    # Create explicit copy to prevent chained assignment warnings
    df_clean = df.copy()

    # 1. Deduplication
    dups = df_clean.duplicated(subset=["AppointmentID"]).sum()
    if dups > 0:
        df_clean = df_clean.drop_duplicates(subset=["AppointmentID"])
    audit["dropped_reasons"]["duplicate_appointment_id"] = int(dups)

    # 2. Parse Timestamps
    df_clean["ScheduledDay"] = pd.to_datetime(df_clean["ScheduledDay"], utc=True)
    df_clean["AppointmentDay"] = pd.to_datetime(df_clean["AppointmentDay"], utc=True)

    # Extract date components for waiting days calculation
    sched_date = df_clean["ScheduledDay"].dt.floor("D")
    app_date = df_clean["AppointmentDay"].dt.floor("D")
    df_clean["waiting_days"] = (app_date - sched_date).dt.days

    # 3. Filter Negative Waiting Days (Temporal Causality Anomaly)
    neg_wait_mask = df_clean["waiting_days"] < 0
    neg_wait_count = int(neg_wait_mask.sum())
    audit["dropped_reasons"]["negative_waiting_days"] = neg_wait_count
    df_clean = df_clean[~neg_wait_mask]

    # 4. Filter Impossible Ages (Physiological Bounds: 0 to 115)
    invalid_age_mask = (df_clean["Age"] < 0) | (df_clean["Age"] > 115)
    invalid_age_count = int(invalid_age_mask.sum())
    audit["dropped_reasons"]["invalid_age"] = invalid_age_count
    df_clean = df_clean[~invalid_age_mask]

    # 5. Filter Missing or Malformed Values
    null_counts = int(df_clean.isnull().any(axis=1).sum())
    audit["dropped_reasons"]["null_records"] = null_counts
    if null_counts > 0:
        df_clean = df_clean.dropna()

    # 6. Target Encoding
    target_series = df_clean[RAW_TARGET_COL].astype(str).str.strip().str.capitalize()
    invalid_target = ~target_series.isin(["Yes", "No"])
    if invalid_target.sum() > 0:
        audit["dropped_reasons"]["invalid_target_values"] = int(invalid_target.sum())
        df_clean = df_clean[~invalid_target]
        target_series = target_series[~invalid_target]

    df_clean[TARGET_COL] = (target_series == "Yes").astype(int)

    # Summary metrics
    final_rows = len(df_clean)
    audit["final_rows"] = final_rows
    audit["total_dropped"] = initial_rows - final_rows
    audit["retention_rate"] = round(final_rows / initial_rows * 100, 2)
    audit["no_show_count"] = int(df_clean[TARGET_COL].sum())
    audit["attended_count"] = int(final_rows - df_clean[TARGET_COL].sum())
    audit["no_show_rate"] = round(df_clean[TARGET_COL].mean() * 100, 2)

    return df_clean, audit


def load_and_preprocess(raw_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV from specified path (or default config path) and runs full cleaning.
    """
    path = raw_path or RAW_DATA_FILE
    df_raw = pd.read_csv(path)
    cleaned_df, audit = clean_appointment_data(df_raw)
    return cleaned_df, audit


def save_processed_data(df: pd.DataFrame) -> None:
    """Saves cleaned DataFrame to both parquet and csv for efficient storage and inspection."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_FILE, index=False)
    # Save a sample or full CSV for review
    df.to_csv(PROCESSED_CSV_FILE, index=False)


if __name__ == "__main__":
    df_clean, report = load_and_preprocess()
    save_processed_data(df_clean)
    print("Preprocessing completed successfully!")
    print(f"Initial: {report['initial_rows']:,} | Final: {report['final_rows']:,} | Dropped: {report['total_dropped']:,}")
    print(f"No-show rate: {report['no_show_rate']}% | Dropped details: {report['dropped_reasons']}")

