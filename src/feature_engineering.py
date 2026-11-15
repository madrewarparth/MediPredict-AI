"""
MediPredict AI - Point-in-Time Feature Engineering & Chronological Splitting
Enforces strict booking-time feature discipline, prevents temporal data leakage,
and builds reproducible scikit-learn transformers.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import (
    MODEL_NUMERIC_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_BINARY_FEATURES,
    ALL_MODEL_FEATURES,
    TOP_NEIGHBOURHOODS_COUNT,
    TARGET_COL,
    SPLIT_RATIOS,
)


class LeakageError(ValueError):
    """Raised when prohibited or future-leaking features enter the model matrix."""
    pass


PROHIBITED_COLUMNS = {
    "PatientId",
    "AppointmentID",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
    "no_show",
}


def verify_feature_discipline(features: List[str]) -> None:
    """
    Enforces strict point-in-time and governance policies:
      - No patient or record identifiers
      - No sensitive clinical diagnosis/treatment variables
      - No post-booking operational events (e.g. SMS received)
      - No target variable in feature matrix
    """
    violations = [col for col in features if col in PROHIBITED_COLUMNS]
    if violations:
        raise LeakageError(
            f"CRITICAL: Prohibited columns detected in model feature set: {violations}. "
            f"Violates booking-time discipline and non-clinical operational governance."
        )


def derive_point_in_time_features(
    df: pd.DataFrame, top_neighbourhoods: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts strictly booking-time operational and demographic features.
    If top_neighbourhoods is None, calculates top neighbourhoods from df.
    """
    df_feat = df.copy()

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_feat["ScheduledDay"]):
        df_feat["ScheduledDay"] = pd.to_datetime(df_feat["ScheduledDay"], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(df_feat["AppointmentDay"]):
        df_feat["AppointmentDay"] = pd.to_datetime(df_feat["AppointmentDay"], utc=True)

    # Lead / waiting time
    if "waiting_days" not in df_feat.columns:
        sched_date = df_feat["ScheduledDay"].dt.floor("D")
        app_date = df_feat["AppointmentDay"].dt.floor("D")
        df_feat["waiting_days"] = (app_date - sched_date).dt.days

    # Same-day booking indicator and log-scaled lead time
    df_feat["is_same_day"] = (df_feat["waiting_days"] == 0).astype(int)
    df_feat["log_waiting_days"] = np.log1p(df_feat["waiting_days"].clip(lower=0))

    # Booking moment temporal features
    df_feat["scheduled_hour"] = df_feat["ScheduledDay"].dt.hour
    df_feat["scheduled_day_of_week"] = df_feat["ScheduledDay"].dt.day_name()
    df_feat["appointment_day_of_week"] = df_feat["AppointmentDay"].dt.day_name()

    # Demographic features
    df_feat["age"] = df_feat["Age"].astype(int)
    df_feat["gender"] = df_feat["Gender"].astype(str).str.upper()
    df_feat["scholarship"] = df_feat["Scholarship"].astype(int)

    # Neighbourhood grouping (prevent high-cardinality overfitting).
    # Names are normalised to stripped upper-case so training and inference match.
    df_feat["Neighbourhood"] = df_feat["Neighbourhood"].astype(str).str.strip().str.upper()
    if top_neighbourhoods is None:
        top_neighbourhoods = (
            df_feat["Neighbourhood"].value_counts().nlargest(TOP_NEIGHBOURHOODS_COUNT).index.tolist()
        )

    df_feat["neighbourhood_grouped"] = df_feat["Neighbourhood"].where(
        df_feat["Neighbourhood"].isin(top_neighbourhoods), "Other"
    )

    # Select only approved model features
    X = df_feat[ALL_MODEL_FEATURES].copy()
    verify_feature_discipline(list(X.columns))

    return X, top_neighbourhoods


def build_preprocessor() -> ColumnTransformer:
    """
    Builds the scikit-learn ColumnTransformer for preprocessing:
      - Standardizes numeric features (age, waiting_days, scheduled_hour)
      - One-hot encodes categoricals with unknown category handling
      - Passes through binary flags (scholarship, is_same_day)
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, MODEL_CATEGORICAL_FEATURES),
            ("bin", "passthrough", MODEL_BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_chronologically(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any], List[str]]:
    """
    Performs chronological train/validation/test split based on ScheduledDay.
    Guarantees no future appointments leak into training.
    """
    # 1. Sort by ScheduledDay
    df_sorted = df.sort_values(by="ScheduledDay").reset_index(drop=True)

    n_total = len(df_sorted)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])

    train_raw = df_sorted.iloc[:n_train].copy()
    val_raw = df_sorted.iloc[n_train : n_train + n_val].copy()
    test_raw = df_sorted.iloc[n_train + n_val :].copy()

    # 2. Extract top neighbourhoods strictly from training partition
    top_neighbourhoods = (
        train_raw["Neighbourhood"]
        .astype(str).str.strip().str.upper()
        .value_counts()
        .nlargest(TOP_NEIGHBOURHOODS_COUNT)
        .index.tolist()
    )

    # 3. Derive features
    X_train, _ = derive_point_in_time_features(train_raw, top_neighbourhoods)
    y_train = train_raw[TARGET_COL].astype(int)

    X_val, _ = derive_point_in_time_features(val_raw, top_neighbourhoods)
    y_val = val_raw[TARGET_COL].astype(int)

    X_test, _ = derive_point_in_time_features(test_raw, top_neighbourhoods)
    y_test = test_raw[TARGET_COL].astype(int)

    split_metadata = {
        "strategy": "Chronological (ScheduledDay)",
        "train": {
            "samples": len(train_raw),
            "no_show_rate": round(float(y_train.mean() * 100), 2),
            "start_time": str(train_raw["ScheduledDay"].min()),
            "end_time": str(train_raw["ScheduledDay"].max()),
        },
        "validation": {
            "samples": len(val_raw),
            "no_show_rate": round(float(y_val.mean() * 100), 2),
            "start_time": str(val_raw["ScheduledDay"].min()),
            "end_time": str(val_raw["ScheduledDay"].max()),
        },
        "test": {
            "samples": len(test_raw),
            "no_show_rate": round(float(y_test.mean() * 100), 2),
            "start_time": str(test_raw["ScheduledDay"].min()),
            "end_time": str(test_raw["ScheduledDay"].max()),
        },
    }

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        split_metadata,
        top_neighbourhoods,
    )


if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess

    df_clean, _ = load_and_preprocess()
    X_train, y_train, X_val, y_val, X_test, y_test, meta, top_n = split_chronologically(df_clean)
    print("Chronological split completed!")
    print(f"Train: {meta['train']['samples']} rows ({meta['train']['start_time']} to {meta['train']['end_time']})")
    print(f"Val:   {meta['validation']['samples']} rows ({meta['validation']['start_time']} to {meta['validation']['end_time']})")
    print(f"Test:  {meta['test']['samples']} rows ({meta['test']['start_time']} to {meta['test']['end_time']})")
    print(f"Features: {list(X_train.columns)}")

