"""
MediPredict AI - Production Inference Engine
Handles single and batch inference, robust schema and range validation,
point-in-time feature transformation, and risk band classification.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODEL_PIPELINE_FILE,
    MODEL_METADATA_FILE,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    RISK_BANDS,
    ALL_MODEL_FEATURES,
)


class ModelArtifactError(FileNotFoundError):
    """Raised when model artifact or metadata is missing or corrupted."""
    pass


class ValidationError(ValueError):
    """Raised when inference input data is malformed or out of valid range."""
    pass


class InferenceEngine:
    """Production inference wrapper around the fitted preprocessing-and-model pipeline."""

    def __init__(
        self,
        pipeline_path: Path = MODEL_PIPELINE_FILE,
        metadata_path: Path = MODEL_METADATA_FILE,
    ):
        self.pipeline_path = Path(pipeline_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline = None
        self.metadata = None
        self.top_neighbourhoods = []
        self.threshold = DEFAULT_CLASSIFICATION_THRESHOLD
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads fitted pipeline and companion metadata from disk."""
        if not self.pipeline_path.exists():
            raise ModelArtifactError(
                f"Model pipeline artifact not found at: {self.pipeline_path}. "
                "Ensure training workflow has been run."
            )
        if not self.metadata_path.exists():
            raise ModelArtifactError(
                f"Model metadata file not found at: {self.metadata_path}."
            )

        try:
            self.pipeline = joblib.load(self.pipeline_path)
        except Exception as e:
            raise ModelArtifactError(f"Failed loading joblib pipeline: {e}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.top_neighbourhoods = self.metadata.get("features", {}).get(
                "top_neighbourhoods", []
            )
            self.threshold = self.metadata.get(
                "selected_operating_threshold", DEFAULT_CLASSIFICATION_THRESHOLD
            )
        except Exception as e:
            raise ModelArtifactError(f"Failed parsing metadata JSON: {e}")

    def validate_record_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and cleans a single input appointment dictionary."""
        # Age validation
        try:
            age = int(record["Age"])
        except (KeyError, ValueError, TypeError):
            raise ValidationError("Field 'Age' is required and must be an integer.")
        if age < 0 or age > 115:
            raise ValidationError(
                f"Invalid Age: {age}. Must be between 0 and 115 years."
            )

        # Gender validation
        gender = str(record.get("Gender", "F")).strip().upper()
        if gender not in ["F", "M"]:
            raise ValidationError(f"Invalid Gender: '{gender}'. Must be 'F' or 'M'.")

        # Scholarship
        try:
            scholarship = int(record.get("Scholarship", 0))
            if scholarship not in [0, 1]:
                scholarship = 1 if scholarship > 0 else 0
        except (ValueError, TypeError):
            scholarship = 0

        # Neighbourhood
        neighbourhood = str(record.get("Neighbourhood", "Other")).strip().upper()

        # Date validations
        sched_val = record.get("ScheduledDay")
        app_val = record.get("AppointmentDay")
        if not sched_val or not app_val:
            raise ValidationError(
                "Both 'ScheduledDay' and 'AppointmentDay' timestamps are required."
            )

        try:
            sched_dt = pd.to_datetime(sched_val, utc=True)
        except Exception as e:
            raise ValidationError(f"Could not parse 'ScheduledDay' timestamp: {sched_val}. Error: {e}")

        try:
            app_dt = pd.to_datetime(app_val, utc=True)
        except Exception as e:
            raise ValidationError(f"Could not parse 'AppointmentDay' timestamp: {app_val}. Error: {e}")

        sched_date = sched_dt.floor("D")
        app_date = app_dt.floor("D")
        waiting_days = (app_date - sched_date).days

        if waiting_days < 0:
            raise ValidationError(
                f"Temporal anomaly: AppointmentDay ({app_date.strftime('%Y-%m-%d')}) "
                f"cannot occur before ScheduledDay ({sched_date.strftime('%Y-%m-%d')}). "
                f"Waiting days cannot be negative ({waiting_days})."
            )

        return {
            "Age": age,
            "Gender": gender,
            "Scholarship": scholarship,
            "Neighbourhood": neighbourhood,
            "ScheduledDay": sched_dt,
            "AppointmentDay": app_dt,
            "waiting_days": waiting_days,
            "is_same_day": 1 if waiting_days == 0 else 0,
            "scheduled_hour": int(sched_dt.hour),
            "scheduled_day_of_week": sched_dt.day_name(),
            "appointment_day_of_week": app_dt.day_name(),
        }

    def prepare_feature_row(self, clean_rec: Dict[str, Any]) -> pd.DataFrame:
        """Prepares a single-row DataFrame formatted with exact model features."""
        neighbourhood = clean_rec["Neighbourhood"]
        neighbourhood_grouped = (
            neighbourhood if neighbourhood in self.top_neighbourhoods else "Other"
        )

        row_dict = {
            "age": clean_rec["Age"],
            "waiting_days": clean_rec["waiting_days"],
            "scheduled_hour": clean_rec["scheduled_hour"],
            "gender": clean_rec["Gender"],
            "neighbourhood_grouped": neighbourhood_grouped,
            "scheduled_day_of_week": clean_rec["scheduled_day_of_week"],
            "appointment_day_of_week": clean_rec["appointment_day_of_week"],
            "scholarship": clean_rec["Scholarship"],
            "is_same_day": clean_rec["is_same_day"],
        }
        return pd.DataFrame([row_dict])[ALL_MODEL_FEATURES]

    def _determine_risk_band(self, prob: float) -> Dict[str, Any]:
        """Categorizes predicted probability into human-interpretable operational risk bands."""
        if prob < RISK_BANDS["LOWER"]["max_prob"]:
            band_info = RISK_BANDS["LOWER"]
            priority = "Standard Operations"
        elif prob < RISK_BANDS["MODERATE"]["max_prob"]:
            band_info = RISK_BANDS["MODERATE"]
            priority = "Medium Priority Review"
        else:
            band_info = RISK_BANDS["HIGHER"]
            priority = "High Priority Review"

        return {
            "band_name": band_info["name"],
            "color": band_info["color"],
            "priority": priority,
            "action": band_info["action"],
        }

    def predict_single(
        self, record: Dict[str, Any], custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates probability estimate, operational risk band, and feature
        breakdown for a single appointment booking record.
        """
        clean_rec = self.validate_record_dict(record)
        feature_df = self.prepare_feature_row(clean_rec)

        threshold = custom_threshold if custom_threshold is not None else self.threshold
        threshold = max(0.01, min(0.99, float(threshold)))

        # Pipeline returns probabilities
        prob_arr = self.pipeline.predict_proba(feature_df)
        prob = float(prob_arr[0, 1])

        # Flagged if prob >= operating threshold
        is_flagged = bool(prob >= threshold)
        risk = self._determine_risk_band(prob)

        # Transparent feature contribution explanation (heuristic comparison to baseline averages)
        factors = []
        if clean_rec["is_same_day"] == 1:
            factors.append({
                "factor": "Same-Day Booking",
                "influence": "Substantially Decreases Risk",
                "detail": "Appointment booked for the same calendar date. Historical same-day no-show rate is under 5%.",
            })
        elif clean_rec["waiting_days"] > 14:
            factors.append({
                "factor": "Extended Waiting Time (Lead Time)",
                "influence": "Increases Risk",
                "detail": f"{clean_rec['waiting_days']} days between booking and appointment. Longer lead times correlate with higher missed rates.",
            })
        elif clean_rec["waiting_days"] <= 3:
            factors.append({
                "factor": "Short Waiting Time",
                "influence": "Decreases Risk",
                "detail": f"Only {clean_rec['waiting_days']} day(s) lead time.",
            })

        if clean_rec["Age"] < 25:
            factors.append({
                "factor": "Younger Age Demographic",
                "influence": "Slightly Increases Risk",
                "detail": f"Age {clean_rec['Age']}. Younger adult cohorts historically show slightly elevated missed appointment frequencies.",
            })
        elif clean_rec["Age"] >= 60:
            factors.append({
                "factor": "Senior Age Demographic",
                "influence": "Decreases Risk",
                "detail": f"Age {clean_rec['Age']}. Older demographic cohorts show high consistency in clinic attendance.",
            })

        if clean_rec["Scholarship"] == 1:
            factors.append({
                "factor": "Social Welfare Enrollment (Bolsa Família)",
                "influence": "Moderate Risk Association",
                "detail": "Enrolled in social welfare assistance program.",
            })

        return {
            "estimated_probability": round(prob, 4),
            "probability_percent": round(prob * 100, 2),
            "threshold_used": round(threshold, 3),
            "review_flag": is_flagged,
            "status_label": "Review Candidate (Elevated Miss Probability)" if is_flagged else "Standard Protocol",
            "risk_band": risk["band_name"],
            "risk_color": risk["color"],
            "review_priority": risk["priority"],
            "operational_action": risk["action"],
            "waiting_days": clean_rec["waiting_days"],
            "is_same_day": bool(clean_rec["is_same_day"]),
            "factors": factors,
            "governance_notice": (
                "MediPredict AI provides statistical risk estimates to aid staff prioritization. "
                "This is not a certainty and must not be used to cancel appointments or deny care."
            ),
        }

    def predict_batch(
        self, df: pd.DataFrame, custom_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Scores a batch of appointments.
        Guarantees preserving original row count, row order, and original columns,
        augmenting with model predictions and risk bands.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        threshold = max(0.01, min(0.99, float(threshold)))

        res_df = df.copy()
        n_rows = len(res_df)

        probs = []
        bands = []
        priorities = []
        flags = []
        colors = []
        errors = []

        for idx, row in res_df.iterrows():
            rec_dict = row.to_dict()
            try:
                pred = self.predict_single(rec_dict, custom_threshold=threshold)
                probs.append(pred["estimated_probability"])
                bands.append(pred["risk_band"])
                priorities.append(pred["review_priority"])
                flags.append("Yes" if pred["review_flag"] else "No")
                colors.append(pred["risk_color"])
                errors.append(None)
            except Exception as e:
                probs.append(np.nan)
                bands.append("Not Scored")
                priorities.append("Validation Error")
                flags.append("N/A")
                colors.append(RISK_BANDS["LOWER"]["color"])
                errors.append(str(e))

        res_df["estimated_no_show_prob"] = probs
        res_df["risk_band"] = bands
        res_df["review_priority"] = priorities
        res_df["flagged_for_review"] = flags
        res_df["validation_error"] = errors

        assert len(res_df) == n_rows, "Batch inference must preserve exact row count!"
        return res_df


# Module-level singleton
_inference_engine_instance = None


def get_inference_engine() -> InferenceEngine:
    """Returns a cached singleton instance of InferenceEngine."""
    global _inference_engine_instance
    if _inference_engine_instance is None:
        _inference_engine_instance = InferenceEngine()
    return _inference_engine_instance


if __name__ == "__main__":
    engine = get_inference_engine()
    sample = {
        "Age": 29,
        "Gender": "F",
        "Scholarship": 1,
        "Neighbourhood": "JARDIM DA PENHA",
        "ScheduledDay": "2016-05-10T08:30:00Z",
        "AppointmentDay": "2016-05-25T00:00:00Z",
    }
    result = engine.predict_single(sample)
    print("Single prediction test result:")
    print(f"Prob: {result['estimated_probability']} | Risk Band: {result['risk_band']} | Flagged: {result['review_flag']}")

