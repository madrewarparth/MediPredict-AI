"""
MediPredict AI - Production Inference Engine
Handles single and batch inference, schema and range validation,
point-in-time feature transformation, risk banding and model-based explanations.
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
    DEFAULT_BAND_CUTOFFS,
    RISK_BANDS,
    ALL_MODEL_FEATURES,
    FEATURE_LABELS,
)


class ModelArtifactError(FileNotFoundError):
    """Raised when model artifact or metadata is missing or corrupted."""


class ValidationError(ValueError):
    """Raised when inference input data is malformed or out of valid range."""


MIN_AGE, MAX_AGE = 0, 115
# Raw booking columns the engine reads. Neighbourhood and Scholarship are optional.
REQUIRED_INPUT_COLUMNS = ["Age", "Gender", "ScheduledDay", "AppointmentDay"]
OPTIONAL_INPUT_COLUMNS = ["Neighbourhood", "Scholarship"]

_GENDER_ALIASES = {"F": "F", "FEMALE": "F", "W": "F", "M": "M", "MALE": "M"}


def _parse_timestamp(value: Any, field: str) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        raise ValidationError(f"Field '{field}' is required.")
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception as e:  # noqa: BLE001 - surface the parse problem to the user
        raise ValidationError(f"Could not parse '{field}' timestamp: {value!r} ({e}).") from e
    if pd.isna(ts):
        raise ValidationError(f"Field '{field}' is required.")
    return ts


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
        self.metadata: Dict[str, Any] = {}
        self.top_neighbourhoods: List[str] = []
        self.threshold = DEFAULT_CLASSIFICATION_THRESHOLD
        self.band_cutoffs = dict(DEFAULT_BAND_CUTOFFS)
        self.reference_profile: Dict[str, Any] = {}
        self._load_artifacts()

    # ------------------------------------------------------------------ loading
    def _load_artifacts(self) -> None:
        if not self.pipeline_path.exists():
            raise ModelArtifactError(
                f"Model pipeline artifact not found at: {self.pipeline_path}. "
                "Run `python -m src.train_model` first."
            )
        if not self.metadata_path.exists():
            raise ModelArtifactError(f"Model metadata file not found at: {self.metadata_path}.")

        try:
            self.pipeline = joblib.load(self.pipeline_path)
        except Exception as e:
            raise ModelArtifactError(f"Failed loading joblib pipeline: {e}") from e

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ModelArtifactError(f"Failed parsing metadata JSON: {e}") from e

        features = self.metadata.get("features", {})
        self.top_neighbourhoods = [str(n).strip().upper() for n in features.get("top_neighbourhoods", [])]
        self.threshold = float(self.metadata.get("selected_operating_threshold", DEFAULT_CLASSIFICATION_THRESHOLD))
        self.band_cutoffs = {**DEFAULT_BAND_CUTOFFS, **self.metadata.get("risk_band_cutoffs", {})}
        self.reference_profile = self.metadata.get("reference_profile", {})
        self.known_categories = self.metadata.get("known_categories", {})

        expected = features.get("all_model_features")
        if expected and list(expected) != list(ALL_MODEL_FEATURES):
            raise ModelArtifactError(
                "Model artifact was trained with a different feature set than src/config.py. "
                "Re-run `python -m src.train_model`."
            )

    # --------------------------------------------------------------- validation
    def validate_record_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalises one raw booking record. Raises ValidationError."""
        raw_age = record.get("Age")
        try:
            age_f = float(raw_age)
            if np.isnan(age_f) or age_f != int(age_f):
                raise ValueError
            age = int(age_f)
        except (ValueError, TypeError):
            raise ValidationError("Field 'Age' is required and must be a whole number.") from None
        if age < MIN_AGE or age > MAX_AGE:
            raise ValidationError(f"Invalid Age: {age}. Must be between {MIN_AGE} and {MAX_AGE} years.")

        raw_gender = record.get("Gender")
        if raw_gender is None or (isinstance(raw_gender, float) and np.isnan(raw_gender)):
            raise ValidationError("Field 'Gender' is required ('F' or 'M').")
        gender = _GENDER_ALIASES.get(str(raw_gender).strip().upper())
        if gender is None:
            raise ValidationError(f"Invalid Gender: '{raw_gender}'. Must be 'F' or 'M'.")

        raw_schol = record.get("Scholarship", 0)
        if raw_schol is None or (isinstance(raw_schol, float) and np.isnan(raw_schol)):
            raw_schol = 0
        schol_str = str(raw_schol).strip().lower()
        if schol_str in {"1", "1.0", "yes", "y", "true", "enrolled"}:
            scholarship = 1
        elif schol_str in {"0", "0.0", "no", "n", "false", "", "not enrolled"}:
            scholarship = 0
        else:
            raise ValidationError(f"Invalid Scholarship value: '{raw_schol}'. Use 0/1 or Yes/No.")

        raw_n = record.get("Neighbourhood")
        if raw_n is None or (isinstance(raw_n, float) and np.isnan(raw_n)) or str(raw_n).strip() == "":
            neighbourhood = "OTHER"
        else:
            neighbourhood = str(raw_n).strip().upper()

        sched_dt = _parse_timestamp(record.get("ScheduledDay"), "ScheduledDay")
        app_dt = _parse_timestamp(record.get("AppointmentDay"), "AppointmentDay")
        waiting_days = int((app_dt.floor("D") - sched_dt.floor("D")).days)
        if waiting_days < 0:
            raise ValidationError(
                f"Temporal anomaly: AppointmentDay ({app_dt:%Y-%m-%d}) cannot occur before "
                f"ScheduledDay ({sched_dt:%Y-%m-%d}). Waiting days cannot be negative ({waiting_days})."
            )

        return {
            "Age": age,
            "Gender": gender,
            "Scholarship": scholarship,
            "Neighbourhood": neighbourhood,
            "ScheduledDay": sched_dt,
            "AppointmentDay": app_dt,
            "waiting_days": waiting_days,
            "is_same_day": int(waiting_days == 0),
            "scheduled_hour": int(sched_dt.hour),
            "scheduled_day_of_week": sched_dt.day_name(),
            "appointment_day_of_week": app_dt.day_name(),
        }

    def _feature_dict(self, clean_rec: Dict[str, Any]) -> Dict[str, Any]:
        n = clean_rec["Neighbourhood"]
        return {
            "age": clean_rec["Age"],
            "waiting_days": clean_rec["waiting_days"],
            "log_waiting_days": float(np.log1p(clean_rec["waiting_days"])),
            "scheduled_hour": clean_rec["scheduled_hour"],
            "gender": clean_rec["Gender"],
            "neighbourhood_grouped": n if n in self.top_neighbourhoods else "Other",
            "scheduled_day_of_week": clean_rec["scheduled_day_of_week"],
            "appointment_day_of_week": clean_rec["appointment_day_of_week"],
            "scholarship": clean_rec["Scholarship"],
            "is_same_day": clean_rec["is_same_day"],
        }

    def prepare_feature_row(self, clean_rec: Dict[str, Any]) -> pd.DataFrame:
        """Single-row DataFrame with exactly the model's feature columns."""
        return pd.DataFrame([self._feature_dict(clean_rec)])[ALL_MODEL_FEATURES]

    # ------------------------------------------------------------------ banding
    def _clip_threshold(self, custom_threshold: Optional[float]) -> float:
        t = self.threshold if custom_threshold is None else float(custom_threshold)
        return max(0.01, min(0.99, t))

    def band_key(self, prob: float, threshold: Optional[float] = None) -> str:
        """LOWER below the review threshold, HIGHER at/above the priority cut-off, else MODERATE."""
        t = self._clip_threshold(threshold)
        higher = max(self.band_cutoffs["higher"], t)
        if prob >= higher:
            return "HIGHER"
        if prob >= t:
            return "MODERATE"
        return "LOWER"

    def _determine_risk_band(self, prob: float, threshold: Optional[float] = None) -> Dict[str, Any]:
        key = self.band_key(prob, threshold)
        b = RISK_BANDS[key]
        return {"key": key, "band_name": b["name"], "color": b["color"], "priority": b["priority"], "action": b["action"]}

    # -------------------------------------------------------------- explanation
    def explain(self, feature_df: pd.DataFrame, base_prob: float, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Model-based, per-appointment explanation. For each input we swap in the value a
        typical training appointment has (median / most common) and re-score; the change
        in probability is how much that input moved this estimate. Associations, not causes.
        """
        if not self.reference_profile:
            return []
        groups = {
            "waiting_days": ["waiting_days", "log_waiting_days", "is_same_day"],
            "age": ["age"],
            "scholarship": ["scholarship"],
            "gender": ["gender"],
            "neighbourhood_grouped": ["neighbourhood_grouped"],
            "scheduled_hour": ["scheduled_hour"],
            "scheduled_day_of_week": ["scheduled_day_of_week"],
            "appointment_day_of_week": ["appointment_day_of_week"],
        }
        variants, names = [], []
        row = feature_df.iloc[0].to_dict()
        for name, cols in groups.items():
            ref = dict(row)
            if name == "waiting_days":
                w = float(self.reference_profile.get("waiting_days", row["waiting_days"]))
                ref["waiting_days"] = w
                ref["log_waiting_days"] = float(np.log1p(w))
                ref["is_same_day"] = int(w == 0)
            else:
                for c in cols:
                    if c in self.reference_profile:
                        ref[c] = self.reference_profile[c]
            if all(ref[c] == row[c] for c in cols):
                continue
            variants.append(ref)
            names.append(name)
        if not variants:
            return []
        probs = self.pipeline.predict_proba(pd.DataFrame(variants)[ALL_MODEL_FEATURES])[:, 1]

        def _fmt(name: str, value: Any) -> str:
            if name == "waiting_days":
                return "same day" if int(value) == 0 else f"{int(value)} days"
            if name == "scholarship":
                return "enrolled" if int(value) == 1 else "not enrolled"
            if name == "gender":
                return {"F": "female", "M": "male"}.get(str(value), str(value))
            if name == "scheduled_hour":
                return f"{int(value):02d}:00"
            if name == "neighbourhood_grouped":
                return str(value).title()
            return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)

        out = []
        for name, p_ref in zip(names, probs, strict=True):
            delta = float(base_prob - p_ref)
            out.append({
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": _fmt(name, row[name]),
                "reference": _fmt(name, self.reference_profile.get(name, row[name])),
                "delta": round(delta, 4),
                "direction": "raises" if delta > 0 else "lowers",
            })
        out.sort(key=lambda d: abs(d["delta"]), reverse=True)
        return [d for d in out if abs(d["delta"]) >= 0.005][:top_k]

    # ----------------------------------------------------------------- predict
    def predict_single(self, record: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Probability, risk band, review flag and explanation for one booking record."""
        clean_rec = self.validate_record_dict(record)
        feature_df = self.prepare_feature_row(clean_rec)
        threshold = self._clip_threshold(custom_threshold)

        prob = float(self.pipeline.predict_proba(feature_df)[0, 1])
        is_flagged = bool(prob >= threshold)
        risk = self._determine_risk_band(prob, threshold)
        factors = self.explain(feature_df, prob)

        warnings = []
        unseen_day = feature_df["appointment_day_of_week"].iloc[0]
        if self.known_categories and unseen_day not in self.known_categories.get("appointment_day_of_week", [unseen_day]):
            warnings.append(
                f"No {unseen_day} appointments exist in the training data, so the model cannot judge that weekday. "
                "Treat this estimate with extra caution."
            )
            factors = [f for f in factors if f["feature"] != "appointment_day_of_week"]

        return {
            "estimated_probability": round(prob, 4),
            "probability_percent": round(prob * 100, 2),
            "threshold_used": round(threshold, 3),
            "review_flag": is_flagged,
            "status_label": "Flagged for reminder review" if is_flagged else "Standard protocol",
            "risk_key": risk["key"],
            "risk_band": risk["band_name"],
            "risk_color": risk["color"],
            "review_priority": risk["priority"],
            "operational_action": risk["action"],
            "waiting_days": clean_rec["waiting_days"],
            "is_same_day": bool(clean_rec["is_same_day"]),
            "neighbourhood_in_model": clean_rec["Neighbourhood"] in self.top_neighbourhoods,
            "factors": factors,
            "warnings": warnings,
            "governance_notice": (
                "MediPredict AI provides statistical risk estimates to aid staff prioritisation. "
                "This is not a certainty and must not be used to cancel appointments or deny care."
            ),
        }

    def predict_batch(self, df: pd.DataFrame, custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Scores a batch of raw booking records in one model call.
        Preserves row count, order and original columns; invalid rows get NaN plus a reason.
        """
        threshold = self._clip_threshold(custom_threshold)
        res_df = df.copy()
        n_rows = len(res_df)

        feats, ok_idx, errors = [], [], [None] * n_rows
        for i, rec in enumerate(res_df.to_dict(orient="records")):
            try:
                feats.append(self._feature_dict(self.validate_record_dict(rec)))
                ok_idx.append(i)
            except ValidationError as e:
                errors[i] = str(e)

        probs = np.full(n_rows, np.nan)
        waits = np.full(n_rows, np.nan)
        if feats:
            X = pd.DataFrame(feats)[ALL_MODEL_FEATURES]
            probs[ok_idx] = self.pipeline.predict_proba(X)[:, 1]
            waits[ok_idx] = X["waiting_days"].to_numpy()

        bands, priorities, flags = [], [], []
        for p in probs:
            if np.isnan(p):
                bands.append("Not scored")
                priorities.append("Validation error")
                flags.append("N/A")
            else:
                b = RISK_BANDS[self.band_key(float(p), threshold)]
                bands.append(b["name"])
                priorities.append(b["priority"])
                flags.append("Yes" if p >= threshold else "No")

        res_df["waiting_days"] = waits
        res_df["estimated_no_show_prob"] = np.round(probs, 4)
        res_df["risk_band"] = bands
        res_df["review_priority"] = priorities
        res_df["flagged_for_review"] = flags
        res_df["validation_error"] = errors

        assert len(res_df) == n_rows, "Batch inference must preserve exact row count!"
        return res_df


_inference_engine_instance: Optional[InferenceEngine] = None


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
    print(f"Prob: {result['estimated_probability']} | Band: {result['risk_band']} | Flagged: {result['review_flag']}")
    for f in result["factors"]:
        print(f"  {f['label']}: {f['value']} {f['direction']} risk by {abs(f['delta']) * 100:.1f} pts")
