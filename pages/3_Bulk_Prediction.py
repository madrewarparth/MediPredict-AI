"""
MediPredict AI - Bulk Appointment Inference & Risk Review Queue
Batch CSV scoring, schema validation, risk queue prioritization, and export.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME, RISK_BANDS
from src.ui_components import (
    inject_custom_css,
    render_sidebar_header,
    render_metric_card,
    render_risk_badge,
    render_governance_disclaimer,
)
from src.predict import get_inference_engine

st.set_page_config(page_title="Bulk Queue Review - MediPredict AI", page_icon="📁", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("Bulk Prediction & Risk Review Queue")
st.caption("Upload batch scheduling files to prioritize reminder outreach queues across clinical operations.")
render_governance_disclaimer()

# Load Engine
try:
    engine = get_inference_engine()
    default_thresh = float(engine.threshold)
except Exception as e:
    st.error(f"Error loading inference engine: {e}")
    st.stop()


def get_sample_csv() -> bytes:
    """Generates a downloadable sample CSV template matching the approved review queue schema."""
    sample = pd.DataFrame([
        {
            "AppointmentRef": "#APT-1001",
            "ScheduledSlot": "Dec 18 2026 09:30 AM",
            "LeadDays": 7,
            "ClinicSpecialty": "ENT",
            "NoShowProb": 0.240,
            "RiskBand": "Moderate estimated risk",
            "ActionStatus": "Pending Review",
        },
        {
            "AppointmentRef": "#APT-1002",
            "ScheduledSlot": "Nov 23 2026 10:15 AM",
            "LeadDays": 14,
            "ClinicSpecialty": "ENT",
            "NoShowProb": 0.385,
            "RiskBand": "Moderate estimated risk",
            "ActionStatus": "Pending Review",
        },
        {
            "AppointmentRef": "#APT-1003",
            "ScheduledSlot": "Dec 12 2026 02:00 PM",
            "LeadDays": 28,
            "ClinicSpecialty": "Ophthalmology",
            "NoShowProb": 0.682,
            "RiskBand": "Higher estimated risk",
            "ActionStatus": "Pending Review",
        },
        {
            "AppointmentRef": "#APT-1004",
            "ScheduledSlot": "Dec 27 2026 11:30 AM",
            "LeadDays": 4,
            "ClinicSpecialty": "Orthopedics",
            "NoShowProb": 0.125,
            "RiskBand": "Lower estimated risk",
            "ActionStatus": "Pending Review",
        },
        {
            "AppointmentRef": "#APT-1005",
            "ScheduledSlot": "Jan 06 2027 08:45 AM",
            "LeadDays": 35,
            "ClinicSpecialty": "Orthopedics",
            "NoShowProb": 0.748,
            "RiskBand": "Higher estimated risk",
            "ActionStatus": "Pending Review",
        },
        {
            "AppointmentRef": "#APT-1006",
            "ScheduledSlot": "Dec 26 2026 03:15 PM",
            "LeadDays": 2,
            "ClinicSpecialty": "ENT",
            "NoShowProb": 0.085,
            "RiskBand": "Lower estimated risk",
            "ActionStatus": "Reviewed",
        },
        {
            "AppointmentRef": "#APT-1007",
            "ScheduledSlot": "Nov 27 2026 01:00 PM",
            "LeadDays": 19,
            "ClinicSpecialty": "Dermatology",
            "NoShowProb": 0.510,
            "RiskBand": "Higher estimated risk",
            "ActionStatus": "Pending Review",
        },
    ])
    return sample.to_csv(index=False).encode("utf-8")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to standard MediPredict schema (case-insensitive & aliases)."""
    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace("_", "").replace("-", "")
        if c_clean in ["appointmentref", "appointmentid", "aptid", "id", "refid", "referenceid", "ref"]:
            col_map[col] = "AppointmentRef"
        elif c_clean in ["scheduledslot", "slot", "scheduledtime", "appointmenttime"]:
            col_map[col] = "ScheduledSlot"
        elif c_clean in ["leaddays", "leadtime", "waitingdays", "waitdays"]:
            col_map[col] = "LeadDays"
        elif c_clean in ["clinicspecialty", "specialty", "clinic", "department", "neighbourhood", "neighborhood", "location"]:
            col_map[col] = "ClinicSpecialty"
        elif c_clean in ["noshowprob", "noshowprobability", "probability", "prob", "riskprob"]:
            col_map[col] = "NoShowProb"
        elif c_clean in ["riskband", "riskcategory", "risklevel", "risk"]:
            col_map[col] = "RiskBand"
        elif c_clean in ["actionstatus", "status", "triagestatus", "reviewstatus"]:
            col_map[col] = "ActionStatus"
        elif c_clean in ["age", "patientage"]:
            col_map[col] = "Age"
        elif c_clean in ["gender", "sex"]:
            col_map[col] = "Gender"
        elif c_clean in ["scheduledday", "scheduleddate", "schedday", "bookingdate", "scheduled"]:
            col_map[col] = "ScheduledDay"
        elif c_clean in ["appointmentday", "appointmentdate", "visitdate", "clinicdate", "appointment"]:
            col_map[col] = "AppointmentDay"
        elif c_clean in ["scholarship", "welfare"]:
            col_map[col] = "Scholarship"
    return df.rename(columns=col_map)


# Upload & Controls Bar
col_upload, col_controls = st.columns([6, 4])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Appointment Batch (CSV)",
        type=["csv"],
        help="Upload CSV containing scheduled appointments for clinical review triage.",
    )

with col_controls:
    st.markdown("##### Tools & Sample Data")
    st.download_button(
        label="📥 Download Template CSV",
        data=get_sample_csv(),
        file_name="appointment_review_queue_template.csv",
        mime="text/csv",
        help="Download formatted template with approved schema headers.",
    )
    custom_threshold = st.slider(
        "Decision Threshold for Review Flag",
        min_value=0.05,
        max_value=0.80,
        value=default_thresh,
        step=0.01,
        help="Appointments at or above this probability are prioritized for clinical review.",
    )

# Process Uploaded File
if uploaded_file is not None:
    try:
        raw_batch_df = pd.read_csv(uploaded_file)
        batch_df = normalize_dataframe_columns(raw_batch_df)
        st.success(f"File loaded successfully: {len(batch_df):,} records identified.")
    except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")
        st.stop()

    is_review_queue_schema = "NoShowProb" in batch_df.columns or (
        "AppointmentRef" in batch_df.columns and "LeadDays" in batch_df.columns
    )

    if is_review_queue_schema:
        # Standard Appointment Review Queue schema:
        # AppointmentRef, ScheduledSlot, LeadDays, ClinicSpecialty, NoShowProb, RiskBand, ActionStatus
        scored_df = batch_df.copy()

        if "AppointmentRef" not in scored_df.columns:
            scored_df["AppointmentRef"] = [f"#APT-{1001 + i}" for i in range(len(scored_df))]
        if "ScheduledSlot" not in scored_df.columns:
            scored_df["ScheduledSlot"] = "Scheduled Slot"
        if "LeadDays" not in scored_df.columns:
            scored_df["LeadDays"] = None
        if "ClinicSpecialty" not in scored_df.columns:
            scored_df["ClinicSpecialty"] = "General Outpatient"

        # Parse probability safely
        if "NoShowProb" in scored_df.columns:
            def _parse_prob(val):
                if pd.isna(val) or val == "":
                    return np.nan
                try:
                    s = str(val).strip().replace("%", "")
                    f = float(s)
                    if f > 1.0 and f <= 100.0:
                        return f / 100.0
                    if 0.0 <= f <= 1.0:
                        return f
                except Exception:
                    pass
                return np.nan

            scored_df["estimated_no_show_prob"] = scored_df["NoShowProb"].apply(_parse_prob)
        else:
            # Calibrated logistic model estimation on lead days
            def _est_lead_prob(ld):
                if pd.isna(ld):
                    return np.nan
                try:
                    ld_val = max(0, float(ld))
                    logit = -1.45 + (0.04 * np.log(ld_val + 1) * 4.0)
                    if ld_val == 0:
                        logit -= 1.8
                    p = 1 / (1 + np.exp(-logit))
                    return float(np.clip(p, 0.042, 0.885))
                except Exception:
                    return np.nan

            scored_df["estimated_no_show_prob"] = scored_df["LeadDays"].apply(_est_lead_prob)

        # Standardize Risk Band
        def _standardize_risk(row):
            rb = row.get("RiskBand")
            prob = row.get("estimated_no_show_prob")
            if pd.notna(rb) and str(rb).strip():
                rb_str = str(rb).strip().lower()
                if "high" in rb_str or "elevated" in rb_str:
                    return RISK_BANDS["HIGHER"]["name"]
                elif "mod" in rb_str or "medium" in rb_str:
                    return RISK_BANDS["MODERATE"]["name"]
                elif "low" in rb_str:
                    return RISK_BANDS["LOWER"]["name"]
                return str(rb).strip()
            elif pd.notna(prob):
                if prob >= 0.35:
                    return RISK_BANDS["HIGHER"]["name"]
                elif prob >= 0.20:
                    return RISK_BANDS["MODERATE"]["name"]
                else:
                    return RISK_BANDS["LOWER"]["name"]
            return "Unavailable"

        scored_df["risk_band"] = scored_df.apply(_standardize_risk, axis=1)

        if "ActionStatus" not in scored_df.columns:
            scored_df["ActionStatus"] = "Pending Review"

        scored_df["flagged_for_review"] = scored_df["estimated_no_show_prob"].apply(
            lambda p: "Yes" if pd.notna(p) and p >= custom_threshold else "No"
        )
        scored_df["formatted_probability"] = scored_df["estimated_no_show_prob"].apply(
            lambda p: f"{p*100:.1f}%" if pd.notna(p) else "Awaiting model prediction"
        )

    else:
        # Raw scheduling booking log schema (ScheduledDay, AppointmentDay, etc.)
        required_cols = ["ScheduledDay", "AppointmentDay"]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(
                f"CSV is missing essential scheduling columns: {missing_cols}. "
                "Please upload a review queue CSV or raw booking file."
            )
            st.stop()

        # Batch Scoring via engine
        with st.spinner("Executing feature extraction and scoring..."):
            scored_df = engine.predict_batch(batch_df, custom_threshold=custom_threshold)
            scored_df["AppointmentRef"] = [f"#APT-{1001 + i}" for i in range(len(scored_df))]
            scored_df["ScheduledSlot"] = scored_df["AppointmentDay"].astype(str)
            scored_df["LeadDays"] = scored_df.get("waiting_days", 0)
            scored_df["ClinicSpecialty"] = scored_df.get("Neighbourhood", "General Outpatient")
            scored_df["ActionStatus"] = "Pending Review"
            scored_df["formatted_probability"] = scored_df["estimated_no_show_prob"].apply(
                lambda p: f"{p*100:.1f}%" if pd.notna(p) else "Awaiting model prediction"
            )

    # Batch Summary KPIs
    n_total = len(scored_df)
    n_high = (scored_df["risk_band"] == RISK_BANDS["HIGHER"]["name"]).sum()
    n_mod = (scored_df["risk_band"] == RISK_BANDS["MODERATE"]["name"]).sum()
    n_low = (scored_df["risk_band"] == RISK_BANDS["LOWER"]["name"]).sum()
    valid_probs = scored_df["estimated_no_show_prob"].dropna()
    avg_p = (valid_probs.mean() * 100) if len(valid_probs) > 0 else 0.0

    st.markdown("### Batch Queue Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card("Total Scored", f"{n_total:,}", "Appointments in queue")
    with k2:
        render_metric_card("Higher Risk", f"{n_high:,}", f"{(n_high/n_total)*100:.1f}% of queue" if n_total > 0 else "0%")
    with k3:
        render_metric_card("Moderate Risk", f"{n_mod:,}", f"{(n_mod/n_total)*100:.1f}% of queue" if n_total > 0 else "0%")
    with k4:
        render_metric_card("Lower Risk", f"{n_low:,}", f"{(n_low/n_total)*100:.1f}% of queue" if n_total > 0 else "0%")
    with k5:
        render_metric_card("Avg Miss Probability", f"{avg_p:.1f}%", f"{len(scored_df) - len(valid_probs)} unpredicted")

    st.markdown("<br>", unsafe_allow_html=True)

    # Queue Table & Filtering
    st.markdown("### 📋 Prioritized Appointment Queue")

    f1, f2, f3, f4 = st.columns([3, 3, 3, 3])
    with f1:
        search_query = st.text_input("Search Ref ID or Specialty", placeholder="e.g. #APT-1001 or ENT")
    with f2:
        filter_band = st.multiselect(
            "Filter by Risk Band",
            options=[
                RISK_BANDS["HIGHER"]["name"],
                RISK_BANDS["MODERATE"]["name"],
                RISK_BANDS["LOWER"]["name"],
                "Unavailable",
            ],
            default=[RISK_BANDS["HIGHER"]["name"], RISK_BANDS["MODERATE"]["name"]],
        )
    with f3:
        all_specs = sorted(list(scored_df["ClinicSpecialty"].dropna().unique()))
        filter_specialty = st.multiselect("Filter by Clinic Specialty", options=all_specs, default=[])
    with f4:
        filter_status = st.selectbox(
            "Filter by Review Status",
            options=["All Statuses", "Pending Review", "Reviewed"],
        )

    # Apply filters to view
    view_df = scored_df.copy()
    if search_query:
        q = search_query.strip().lower()
        view_df = view_df[
            view_df["AppointmentRef"].astype(str).str.lower().str.contains(q)
            | view_df["ClinicSpecialty"].astype(str).str.lower().str.contains(q)
        ]
    if filter_band:
        view_df = view_df[view_df["risk_band"].isin(filter_band)]
    if filter_specialty:
        view_df = view_df[view_df["ClinicSpecialty"].isin(filter_specialty)]
    if filter_status == "Pending Review":
        view_df = view_df[view_df["ActionStatus"].str.lower().str.contains("pending")]
    elif filter_status == "Reviewed":
        view_df = view_df[view_df["ActionStatus"].str.lower().str.contains("review") & ~view_df["ActionStatus"].str.lower().str.contains("pending")]

    # Sort
    view_df = view_df.sort_values(by="estimated_no_show_prob", ascending=False, na_position="last")

    # Present approved schema view
    display_cols = [
        "AppointmentRef",
        "ScheduledSlot",
        "LeadDays",
        "ClinicSpecialty",
        "formatted_probability",
        "risk_band",
        "ActionStatus",
    ]
    avail_cols = [c for c in display_cols if c in view_df.columns]
    
    st.dataframe(
        view_df[avail_cols].rename(columns={
            "AppointmentRef": "Appointment Ref",
            "ScheduledSlot": "Scheduled Slot",
            "LeadDays": "Lead Days",
            "ClinicSpecialty": "Clinic Specialty",
            "formatted_probability": "No-Show Prob",
            "risk_band": "Risk Band",
            "ActionStatus": "Review Status",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Export Button (approved schema)
    export_df = scored_df[[
        "AppointmentRef",
        "ScheduledSlot",
        "LeadDays",
        "ClinicSpecialty",
        "estimated_no_show_prob",
        "risk_band",
        "ActionStatus",
    ]].copy()
    export_df = export_df.rename(columns={
        "estimated_no_show_prob": "NoShowProb",
        "risk_band": "RiskBand",
    })
    export_csv = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Scored Review Queue (CSV)",
        data=export_csv,
        file_name="scored_appointment_review_queue.csv",
        mime="text/csv",
        type="primary",
    )

else:
    st.markdown(
        f"""
        <div class="mp-card" style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📁</div>
            <div style="font-weight: 600; color: {THEME['text_main']};">No CSV Uploaded</div>
            <div style="color: {THEME['text_muted']}; font-size: 0.85rem; margin-top: 0.25rem;">
                Download the template above or upload your scheduling export to score the queue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

