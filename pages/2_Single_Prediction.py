"""
MediPredict AI - Single Appointment Inference
Interactive scheduling-time risk assessment and decision-support tool.
"""

from datetime import datetime, date, time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import THEME, RISK_BANDS
from src.ui_components import (
    inject_custom_css,
    render_sidebar_header,
    render_risk_badge,
    render_governance_disclaimer,
    get_plotly_theme_layout,
)
from src.predict import get_inference_engine, ValidationError

st.set_page_config(page_title="Single Prediction - MediPredict AI", page_icon="🔍", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("Single Appointment Risk Scoring")
st.caption("Point-in-time decision support for administrative staff at the moment an appointment is booked.")
render_governance_disclaimer()

# Load Engine
try:
    engine = get_inference_engine()
    meta = engine.metadata
    top_neighbourhoods = engine.top_neighbourhoods
    default_thresh = float(engine.threshold)
except Exception as e:
    st.error(f"Error loading inference engine: {e}")
    st.stop()

# Layout: Form on Left, Prediction Output on Right
col_form, col_result = st.columns([5, 5])

with col_form:
    st.markdown("### 📋 Appointment Details (Booking Moment)")
    st.info("💡 Only input information known at the time of appointment scheduling.")

    with st.form("single_pred_form"):
        # Section 1: Demographics
        st.markdown("##### 1. Patient Demographics")
        c_age, c_gen = st.columns(2)
        with c_age:
            age = st.number_input("Patient Age", min_value=0, max_value=115, value=35, step=1)
        with c_gen:
            gender = st.selectbox("Gender", options=["F", "M"], format_func=lambda x: "Female (F)" if x == "F" else "Male (M)")

        # Section 2: Temporal Scheduling
        st.markdown("##### 2. Scheduling Timestamps")
        c_sched_d, c_sched_t = st.columns(2)
        with c_sched_d:
            sched_date = st.date_input("Booking Date (ScheduledDay)", value=date.today())
        with c_sched_t:
            sched_time = st.time_input("Booking Time", value=time(9, 30))

        app_date = st.date_input("Appointment Date (AppointmentDay)", value=date.today())

        # Section 3: Operational & Catchment
        st.markdown("##### 3. Catchment & Socioeconomic Program")
        c_neigh, c_schol = st.columns(2)
        with c_neigh:
            neighbourhood = st.selectbox("Clinic Neighbourhood", options=top_neighbourhoods + ["Other"])
        with c_schol:
            scholarship = st.selectbox(
                "Social Welfare (Bolsa Família)",
                options=[0, 1],
                format_func=lambda x: "Enrolled (Yes)" if x == 1 else "Not Enrolled (No)",
            )

        # Operational threshold override
        with st.expander("⚙️ Decision Threshold Setting", expanded=False):
            custom_thresh = st.slider(
                "Review Decision Threshold",
                min_value=0.05,
                max_value=0.80,
                value=default_thresh,
                step=0.01,
                help="Threshold at which an appointment is flagged for priority administrative review.",
            )

        submit_btn = st.form_submit_button("Generate Risk Assessment", type="primary", use_container_width=True)

# Instant Lead Time Calculation
sched_combined = datetime.combine(sched_date, sched_time)
app_combined = datetime.combine(app_date, time(0, 0))
lead_time_days = (app_date - sched_date).days

with col_result:
    st.markdown("### 📊 Decision Support Assessment")

    if not submit_btn:
        st.markdown(
            f"""
            <div class="mp-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⏱️</div>
                <div style="font-weight: 600; color: {THEME['text_main']};">Awaiting Appointment Details</div>
                <div style="color: {THEME['text_muted']}; font-size: 0.85rem; margin-top: 0.25rem;">
                    Fill out the scheduling parameters and click <strong>Generate Risk Assessment</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Validate inputs
        if lead_time_days < 0:
            st.error(
                f"❌ Temporal Error: Appointment date ({app_date}) cannot be earlier than booking date ({sched_date}). "
                f"Negative lead time ({lead_time_days} days) is invalid."
            )
        else:
            record_payload = {
                "Age": age,
                "Gender": gender,
                "Scholarship": scholarship,
                "Neighbourhood": neighbourhood,
                "ScheduledDay": sched_combined.isoformat() + "Z",
                "AppointmentDay": app_combined.isoformat() + "Z",
            }

            try:
                res = engine.predict_single(record_payload, custom_threshold=custom_thresh)
                prob = res["estimated_probability"]
                prob_pct = res["probability_percent"]
                band = res["risk_band"]
                priority = res["review_priority"]
                color = res["risk_color"]
                flagged = res["review_flag"]

                # Result Card
                st.markdown(
                    f"""
                    <div class="mp-card" style="border-left: 5px solid {color};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="font-size: 0.85rem; color: {THEME['text_muted']}; text-transform: uppercase;">
                                Estimated No-Show Risk
                            </div>
                            <div>{render_risk_badge(band)}</div>
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                            <span style="font-size: 2.5rem; font-weight: 700; color: {THEME['text_main']};">
                                {prob_pct:.1f}%
                            </span>
                            <span style="color: {THEME['text_muted']}; font-size: 0.9rem;">probability of miss</span>
                        </div>
                        <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; background: {THEME['surface_elevated']}; border-radius: 6px;">
                            <strong style="color: {THEME['text_main']};">Operational Review Priority:</strong>
                            <span style="color: {color}; font-weight: 600;"> {priority}</span>
                            <div style="font-size: 0.8rem; color: {THEME['text_muted']}; margin-top: 0.2rem;">
                                {res['operational_action']}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Visual Gauge
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        domain={"x": [0, 1], "y": [0, 1]},
                        number={"suffix": "%", "font": {"color": THEME["text_main"], "size": 24}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": THEME["text_muted"], "tickwidth": 1},
                            "bar": {"color": color},
                            "bgcolor": THEME["surface_elevated"],
                            "borderwidth": 1,
                            "bordercolor": THEME["border"],
                            "steps": [
                                {"range": [0, 25], "color": "rgba(34, 197, 94, 0.15)"},
                                {"range": [25, 50], "color": "rgba(245, 158, 11, 0.15)"},
                                {"range": [50, 100], "color": "rgba(248, 113, 113, 0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": THEME["text_main"], "width": 3},
                                "thickness": 0.8,
                                "value": custom_thresh * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(l=25, r=25, t=25, b=25),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=THEME["text_main"]),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Contributing Factors Analysis
                st.markdown("#### Key Predictive Factors")
                if res["factors"]:
                    for f in res["factors"]:
                        st.markdown(
                            f"""
                            <div style="background: {THEME['card_bg']}; border-left: 3px solid {THEME['primary_teal']}; padding: 0.5rem 0.75rem; border-radius: 4px; margin-bottom: 0.4rem; font-size: 0.85rem;">
                                <strong>{f['factor']}</strong> — <em>{f['influence']}</em><br>
                                <span style="color: {THEME['text_muted']};">{f['detail']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Standard operational factors within normal baseline parameters.")

                # Operational Guidance box
                st.markdown(
                    f"""
                    <div class="mp-callout-info" style="margin-top: 1rem;">
                        <strong>Staff Action Protocol:</strong><br>
                        This appointment has an estimated lead time of <strong>{res['waiting_days']} days</strong>. 
                        {"Consider flagging for a reminder telephone call 48 hours prior to appointment." if flagged else "Routine automated SMS confirmation sufficient under standard protocol."}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as ex:
                st.error(f"Inference failure: {ex}")

