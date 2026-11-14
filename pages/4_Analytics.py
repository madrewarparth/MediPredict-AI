"""
MediPredict AI - Operational Analytics & Behavioral Trends
In-depth exploration of operational factors, lead times, and scheduling patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import THEME, PROCESSED_DATA_FILE
from src.ui_components import (
    inject_custom_css,
    render_sidebar_header,
    render_metric_card,
    get_plotly_theme_layout,
    render_governance_disclaimer,
)

st.set_page_config(page_title="Analytics - MediPredict AI", page_icon="📈", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("Operational Analytics & Patterns")
st.caption("Empirical operational insights to inform clinic scheduling rules and reminder strategies.")
render_governance_disclaimer()


@st.cache_data
def load_data():
    if not PROCESSED_DATA_FILE.exists():
        st.error("Processed dataset not found.")
        return pd.DataFrame()
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    return df


df = load_data()
if df.empty:
    st.stop()

# Row 1: Waiting Days Decay Curve
st.markdown("### 1. Lead Time Impact on Appointment Attendance")
c1, c2 = st.columns([7, 3])

with c1:
    # Lead time curve
    wait_curve = (
        df[df["waiting_days"] <= 45]
        .groupby("waiting_days")["no_show"]
        .agg(["count", "mean"])
        .reset_index()
    )
    wait_curve["no_show_rate"] = wait_curve["mean"] * 100

    fig_decay = go.Figure()
    fig_decay.add_trace(
        go.Scatter(
            x=wait_curve["waiting_days"],
            y=wait_curve["no_show_rate"],
            mode="lines+markers",
            name="No-Show Rate",
            line=dict(color=THEME["primary_teal"], width=3),
            marker=dict(size=6, color=THEME["primary_teal"]),
        )
    )
    # Add benchmark reference line
    overall_rate = df["no_show"].mean() * 100
    fig_decay.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color=THEME["warning"],
        annotation_text=f"Baseline Average: {overall_rate:.1f}%",
        annotation_position="bottom right",
    )
    fig_decay.update_layout(
        title="Missed Rate vs Days Between Booking and Appointment",
        xaxis_title="Lead Time (Waiting Days)",
        yaxis_title="No-Show Rate (%)",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_decay, use_container_width=True)

with c2:
    st.markdown(
        f"""
        <div class="mp-card" style="margin-top: 1.5rem;">
            <div style="font-size: 1rem; font-weight: 600; color: {THEME['primary_teal']}; margin-bottom: 0.5rem;">
                Key Operational Finding
            </div>
            <p style="color: {THEME['text_muted']}; font-size: 0.85rem; line-height: 1.5;">
                <strong>Same-Day Bookings:</strong> Under 5% no-show rate. Patients booking for today almost always attend.<br><br>
                <strong>The 14-Day Cliff:</strong> Appointments scheduled more than 14 days in advance exhibit elevated no-show rates (>30%), representing prime targets for 48-hour reminder calls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Scheduling Hour & Day of Week Heatmap
st.markdown("### 2. Temporal Scheduling Behaviors")
c3, c4 = st.columns(2)

with c3:
    # Booking Hour
    df["sched_hour"] = df["ScheduledDay"].dt.hour
    hour_stats = (
        df.groupby("sched_hour")["no_show"]
        .agg(["count", "mean"])
        .reset_index()
    )
    hour_stats["no_show_rate"] = hour_stats["mean"] * 100

    fig_hour = px.bar(
        hour_stats,
        x="sched_hour",
        y="count",
        color="no_show_rate",
        color_continuous_scale=[[0, THEME["primary_teal"]], [1, THEME["error"]]],
        labels={"sched_hour": "Hour of Day (24h)", "count": "Bookings Created", "no_show_rate": "No-Show %"},
        title="Appointment Booking Volume & Missed Rate by Hour",
    )
    fig_hour.update_layout(coloraxis_colorbar_title="Missed %", **get_plotly_theme_layout())
    st.plotly_chart(fig_hour, use_container_width=True)

with c4:
    # Day of Week Attendance
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    df["app_dow"] = df["AppointmentDay"].dt.day_name()
    dow_stats = (
        df.groupby("app_dow", observed=False)["no_show"]
        .agg(["count", "mean"])
        .reindex(day_order)
        .dropna()
        .reset_index()
    )
    dow_stats["no_show_rate"] = dow_stats["mean"] * 100

    fig_dow = px.bar(
        dow_stats,
        x="app_dow",
        y="no_show_rate",
        text=dow_stats["no_show_rate"].apply(lambda x: f"{x:.1f}%"),
        title="No-Show Rate by Appointment Day of Week",
        color="no_show_rate",
        color_continuous_scale=[[0, THEME["primary_teal"]], [1, THEME["error"]]],
    )
    fig_dow.update_layout(
        coloraxis_showscale=False,
        yaxis_title="No-Show Rate (%)",
        xaxis_title="Day of Week",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# Row 3: Demographics & Socioeconomic Factors
st.markdown("### 3. Demographic & Socioeconomic Profiles")
c5, c6 = st.columns(2)

with c5:
    # Age Bins
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 24, 39, 59, 120],
        labels=["Child (0-12)", "Youth (13-24)", "Adult (25-39)", "Middle Age (40-59)", "Senior (60+)"],
    )
    age_stats = (
        df.groupby("age_group", observed=False)["no_show"]
        .agg(["count", "mean"])
        .reset_index()
    )
    age_stats["no_show_rate"] = age_stats["mean"] * 100

    fig_age = px.bar(
        age_stats,
        x="age_group",
        y="no_show_rate",
        text=age_stats["no_show_rate"].apply(lambda x: f"{x:.1f}%"),
        title="No-Show Rate Across Patient Age Cohorts",
        color="no_show_rate",
        color_continuous_scale=[[0, THEME["primary_teal"]], [1, THEME["error"]]],
    )
    fig_age.update_layout(
        coloraxis_showscale=False,
        yaxis_title="No-Show Rate (%)",
        xaxis_title="Age Demographic Group",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_age, use_container_width=True)

with c6:
    # Welfare Scholarship
    schol_stats = (
        df.groupby("Scholarship")["no_show"]
        .agg(["count", "mean"])
        .reset_index()
    )
    schol_stats["Label"] = schol_stats["Scholarship"].map({0: "Non-Enrolled", 1: "Social Welfare (Bolsa Família)"})
    schol_stats["no_show_rate"] = schol_stats["mean"] * 100

    fig_schol = px.bar(
        schol_stats,
        x="Label",
        y="no_show_rate",
        text=schol_stats["no_show_rate"].apply(lambda x: f"{x:.1f}%"),
        title="Attendance by Welfare Assistance Enrollment",
        color="no_show_rate",
        color_continuous_scale=[[0, THEME["primary_teal"]], [1, THEME["error"]]],
    )
    fig_schol.update_layout(
        coloraxis_showscale=False,
        yaxis_title="No-Show Rate (%)",
        xaxis_title="Enrollment Status",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_schol, use_container_width=True)

