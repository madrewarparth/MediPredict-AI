"""
MediPredict AI - Overview Dashboard
Operational summary of appointment volumes, attendance distributions,
daily trends, and catchment statistics.
"""

import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Overview Dashboard - MediPredict AI", page_icon="📊", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("Operational Overview Dashboard")
st.caption("Comprehensive operational monitoring of scheduled hospital appointments, historical attendance, and clinic capacity.")
render_governance_disclaimer()


@st.cache_data
def load_data():
    """Loads and caches processed appointment dataset."""
    if not PROCESSED_DATA_FILE.exists():
        st.error("Processed dataset not found. Please run preprocessing pipeline.")
        return pd.DataFrame()
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    return df


df = load_data()

if df.empty:
    st.warning("No data available to display.")
    st.stop()

# Sidebar Filters
with st.sidebar:
    st.markdown("### Operational Filters")
    min_date = df["AppointmentDay"].dt.date.min()
    max_date = df["AppointmentDay"].dt.date.max()

    selected_dates = st.date_input(
        "Appointment Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_neighbourhoods = ["All Catchment Areas"] + sorted(df["Neighbourhood"].unique().tolist())
    selected_neighbourhood = st.selectbox("Clinic Neighbourhood", all_neighbourhoods)

# Apply Filters
df_filtered = df.copy()
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    df_filtered = df_filtered[
        (df_filtered["AppointmentDay"].dt.date >= start_d)
        & (df_filtered["AppointmentDay"].dt.date <= end_d)
    ]

if selected_neighbourhood != "All Catchment Areas":
    df_filtered = df_filtered[df_filtered["Neighbourhood"] == selected_neighbourhood]

total_count = len(df_filtered)
if total_count == 0:
    st.warning("No appointments found matching the selected filters.")
    st.stop()

no_show_count = int(df_filtered["no_show"].sum())
attended_count = total_count - no_show_count
no_show_rate = (no_show_count / total_count) * 100
avg_wait = df_filtered["waiting_days"].mean()
same_day_pct = (df_filtered["waiting_days"] == 0).mean() * 100

# Top KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Total Appointments", f"{total_count:,}", f"From {len(df):,} total")
with col2:
    render_metric_card("Attended Appointments", f"{attended_count:,}", f"{100 - no_show_rate:.1f}% attendance")
with col3:
    render_metric_card("Missed Appointments", f"{no_show_count:,}", f"{no_show_rate:.1f}% overall rate")
with col4:
    render_metric_card("Avg Waiting Time", f"{avg_wait:.1f} days", "From booking to date")
with col5:
    render_metric_card("Same-Day Bookings", f"{same_day_pct:.1f}%", f"{(df_filtered['waiting_days'] == 0).sum():,} records")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations Row 1: Daily Trend & Distribution
c1, c2 = st.columns([7, 3])

with c1:
    # Daily Volume Trend
    df_filtered["app_date_only"] = df_filtered["AppointmentDay"].dt.date
    daily_stats = (
        df_filtered.groupby(["app_date_only", "no_show"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if 0 not in daily_stats.columns:
        daily_stats[0] = 0
    if 1 not in daily_stats.columns:
        daily_stats[1] = 0

    daily_stats["Total"] = daily_stats[0] + daily_stats[1]
    daily_stats["NoShowRate"] = (daily_stats[1] / daily_stats["Total"]) * 100

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Bar(
            x=daily_stats["app_date_only"],
            y=daily_stats[0],
            name="Attended",
            marker_color=THEME["primary_teal"],
        )
    )
    fig_trend.add_trace(
        go.Bar(
            x=daily_stats["app_date_only"],
            y=daily_stats[1],
            name="No-Show",
            marker_color=THEME["error"],
        )
    )
    fig_trend.update_layout(
        barmode="stack",
        title="Appointment Volume & Attendance by Date",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    # Attendance Breakdown Donut
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Attended", "No-Show (Missed)"],
                values=[attended_count, no_show_count],
                hole=0.6,
                marker=dict(colors=[THEME["primary_teal"], THEME["error"]]),
                textinfo="percent+label",
            )
        ]
    )
    fig_donut.update_layout(
        title="Attendance Proportion",
        showlegend=False,
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Visualizations Row 2: Lead Time Distribution & Top Neighbourhoods
c3, c4 = st.columns(2)

with c3:
    # Waiting days binning
    df_filtered["wait_bucket"] = pd.cut(
        df_filtered["waiting_days"],
        bins=[-1, 0, 3, 7, 14, 30, 180],
        labels=["Same-day (0d)", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"],
    )
    wait_summary = (
        df_filtered.groupby("wait_bucket", observed=False)["no_show"]
        .agg(["count", "mean"])
        .reset_index()
    )
    wait_summary["no_show_rate"] = wait_summary["mean"] * 100

    fig_wait = px.bar(
        wait_summary,
        x="wait_bucket",
        y="no_show_rate",
        text=wait_summary["no_show_rate"].apply(lambda x: f"{x:.1f}%"),
        title="No-Show Rate by Lead Time (Waiting Days)",
        color="no_show_rate",
        color_continuous_scale=[[0, THEME["primary_teal"]], [1, THEME["error"]]],
    )
    fig_wait.update_layout(
        coloraxis_showscale=False,
        yaxis_title="No-Show Rate (%)",
        xaxis_title="Booking Lead Time",
        **get_plotly_theme_layout(),
    )
    st.plotly_chart(fig_wait, use_container_width=True)

with c4:
    # Top 10 Neighbourhoods by Volume
    top_neighbourhoods = (
        df_filtered.groupby("Neighbourhood")
        .agg(Total=("no_show", "count"), NoShowRate=("no_show", lambda x: x.mean() * 100))
        .sort_values(by="Total", ascending=False)
        .head(10)
        .reset_index()
    )

    fig_neigh = go.Figure()
    fig_neigh.add_trace(
        go.Bar(
            x=top_neighbourhoods["Neighbourhood"],
            y=top_neighbourhoods["Total"],
            name="Volume",
            marker_color=THEME["surface_elevated"],
            yaxis="y",
        )
    )
    fig_neigh.add_trace(
        go.Scatter(
            x=top_neighbourhoods["Neighbourhood"],
            y=top_neighbourhoods["NoShowRate"],
            name="No-Show %",
            mode="lines+markers",
            marker_color=THEME["warning"],
            yaxis="y2",
        )
    )
    fig_neigh.update_layout(
        title="Top 10 Clinic Catchment Areas (Volume vs Missed Rate)",
        yaxis=dict(title="Volume", gridcolor=THEME["border"]),
        yaxis2=dict(
            title="No-Show Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 40],
            showgrid=False,
        ),
        xaxis=dict(tickangle=-30),
        **get_plotly_theme_layout(exclude=["xaxis", "yaxis"]),
    )
    st.plotly_chart(fig_neigh, use_container_width=True)

