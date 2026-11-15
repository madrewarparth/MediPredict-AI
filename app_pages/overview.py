"""MediPredict AI - Overview of historical appointment volume and attendance."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import THEME
from src.ui_components import (
    kpi_grid,
    load_metadata,
    page_header,
    require_data,
    show_chart,
    style_fig,
    synthetic_data_notice,
)

df = require_data()

page_header(
    "Overview",
    "Historical appointment volume and attendance. Filter by date and neighbourhood.",
    eyebrow="Insights",
)
synthetic_data_notice(load_metadata())

min_date = df["AppointmentDay"].dt.date.min()
max_date = df["AppointmentDay"].dt.date.max()
fc1, fc2 = st.columns([3, 2])
selected_dates = fc1.date_input(
    "Appointment dates", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="DD/MM/YYYY"
)
neighbourhoods = ["All neighbourhoods"] + sorted(df["Neighbourhood"].unique().tolist())
selected_n = fc2.selectbox("Neighbourhood", neighbourhoods, format_func=lambda x: x.title())

d = df
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    days = d["AppointmentDay"].dt.date
    d = d[(days >= start_d) & (days <= end_d)]
if selected_n != "All neighbourhoods":
    d = d[d["Neighbourhood"] == selected_n]

total = len(d)
if total == 0:
    st.warning("No appointments match these filters.")
    st.stop()

no_show = int(d["no_show"].sum())
rate = no_show / total * 100
same_day = int((d["waiting_days"] == 0).sum())

kpi_grid([
    {"label": "Appointments", "value": f"{total:,}", "sub": f"of {len(df):,} in the dataset"},
    {"label": "Attended", "value": f"{total - no_show:,}", "sub": f"{100 - rate:.1f}% attendance", "accent": THEME["attended"]},
    {"label": "Missed", "value": f"{no_show:,}", "sub": f"{rate:.1f}% no-show rate", "accent": THEME["no_show"]},
    {"label": "Median lead time", "value": f"{d['waiting_days'].median():.0f} days", "sub": f"mean {d['waiting_days'].mean():.1f} days"},
    {"label": "Same-day bookings", "value": f"{same_day / total * 100:.1f}%", "sub": f"{same_day:,} appointments"},
])

# --- daily volume
daily = (
    d.assign(day=d["AppointmentDay"].dt.date)
    .groupby("day")["no_show"].agg(total="count", missed="sum").reset_index()
)
daily["attended"] = daily["total"] - daily["missed"]
daily["rate"] = daily["missed"] / daily["total"] * 100

c1, c2 = st.columns([7, 3], gap="medium")
with c1:
    fig = go.Figure()
    fig.add_bar(x=daily["day"], y=daily["attended"], name="Attended", marker_color=THEME["attended"],
                hovertemplate="%{x|%d %b}: %{y:,} attended<extra></extra>")
    fig.add_bar(x=daily["day"], y=daily["missed"], name="Missed", marker_color=THEME["no_show"],
                customdata=daily["rate"], hovertemplate="%{x|%d %b}: %{y:,} missed (%{customdata:.1f}%)<extra></extra>")
    style_fig(fig, "Appointments per day")
    fig.update_layout(barmode="stack", bargap=0.15)
    show_chart(fig)
with c2:
    fig = go.Figure(go.Pie(
        labels=["Attended", "Missed"], values=[total - no_show, no_show], hole=0.68, sort=False,
        marker=dict(colors=[THEME["attended"], THEME["no_show"]], line=dict(color="#fff", width=2)),
        textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    style_fig(fig, "Attendance split")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", yanchor="top"),
                      annotations=[dict(text=f"<b>{rate:.1f}%</b><br>missed", showarrow=False, font=dict(size=15))])
    show_chart(fig)

# --- lead time & neighbourhoods
c3, c4 = st.columns(2, gap="medium")
with c3:
    buckets = pd.cut(
        d["waiting_days"], bins=[-1, 0, 3, 7, 14, 30, 10_000],
        labels=["Same day", "1–3 d", "4–7 d", "8–14 d", "15–30 d", "30+ d"],
    )
    ws = d.groupby(buckets, observed=False)["no_show"].agg(["count", "mean"]).reset_index()
    ws["rate"] = ws["mean"] * 100
    fig = go.Figure(go.Bar(
        x=ws["waiting_days"].astype(str), y=ws["rate"], marker_color=THEME["primary_teal"],
        text=[f"{r:.1f}%" if c else "" for r, c in zip(ws["rate"], ws["count"], strict=True)], textposition="outside",
        customdata=ws["count"], hovertemplate="%{x}: %{y:.1f}% missed (%{customdata:,} appts)<extra></extra>",
    ))
    style_fig(fig, "No-show rate by lead time")
    fig.update_yaxes(title="No-show rate (%)", rangemode="tozero")
    show_chart(fig)
with c4:
    top = (
        d.groupby("Neighbourhood")["no_show"].agg(total="count", rate="mean")
        .sort_values("total", ascending=False).head(10).reset_index()
    )
    top["rate"] *= 100
    top = top.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=top["Neighbourhood"].str.title(), x=top["total"], orientation="h", marker_color="#99D5CE",
        text=[f"{r:.1f}% missed" for r in top["rate"]], textposition="inside", insidetextanchor="end",
        textfont=dict(color=THEME["text_main"]),
        hovertemplate="%{y}: %{x:,} appointments<extra></extra>",
    ))
    style_fig(fig, "Busiest neighbourhoods")
    fig.update_xaxes(title="Appointments")
    show_chart(fig)
