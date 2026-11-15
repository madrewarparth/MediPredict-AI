"""MediPredict AI - Analytics: how attendance varies with lead time, timing and demographics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import THEME
from src.ui_components import (
    callout,
    load_metadata,
    page_header,
    require_data,
    section,
    show_chart,
    style_fig,
    synthetic_data_notice,
)

df = require_data()
overall = df["no_show"].mean() * 100

page_header(
    "Analytics",
    "How the no-show rate changes with lead time, booking timing and patient profile. "
    "Hover any bar for the number of appointments behind it.",
    eyebrow="Insights",
)
synthetic_data_notice(load_metadata())


def rate_table(keys: pd.Series) -> pd.DataFrame:
    g = df.groupby(keys, observed=False)["no_show"].agg(n="count", k="sum").reset_index()
    g.columns = ["key", "n", "k"]
    g = g[g["n"] > 0].copy()
    g["rate"] = g["k"] / g["n"] * 100
    # 95% Wilson interval, so small groups are not over-read
    z, p, n = 1.96, g["k"] / g["n"], g["n"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    g["lo"], g["hi"] = (centre - half) * 100, (centre + half) * 100
    return g


def rate_bar(g: pd.DataFrame, title: str, xtitle: str, color: str = THEME["primary_teal"]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=g["key"].astype(str), y=g["rate"], marker_color=color,
        error_y=dict(type="data", symmetric=False, array=g["hi"] - g["rate"], arrayminus=g["rate"] - g["lo"],
                     color="#94A3B8", thickness=1.2, width=3),
        customdata=np.stack([g["n"], g["lo"], g["hi"]], axis=1),
        hovertemplate="%{x}: %{y:.1f}% missed<br>95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f}%"
                      "<br>%{customdata[0]:,} appointments<extra></extra>",
    ))
    fig.add_hline(y=overall, line_dash="dot", line_color=THEME["warning"])
    style_fig(fig, f"{title}  <span style='font-size:11px;color:{THEME['warning']}'>· dotted = overall {overall:.1f}%</span>")
    fig.update_yaxes(title="No-show rate (%)", rangemode="tozero")
    fig.update_xaxes(title=xtitle)
    return fig


# ---------------------------------------------------------------- lead time
section("1. Lead time", "Days between booking and appointment. Only lead times with at least 100 appointments are plotted.")

curve = rate_table(df["waiting_days"].clip(upper=90))
curve = curve[curve["n"] >= 100]
same_day = df[df["waiting_days"] == 0]["no_show"].mean() * 100
week = df[(df["waiting_days"] >= 1) & (df["waiting_days"] <= 7)]["no_show"].mean() * 100
long_ = df[df["waiting_days"] > 14]["no_show"].mean() * 100
share_long = (df["waiting_days"] > 14).mean() * 100

c1, c2 = st.columns([7, 3], gap="medium")
with c1:
    fig = go.Figure()
    fig.add_scatter(x=curve["key"], y=curve["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=curve["key"], y=curve["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(15,118,110,0.12)", name="95% CI", hoverinfo="skip")
    fig.add_scatter(x=curve["key"], y=curve["rate"], mode="lines+markers", name="No-show rate",
                    line=dict(color=THEME["primary_teal"], width=2.5), marker=dict(size=5),
                    customdata=curve["n"], hovertemplate="%{x} days: %{y:.1f}% (%{customdata:,} appts)<extra></extra>")
    fig.add_hline(y=overall, line_dash="dot", line_color=THEME["warning"])
    style_fig(fig, f"No-show rate by lead time  <span style='font-size:11px;color:{THEME['warning']}'>· dotted = overall {overall:.1f}%</span>")
    fig.update_xaxes(title="Lead time (days)")
    fig.update_yaxes(title="No-show rate (%)", rangemode="tozero")
    show_chart(fig)
with c2:
    st.markdown(
        f"""
        <div class="mp-card" style="height:100%;">
          <h4>What the data shows</h4>
          <p><b>Same-day bookings:</b> {same_day:.1f}% missed.</p>
          <p style="margin-top:.5rem;"><b>1–7 days ahead:</b> {week:.1f}% missed.</p>
          <p style="margin-top:.5rem;"><b>More than 14 days ahead:</b> {long_:.1f}% missed. That's {share_long:.0f}% of all
          bookings, and the most useful group for a reminder call.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- timing
section("2. Timing")
c3, c4 = st.columns(2, gap="medium")
with c3:
    show_chart(rate_bar(rate_table(df["ScheduledDay"].dt.hour), "By booking hour", "Hour booked (UTC)", THEME["secondary_blue"]))
with c4:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = pd.Categorical(df["AppointmentDay"].dt.day_name(), categories=order, ordered=True)
    g = rate_table(pd.Series(dow, index=df.index))
    g["key"] = g["key"].astype(str).str[:3]
    show_chart(rate_bar(g, "By appointment weekday", "Weekday"))

# ---------------------------------------------------------------- profile
section("3. Patient profile", "Reported for transparency. The model is never used to treat groups differently in access to care.")
c5, c6 = st.columns(2, gap="medium")
with c5:
    age_grp = pd.cut(df["Age"], bins=[-1, 12, 24, 39, 59, 200],
                     labels=["0–12", "13–24", "25–39", "40–59", "60+"])
    show_chart(rate_bar(rate_table(age_grp), "By age group", "Age"))
with c6:
    g = rate_table(df["Scholarship"].map({0: "Not enrolled", 1: "Enrolled"}))
    show_chart(rate_bar(g, "By welfare enrolment (Bolsa Família)", "Enrolment", THEME["secondary_blue"]))

callout(
    "These are associations in historical data, not causes. Error bars are 95% Wilson intervals; "
    "overlapping bars usually mean the difference is not meaningful.",
    "info",
)
