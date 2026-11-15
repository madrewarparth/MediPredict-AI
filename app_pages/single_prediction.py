"""MediPredict AI - Score a single appointment at booking time."""

from datetime import date, datetime, time, timedelta

import streamlit as st

from src.config import THEME
from src.predict import ValidationError
from src.ui_components import (
    esc,
    page_header,
    render_governance_disclaimer,
    render_risk_badge,
    require_engine,
    section,
)

engine = require_engine()
default_thresh = round(float(engine.threshold), 2)
cut_higher = float(engine.band_cutoffs["higher"])

page_header(
    "Score an appointment",
    "Enter only what is known at the moment of booking. The estimate updates when you press <b>Estimate risk</b>.",
    eyebrow="Operations",
)
render_governance_disclaimer()

col_form, col_result = st.columns([5, 6], gap="large")

with col_form:
    with st.form("single_pred_form", border=True):
        st.markdown("**Patient**")
        a, g = st.columns(2)
        age = a.number_input("Age", min_value=0, max_value=115, value=35, step=1)
        gender = g.selectbox("Gender", ["F", "M"], format_func=lambda x: {"F": "Female", "M": "Male"}[x])

        n, s = st.columns(2)
        neighbourhood = n.selectbox(
            "Neighbourhood",
            engine.top_neighbourhoods + ["Other"],
            format_func=lambda x: x.title(),
            help="The model knows the 25 busiest neighbourhoods; anything else is grouped as Other.",
        )
        scholarship = s.selectbox(
            "Welfare (Bolsa Família)", [0, 1], format_func=lambda x: "Enrolled" if x else "Not enrolled"
        )

        st.markdown("**Booking**")
        d1, d2 = st.columns(2)
        sched_date = d1.date_input("Booked on", value=date.today(), format="DD/MM/YYYY")
        sched_time = d2.time_input("Booking time", value=time(9, 30), step=timedelta(minutes=15))
        default_app = date.today() + timedelta(days=7)
        if default_app.weekday() == 6:  # clinics in the training data never open on Sunday
            default_app += timedelta(days=1)
        app_date = st.date_input(
            "Appointment date", value=default_app, format="DD/MM/YYYY",
            help="Must be on or after the booking date.",
        )

        with st.expander("Review threshold", expanded=False):
            custom_thresh = st.slider(
                "Flag for review at or above",
                min_value=0.05, max_value=0.80, value=default_thresh, step=0.01,
                help=f"Default {default_thresh:.2f} is the F1-optimal threshold from validation.",
            )

        submitted = st.form_submit_button("Estimate risk", type="primary", width="stretch", icon=":material/bolt:")

lead_days = (app_date - sched_date).days

with col_result:
    if not submitted:
        st.markdown(
            f"""
            <div class="mp-card" style="text-align:center;padding:2.6rem 1.2rem;">
              <div style="font-size:2rem;line-height:1;">🗓️</div>
              <h4 style="margin-top:.7rem;">No estimate yet</h4>
              <p>Fill in the booking and press <b>Estimate risk</b>. Current lead time:
              <b>{lead_days} day{'s' if lead_days != 1 else ''}</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if lead_days < 0:
        st.error(f"The appointment date ({app_date:%d %b %Y}) is before the booking date ({sched_date:%d %b %Y}).")
        st.stop()

    record = {
        "Age": age,
        "Gender": gender,
        "Scholarship": scholarship,
        "Neighbourhood": neighbourhood,
        "ScheduledDay": datetime.combine(sched_date, sched_time).isoformat() + "Z",
        "AppointmentDay": datetime.combine(app_date, time(0, 0)).isoformat() + "Z",
    }
    try:
        res = engine.predict_single(record, custom_threshold=custom_thresh)
    except ValidationError as ex:
        st.error(str(ex))
        st.stop()

    pct = res["probability_percent"]
    color = res["risk_color"]
    t_pct = res["threshold_used"] * 100
    h_pct = max(cut_higher, res["threshold_used"]) * 100
    flag_html = (
        '<span class="mp-badge mp-badge-higher"><i style="background:#DC2626"></i>Flagged for review</span>'
        if res["review_flag"] else
        '<span class="mp-badge mp-badge-lower"><i style="background:#16A34A"></i>Not flagged</span>'
    )
    st.markdown(
        f"""
        <div class="mp-result" style="--mp-accent:{color}">
          <div class="mp-result-top">
            <span class="mp-kpi-label">Estimated no-show probability</span>
            <span style="display:flex;gap:.4rem;flex-wrap:wrap;">{render_risk_badge(res['risk_band'])}{flag_html}</span>
          </div>
          <div class="mp-result-prob" style="color:{color}">{pct:.1f}%</div>
          <div style="color:{THEME['text_muted']};font-size:.88rem;">
            Roughly {round(pct)} in 100 similar bookings are missed · lead time {res['waiting_days']} day{'s' if res['waiting_days'] != 1 else ''}
          </div>
          <div class="mp-meter" style="--mp-t:{t_pct:.1f}%;--mp-h:{h_pct:.1f}%;--mp-accent:{color}">
            <div class="mp-meter-fill" style="width:{min(pct, 100):.1f}%"></div>
            <div class="mp-meter-tick" style="left:calc({t_pct:.1f}% - 1px)" title="Review threshold"></div>
          </div>
          <div class="mp-meter-legend"><span>0%</span><span>review at {t_pct:.0f}% · priority at {h_pct:.0f}%</span><span>100%</span></div>
          <div class="mp-callout mp-callout-info" style="margin:1rem 0 0 0;">
            <strong>{esc(res['review_priority'])}.</strong> {esc(res['operational_action'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for w in res.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    if not res["neighbourhood_in_model"]:
        st.caption("This neighbourhood is outside the model's top 25 and is treated as “Other”.")

    section(
        "What moved this estimate",
        "Each input is swapped for a typical booking's value and re-scored. Points show how far the estimate moved. "
        "These are statistical associations, not causes.",
    )
    if res["factors"]:
        rows = []
        for f in res["factors"]:
            up = f["delta"] > 0
            rows.append(
                f'<div class="mp-factor"><span class="mp-factor-name">{esc(f["label"])}: {esc(f["value"])}</span>'
                f'<span class="mp-factor-delta {"mp-up" if up else "mp-down"}">{"+" if up else "−"}{abs(f["delta"]) * 100:.1f} pts</span>'
                f'<span class="mp-factor-val">{"raises" if up else "lowers"} risk vs. a typical booking ({esc(f["reference"])})</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
    else:
        st.caption("This booking looks like a typical one; no single input moves the estimate by more than half a point.")
