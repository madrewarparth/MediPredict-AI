"""MediPredict AI - Home: what the tool does, live headline numbers, where to go next."""

import streamlit as st

from src.config import RISK_BANDS, THEME
from src.ui_components import (
    callout,
    kpi_grid,
    load_metadata,
    load_processed_data,
    page_header,
    render_governance_disclaimer,
    section,
    synthetic_data_notice,
)

meta = load_metadata()
df = load_processed_data()

page_header(
    "Spot likely no-shows while there's still time to send a reminder",
    "MediPredict AI estimates how likely a booked appointment is to be missed, using only facts known when "
    "it is booked. Staff use those estimates to decide who gets a reminder call first.",
    eyebrow="Appointment operations intelligence",
)

render_governance_disclaimer()

if meta is None or df.empty:
    st.warning("Model or dataset not found. Run `python -m src.train_model` to build them.")
    st.stop()

test = meta["test_evaluation"]["at_optimal_threshold"]
cm = test["confusion_matrix"]
cut = meta.get("risk_band_cutoffs", {})
rates = meta.get("test_band_observed_no_show_rate", {})

kpi_grid([
    {"label": "Appointments analysed", "value": f"{len(df):,}",
     "sub": f"{df['AppointmentDay'].min():%d %b %Y} – {df['AppointmentDay'].max():%d %b %Y}"},
    {"label": "Historical no-show rate", "value": f"{df['no_show'].mean() * 100:.1f}%",
     "sub": f"{int(df['no_show'].sum()):,} missed appointments", "accent": THEME["no_show"]},
    {"label": "Test ROC-AUC", "value": f"{test['roc_auc']:.3f}",
     "sub": "Held-out, chronologically later data", "accent": THEME["secondary_blue"]},
    {"label": "No-shows caught at threshold", "value": f"{test['recall'] * 100:.0f}%",
     "sub": f"{cm['tp']:,} of {cm['tp'] + cm['fn']:,} on the test split", "accent": THEME["success"]},
])

synthetic_data_notice(meta)

section("Start here")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("**Score one booking**")
        st.caption("Enter the booking details and get an estimate, a risk band and what drove it.")
        st.page_link("app_pages/single_prediction.py", label="Score an appointment", icon=":material/arrow_forward:")
with c2:
    with st.container(border=True):
        st.markdown("**Prioritise a whole list**")
        st.caption("Upload a scheduling CSV, get a ranked reminder queue, export it.")
        st.page_link("app_pages/bulk_prediction.py", label="Bulk review queue", icon=":material/arrow_forward:")
with c3:
    with st.container(border=True):
        st.markdown("**Check the evidence**")
        st.caption("Test metrics, calibration and a threshold simulator for staff workload.")
        st.page_link("app_pages/model_performance.py", label="Model performance", icon=":material/arrow_forward:")

section("How the estimate is made")
st.markdown(
    """
    <div class="mp-steps">
      <div class="mp-card"><div class="mp-step-n">1</div><h4>Booking-time facts only</h4>
        <p>Age, gender, neighbourhood, welfare enrolment, booking time and lead time. No diagnoses, no IDs, no SMS status.</p></div>
      <div class="mp-card"><div class="mp-step-n">2</div><h4>Validated on the future</h4>
        <p>Trained on the earliest 70% of bookings, tuned on the next 15%, tested once on the latest 15%.</p></div>
      <div class="mp-card"><div class="mp-step-n">3</div><h4>Calibrated probabilities</h4>
        <p>A 30% estimate means roughly 3 in 10 similar bookings were missed on unseen data.</p></div>
      <div class="mp-card"><div class="mp-step-n">4</div><h4>Staff decide</h4>
        <p>Bands suggest who to call first. Nothing is cancelled, rescheduled or denied automatically.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

section("Risk bands", "Cut-offs are learned on validation data. Observed rates are from the held-out test split.")


def _rate(key):
    v = rates.get(key)
    return "n/a" if v is None else f"{v * 100:.0f}% actually missed"


kpi_grid([
    {"label": RISK_BANDS["LOWER"]["short"], "value": f"< {cut.get('moderate', 0) * 100:.0f}%",
     "sub": f"{_rate('lower')} · {RISK_BANDS['LOWER']['action']}", "accent": RISK_BANDS["LOWER"]["color"]},
    {"label": RISK_BANDS["MODERATE"]["short"], "value": f"{cut.get('moderate', 0) * 100:.0f}–{cut.get('higher', 0) * 100:.0f}%",
     "sub": f"{_rate('moderate')} · {RISK_BANDS['MODERATE']['action']}", "accent": RISK_BANDS["MODERATE"]["color"]},
    {"label": RISK_BANDS["HIGHER"]["short"], "value": f"≥ {cut.get('higher', 0) * 100:.0f}%",
     "sub": f"{_rate('higher')} · {RISK_BANDS['HIGHER']['action']}", "accent": RISK_BANDS["HIGHER"]["color"]},
])

callout(
    f"Model: <strong>{meta['model_name']}</strong> · trained {meta['training_timestamp'][:10]} · "
    f"review threshold {meta['selected_operating_threshold']:.2f}.",
    "info",
)
