"""MediPredict AI - Bulk scoring of a scheduling export into a prioritised reminder queue."""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.batch_schema import (
    MODEL_INPUT_COLUMNS,
    OPTIONAL_MODEL_COLUMNS,
    missing_model_columns,
    normalize_dataframe_columns,
    sample_booking_template,
)
from src.config import RISK_BANDS
from src.ui_components import (
    callout,
    get_engine,
    kpi_grid,
    page_header,
    render_governance_disclaimer,
    require_engine,
    section,
    show_chart,
    style_fig,
)

MAX_ROWS = 200_000

engine = require_engine()
default_thresh = round(float(engine.threshold), 2)

page_header(
    "Bulk review queue",
    "Upload a scheduling export. Every row is scored by the model and ranked so staff can work the riskiest "
    "bookings first.",
    eyebrow="Operations",
)
render_governance_disclaimer()


@st.cache_data(show_spinner=False, max_entries=8)
def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file. Save it as UTF-8 CSV and try again.")


@st.cache_data(show_spinner=False, max_entries=8)
def _score(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return get_engine().predict_batch(df, custom_threshold=threshold)


top_l, top_r = st.columns([3, 2], gap="large")
with top_l:
    uploaded = st.file_uploader("Scheduling export (CSV)", type=["csv"], label_visibility="visible")
with top_r:
    with st.container(border=True):
        st.markdown("**Expected columns**")
        st.caption(
            f"Required: {', '.join(MODEL_INPUT_COLUMNS)}. Optional: {', '.join(OPTIONAL_MODEL_COLUMNS)}, "
            "AppointmentRef. Common spellings (e.g. *appointment_date*, *sex*) are recognised."
        )
        st.download_button(
            "Download sample CSV",
            data=sample_booking_template().to_csv(index=False).encode("utf-8"),
            file_name="medipredict_booking_template.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

if uploaded is None:
    st.markdown(
        """
        <div class="mp-card" style="text-align:center;padding:2.4rem 1.2rem;">
          <div style="font-size:2rem;line-height:1;">📄</div>
          <h4 style="margin-top:.7rem;">No file yet</h4>
          <p>Upload a CSV above, or download the sample to see the format. The sample includes one invalid row
          so you can see how errors are reported.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    raw = _read_csv(uploaded.getvalue())
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the CSV: {e}")
    st.stop()

if raw.empty:
    st.error("The file has no rows.")
    st.stop()
if len(raw) > MAX_ROWS:
    st.error(f"The file has {len(raw):,} rows; the limit is {MAX_ROWS:,}. Split it and upload in parts.")
    st.stop()

batch = normalize_dataframe_columns(raw)
missing = missing_model_columns(batch)
if missing:
    hint = ""
    if "NoShowProb" in batch.columns:
        hint = (" This looks like an already-scored queue. The model needs the raw booking fields "
                "to produce its own estimate.")
    st.error(f"Missing required column(s): {', '.join(missing)}.{hint}")
    with st.expander("Columns found in your file"):
        st.write(list(raw.columns))
    st.stop()

if "Scholarship" not in batch.columns:
    callout("No <b>Scholarship</b> column found, so every row is treated as <i>not enrolled</i>.", "warning")
if "Neighbourhood" not in batch.columns:
    callout("No <b>Neighbourhood</b> column found, so every row is grouped as <i>Other</i>.", "warning")

threshold = st.slider(
    "Flag for review at or above", 0.05, 0.80, default_thresh, 0.01,
    help=f"Default {default_thresh:.2f} is the F1-optimal threshold from validation. Raise it to shrink the queue.",
)

with st.spinner(f"Scoring {len(batch):,} appointments…"):
    scored = _score(batch, threshold)

if "AppointmentRef" not in scored.columns:
    scored.insert(0, "AppointmentRef", [f"ROW-{i + 1:05d}" for i in range(len(scored))])
scored["AppointmentRef"] = scored["AppointmentRef"].astype(str)

valid = scored[scored["estimated_no_show_prob"].notna()].copy()
invalid = scored[scored["estimated_no_show_prob"].isna()].copy()
n_total, n_valid = len(scored), len(valid)
band_counts = {k: int((valid["risk_band"] == b["name"]).sum()) for k, b in RISK_BANDS.items()}
n_flag = int((valid["flagged_for_review"] == "Yes").sum())


def _share(n: int) -> str:
    return f"{n / n_valid * 100:.1f}% of scored" if n_valid else "—"


kpi_grid([
    {"label": "Rows scored", "value": f"{n_valid:,}",
     "sub": f"{n_total - n_valid:,} rejected" if n_total - n_valid else f"from {uploaded.name}"},
    {"label": "Flagged for review", "value": f"{n_flag:,}", "sub": _share(n_flag), "accent": RISK_BANDS["MODERATE"]["color"]},
    {"label": "Higher risk", "value": f"{band_counts['HIGHER']:,}", "sub": _share(band_counts["HIGHER"]), "accent": RISK_BANDS["HIGHER"]["color"]},
    {"label": "Expected no-shows", "value": f"{valid['estimated_no_show_prob'].sum():,.0f}",
     "sub": f"avg {valid['estimated_no_show_prob'].mean() * 100:.1f}% per booking" if n_valid else "—"},
])

if n_valid:
    fig = go.Figure()
    for key in ["LOWER", "MODERATE", "HIGHER"]:
        fig.add_bar(
            y=["Queue"], x=[band_counts[key]], orientation="h", name=RISK_BANDS[key]["short"],
            marker_color=RISK_BANDS[key]["color"], text=[f"{band_counts[key]:,}"], textposition="inside",
            hovertemplate=f"{RISK_BANDS[key]['name']}: %{{x:,}}<extra></extra>",
        )
    style_fig(fig, "Risk band mix", height=150)
    fig.update_layout(barmode="stack", margin=dict(l=8, r=8, t=44, b=8), yaxis=dict(visible=False))
    fig.update_layout(legend=dict(traceorder="normal"))
    fig.update_xaxes(visible=False)
    show_chart(fig)

section("Prioritised queue", "Sorted by estimated risk. Tick <b>Reviewed</b> as you work through it; ticks are included in the export.")

f1, f2 = st.columns([2, 3])
search = f1.text_input("Search reference", placeholder="e.g. APT-1002")
band_filter = f2.multiselect(
    "Risk bands",
    [RISK_BANDS[k]["name"] for k in ["HIGHER", "MODERATE", "LOWER"]],
    default=[RISK_BANDS["HIGHER"]["name"], RISK_BANDS["MODERATE"]["name"]],
)

queue = valid.sort_values("estimated_no_show_prob", ascending=False)
if "reviewed_refs" not in st.session_state:
    st.session_state.reviewed_refs = set()
queue["Reviewed"] = queue["AppointmentRef"].isin(st.session_state.reviewed_refs)

view = queue
if search:
    view = view[view["AppointmentRef"].str.contains(search.strip(), case=False, regex=False)]
if band_filter:
    view = view[view["risk_band"].isin(band_filter)]

cols = ["Reviewed", "AppointmentRef", "estimated_no_show_prob", "risk_band", "review_priority",
        "AppointmentDay", "waiting_days", "Age", "Gender"]
cols += [c for c in ["Neighbourhood"] if c in view.columns]
edited = st.data_editor(
    view[cols],
    hide_index=True,
    width="stretch",
    height=min(560, 38 + 35 * max(len(view), 1)),
    disabled=[c for c in cols if c != "Reviewed"],
    column_config={
        "Reviewed": st.column_config.CheckboxColumn("Reviewed", width="small"),
        "AppointmentRef": st.column_config.TextColumn("Reference"),
        "estimated_no_show_prob": st.column_config.ProgressColumn(
            "No-show risk", format="percent", min_value=0.0, max_value=1.0),
        "risk_band": st.column_config.TextColumn("Band"),
        "review_priority": st.column_config.TextColumn("Priority"),
        "AppointmentDay": st.column_config.TextColumn("Appointment"),
        "waiting_days": st.column_config.NumberColumn("Lead days", format="%d"),
    },
    key="queue_editor",
)
changed = set(edited.loc[edited["Reviewed"], "AppointmentRef"])
unticked = set(edited.loc[~edited["Reviewed"], "AppointmentRef"])
st.session_state.reviewed_refs = (st.session_state.reviewed_refs - unticked) | changed
st.caption(f"Showing {len(view):,} of {n_valid:,} scored appointments.")

if len(invalid):
    with st.expander(f"{len(invalid):,} row(s) could not be scored", icon=":material/error:"):
        st.dataframe(invalid[["AppointmentRef", "validation_error"]].rename(
            columns={"AppointmentRef": "Reference", "validation_error": "Reason"}), hide_index=True, width="stretch")

export = scored.copy()
export["Reviewed"] = export["AppointmentRef"].isin(st.session_state.reviewed_refs)
export = export.rename(columns={"estimated_no_show_prob": "NoShowProb", "risk_band": "RiskBand",
                                "review_priority": "ReviewPriority", "flagged_for_review": "FlaggedForReview",
                                "validation_error": "ValidationError", "waiting_days": "LeadDays"})
export = export.sort_values("NoShowProb", ascending=False, na_position="last")
st.download_button(
    "Download scored queue (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="scored_appointment_review_queue.csv",
    mime="text/csv",
    type="primary",
    icon=":material/download:",
)
