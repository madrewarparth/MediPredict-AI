"""MediPredict AI - Data quality audit, feature policy and responsible-use rules."""

import pandas as pd
import streamlit as st

from src.config import FEATURE_POLICY, THEME
from src.ui_components import (
    callout,
    kpi_grid,
    load_metadata,
    page_header,
    render_governance_disclaimer,
    section,
    synthetic_data_notice,
)

meta = load_metadata()
if not meta:
    st.error("Model metadata not found. Run `python -m src.train_model`.")
    st.stop()

page_header(
    "Data & responsible use",
    "What went into the model, what was deliberately kept out, and how the scores may and may not be used.",
    eyebrow="Model & governance",
)
render_governance_disclaimer(compact=False)
synthetic_data_notice(meta)

audit = meta.get("cleaning_audit", {})
dropped = audit.get("dropped_reasons", {})
section("Data quality audit", "From the most recent training run.")
kpi_grid([
    {"label": "Raw records", "value": f"{audit.get('initial_rows', 0):,}"},
    {"label": "Kept after cleaning", "value": f"{audit.get('final_rows', 0):,}", "sub": f"{audit.get('retention_rate', 0)}% retained", "accent": THEME["success"]},
    {"label": "Invalid age", "value": f"{dropped.get('invalid_age', 0):,}", "sub": "outside 0–115", "accent": THEME["warning"]},
    {"label": "Negative lead time", "value": f"{dropped.get('negative_waiting_days', 0):,}", "sub": "appointment before booking", "accent": THEME["warning"]},
    {"label": "Duplicates / nulls", "value": f"{dropped.get('duplicate_appointment_id', 0) + dropped.get('null_records', 0):,}"},
])

section("Chronological split", "Bookings are ordered by booking time so the model is always tested on the future.")
splits = meta.get("splits", {})
st.dataframe(
    pd.DataFrame([
        {"Split": name.title(), "Bookings": s["samples"], "From": s["start_time"][:10], "To": s["end_time"][:10],
         "No-show rate": s["no_show_rate"] / 100}
        for name, s in splits.items() if isinstance(s, dict)
    ]),
    hide_index=True, width="stretch",
    column_config={"No-show rate": st.column_config.NumberColumn(format="percent"),
                   "Bookings": st.column_config.NumberColumn(format="localized")},
)

section("Feature policy", "Every raw column, and whether the model may use it.")
policy = pd.DataFrame([
    {"Column": col, "Status": p["status"].replace("_", " ").title(), "Category": p["category"], "Reason": p["reason"]}
    for col, p in FEATURE_POLICY.items()
])
st.dataframe(policy, hide_index=True, width="stretch",
             column_config={"Reason": st.column_config.TextColumn(width="large")})

audit_groups = meta.get("subgroup_audit", {})
if audit_groups:
    section(
        "Subgroup audit (test split)",
        f"At the review threshold of {meta['selected_operating_threshold']:.2f}. A group is well calibrated when "
        "<i>mean predicted</i> is close to <i>observed</i>, so the same score means the same thing for everyone.",
    )
    tabs = st.tabs(list(audit_groups))
    for tab, (gname, rows) in zip(tabs, audit_groups.items(), strict=True):
        with tab:
            t = pd.DataFrame(rows)
            t["gap"] = (t["mean_predicted"] - t["observed_rate"]) * 100
            st.dataframe(
                t[["group", "n", "observed_rate", "mean_predicted", "gap", "flag_rate", "recall", "precision", "roc_auc"]],
                hide_index=True, width="stretch",
                column_config={
                    "group": st.column_config.TextColumn(gname),
                    "n": st.column_config.NumberColumn("Bookings", format="localized"),
                    "observed_rate": st.column_config.NumberColumn("Observed", format="percent"),
                    "mean_predicted": st.column_config.NumberColumn("Mean predicted", format="percent"),
                    "gap": st.column_config.NumberColumn("Gap (pts)", format="%+.1f"),
                    "flag_rate": st.column_config.NumberColumn("Flagged", format="percent"),
                    "recall": st.column_config.NumberColumn("Recall", format="percent"),
                    "precision": st.column_config.NumberColumn("Precision", format="percent"),
                    "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                },
            )
    all_rows = [r for rows in audit_groups.values() for r in rows]
    worst_gap = max(abs(r["mean_predicted"] - r["observed_rate"]) for r in all_rows) * 100
    low_recall = [r for r in all_rows if r["recall"] is not None and r["recall"] < 0.8]
    note = f"Largest calibration gap across groups: <b>{worst_gap:.1f} points</b>. "
    if low_recall:
        note += (
            "Lower recall for " + ", ".join(f"<b>{r['group']}</b> ({r['recall'] * 100:.0f}%)" for r in low_recall)
            + ": that group's no-show rate is lower, so fewer of its bookings cross a single global threshold. "
            "If missed visits matter more for this group, review it with a lower threshold rather than retraining on age."
        )
    callout(note, "info")

section("Rules of use")
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown(
        """
        <div class="mp-card"><h4>✅ Allowed</h4><p>
        • Ordering a reminder-call list<br>
        • Choosing who gets an extra SMS or call<br>
        • Planning reminder staffing<br>
        • Reviewing estimates with a human in the loop</p></div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="mp-card"><h4>⛔ Never</h4><p>
        • Cancelling or rescheduling automatically<br>
        • Overbooking clinics on the basis of a score<br>
        • Judging medical urgency or diagnosing<br>
        • Restricting or delaying anyone's access to care</p></div>
        """,
        unsafe_allow_html=True,
    )

callout(
    "Estimates are statistical associations, not causes. Monitor flag rates by age, gender and neighbourhood "
    "after deployment, and re-train when the no-show rate or booking patterns shift.",
    "info",
)
st.caption(f"Trained {meta.get('training_timestamp', '')} · scikit-learn {meta.get('system_info', {}).get('scikit_learn_version', '?')}")
