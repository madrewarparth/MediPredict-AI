"""MediPredict AI - Model performance on the held-out test split, with a threshold simulator."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import THEME
from src.ui_components import (
    callout,
    get_engine,
    kpi_grid,
    load_metadata,
    load_processed_data,
    page_header,
    require_engine,
    section,
    show_chart,
    style_fig,
    synthetic_data_notice,
)

engine = require_engine()
meta = load_metadata()
if not meta:
    st.error("Model metadata not found. Run `python -m src.train_model`.")
    st.stop()


@st.cache_data(show_spinner="Scoring the held-out test split…")
def test_scores():
    from src.feature_engineering import split_chronologically

    df = load_processed_data()
    *_, X_test, y_test, _, _ = split_chronologically(df)
    probs = get_engine().pipeline.predict_proba(X_test)[:, 1]
    return y_test.to_numpy(), probs


y_true, y_prob = test_scores()
base_rate = y_true.mean()
test = meta["test_evaluation"]["at_optimal_threshold"]
ceiling = meta.get("reference_ceiling")

page_header(
    "Model performance",
    f"Everything here is measured on the held-out test split: the latest {len(y_true):,} bookings, which the "
    "model never saw while being trained or tuned.",
    eyebrow="Model & governance",
)
synthetic_data_notice(meta)

kpi_grid([
    {"label": "Model", "value": meta["model_name"].replace(" (Calibrated)", ""), "sub": "sigmoid-calibrated" if "Calibrated" in meta["model_name"] else "uncalibrated"},
    {"label": "ROC-AUC", "value": f"{test['roc_auc']:.3f}",
     "sub": f"ceiling {ceiling['roc_auc']:.3f} on this data" if ceiling else "0.5 = random", "accent": THEME["secondary_blue"]},
    {"label": "PR-AUC", "value": f"{test['pr_auc']:.3f}", "sub": f"base rate {base_rate:.3f}", "accent": THEME["secondary_blue"]},
    {"label": "Brier score", "value": f"{test['brier_score']:.4f}", "sub": "lower is better", "accent": THEME["success"]},
])

if ceiling:
    callout(
        f"<b>Why isn't accuracy higher?</b> The synthetic generator adds random noise on purpose. Scoring the test split "
        f"with the generator's own noise-free formula gives ROC-AUC <b>{ceiling['roc_auc']:.3f}</b>, the best any model "
        f"can reach. This model reaches <b>{test['roc_auc']:.3f}</b>, so it captures nearly all the signal that exists.",
        "info",
    )

# ---------------------------------------------------------------- simulator
section("Threshold simulator", "Every appointment at or above the threshold is flagged for review. "
        "Lower thresholds catch more no-shows but give staff a longer list.")

# Cap the slider where at least ~0.5% of bookings would still be flagged; above that the metrics are noise.
max_thr = float(np.floor(np.quantile(y_prob, 0.995) * 100) / 100)
thr = st.slider("Review threshold", 0.05, max_thr, float(round(engine.threshold, 2)), 0.01,
                help=f"Trained default {engine.threshold:.2f} (F1-optimal on validation).")
pred = y_prob >= thr
tp = int((pred & (y_true == 1)).sum()); fp = int((pred & (y_true == 0)).sum())
fn = int((~pred & (y_true == 1)).sum()); tn = int((~pred & (y_true == 0)).sum())
recall = tp / (tp + fn) if tp + fn else 0
precision = tp / (tp + fp) if tp + fp else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
flagged = tp + fp

kpi_grid([
    {"label": "Staff review list", "value": f"{flagged:,}", "sub": f"{flagged / len(y_true) * 100:.1f}% of all bookings", "accent": THEME["warning"]},
    {"label": "No-shows caught (recall)", "value": f"{recall * 100:.1f}%", "sub": f"{tp:,} of {tp + fn:,}", "accent": THEME["success"]},
    {"label": "Flag hit rate (precision)", "value": f"{precision * 100:.1f}%", "sub": f"vs {base_rate * 100:.1f}% if picked at random"},
    {"label": "F1 score", "value": f"{f1:.3f}", "sub": f"threshold {thr:.2f}"},
])

c1, c2 = st.columns([2, 3], gap="medium")
with c1:
    z = [[tn, fp], [fn, tp]]
    labels = [["Attended · not flagged", "Attended · flagged"], ["Missed · not flagged", "Missed · flagged"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Not flagged", "Flagged"], y=["Attended", "Missed"],
        colorscale=[[0, "#F0FDFA"], [1, THEME["primary_teal"]]], showscale=False,
        text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=16),
        customdata=labels, hovertemplate="%{customdata}: %{z:,}<extra></extra>",
    ))
    style_fig(fig, "Confusion matrix", height=340)
    fig.update_yaxes(autorange="reversed")
    show_chart(fig)
with c2:
    grid = np.round(np.arange(0.05, max_thr + 0.001, 0.01), 2)
    n_pos = max(int((y_true == 1).sum()), 1)
    rec = [((y_prob >= t) & (y_true == 1)).sum() / n_pos * 100 for t in grid]
    load = [(y_prob >= t).mean() * 100 for t in grid]
    # precision is only shown where enough bookings are flagged for it to be stable
    prec = [(((y_prob >= t) & (y_true == 1)).sum() / (y_prob >= t).sum() * 100) if (y_prob >= t).sum() >= 200 else None
            for t in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=rec, name="No-shows caught %", line=dict(color=THEME["success"], width=2.5))
    fig.add_scatter(x=grid, y=load, name="Bookings flagged %", line=dict(color=THEME["warning"], width=2.5))
    fig.add_scatter(x=grid, y=prec, name="Precision %", line=dict(color=THEME["secondary_blue"], width=2, dash="dot"))
    fig.add_vline(x=thr, line_color=THEME["text_main"], line_width=1.5)
    style_fig(fig, "Trade-off across thresholds", height=340)
    fig.update_xaxes(title="Threshold")
    fig.update_yaxes(title="%", range=[0, 102])
    fig.update_xaxes(range=[0.05, max_thr])
    fig.update_layout(hovermode="x unified")
    show_chart(fig)

# ---------------------------------------------------------------- curves
section("Ranking and calibration")
c3, c4 = st.columns(2, gap="medium")
with c3:
    tab_roc, tab_pr = st.tabs(["ROC curve", "Precision–recall"])
    with tab_roc:
        roc = meta.get("roc_curve", {})
        fig = go.Figure()
        fig.add_scatter(x=roc.get("fpr", []), y=roc.get("tpr", []), name=f"Model (AUC {test['roc_auc']:.3f})",
                        line=dict(color=THEME["primary_teal"], width=2.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=THEME["neutral"], dash="dash"))
        style_fig(fig, "ROC curve", height=360)
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
        show_chart(fig)
    with tab_pr:
        pr = meta.get("pr_curve", {})
        fig = go.Figure()
        fig.add_scatter(x=pr.get("recall", []), y=pr.get("precision", []), name=f"Model (AP {test['pr_auc']:.3f})",
                        line=dict(color=THEME["secondary_blue"], width=2.5))
        fig.add_hline(y=base_rate, line_dash="dash", line_color=THEME["neutral"],
                      annotation_text=f"random ({base_rate * 100:.1f}%)", annotation_position="bottom right")
        style_fig(fig, "Precision–recall curve", height=360)
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        show_chart(fig)
with c4:
    cal = meta.get("calibration_curve", {})
    fig = go.Figure()
    fig.add_scatter(x=[0, 0.6], y=[0, 0.6], name="Perfect", line=dict(color=THEME["neutral"], dash="dash"))
    fig.add_scatter(x=cal.get("prob_pred", []), y=cal.get("prob_true", []), name="Model", mode="lines+markers",
                    line=dict(color=THEME["primary_teal"], width=2.5), marker=dict(size=8),
                    hovertemplate="predicted %{x:.1%} → observed %{y:.1%}<extra></extra>")
    style_fig(fig, f"Calibration on test split (Brier {test['brier_score']:.4f})", height=405)
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Observed no-show rate", tickformat=".0%")
    show_chart(fig)

# ---------------------------------------------------------------- candidates & importance
c5, c6 = st.columns([3, 2], gap="medium")
with c5:
    section("Candidate models", "Compared on the validation split; the best by ROC-AUC + PR-AUC was calibrated and tested.")
    rows = []
    for name, data in meta.get("candidate_models_validation", {}).items():
        m = data["metrics_at_optimal_threshold"]
        rows.append({"Model": name, "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Brier": m["brier_score"],
                     "Best threshold": data["optimal_threshold"], "F1": m["f1"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["ROC-AUC", "PR-AUC", "Brier", "Best threshold", "F1"]})
with c6:
    section("What the model relies on", "Drop in validation ROC-AUC when each input is shuffled.")
    feats = pd.DataFrame(meta.get("top_features", []))
    if not feats.empty:
        feats = feats.sort_values("importance")
        fig = go.Figure(go.Bar(
            x=feats["importance"], y=feats["label"], orientation="h", marker_color=THEME["primary_teal"],
            error_x=dict(type="data", array=feats["std"], color="#94A3B8", thickness=1),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        style_fig(fig, None, height=360)
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
        fig.update_xaxes(title="AUC drop")
        show_chart(fig)
    else:
        st.caption("Feature importance not available. Re-run training.")
