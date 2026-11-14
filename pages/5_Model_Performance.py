"""
MediPredict AI - Model Performance & Governance Audit
Rigorous evaluation: candidate model comparison, test set metrics,
interactive threshold simulator, calibration reliability, and ROC/PR curves.
"""

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.config import THEME, MODEL_METADATA_FILE, PROCESSED_DATA_FILE
from src.ui_components import (
    inject_custom_css,
    render_sidebar_header,
    render_metric_card,
    get_plotly_theme_layout,
    render_governance_disclaimer,
)
from src.predict import get_inference_engine

st.set_page_config(page_title="Model Performance - MediPredict AI", page_icon="🧪", layout="wide")
inject_custom_css()
render_sidebar_header()

st.title("Model Performance & Governance Audit")
st.caption("Verification on untouched held-out test data, threshold sensitivity trade-offs, and calibration.")
render_governance_disclaimer()


@st.cache_data
def load_meta_and_test_data():
    if not MODEL_METADATA_FILE.exists():
        return None, None
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load test split for interactive threshold simulator
    engine = get_inference_engine()
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df_sorted = df.sort_values(by="ScheduledDay").reset_index(drop=True)
    n_total = len(df_sorted)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    test_df = df_sorted.iloc[n_train + n_val :].copy()

    # Pre-score test set
    scored_test = engine.predict_batch(test_df)
    return meta, scored_test


meta, scored_test = load_meta_and_test_data()

if not meta:
    st.error("Model metadata not found. Please train the model first.")
    st.stop()

# Section 1: Candidate Model Benchmark (Validation Partition)
st.markdown("### 1. Candidate Architecture Comparison (Validation Partition)")
st.info("Models were trained on chronological Train split (70%) and compared on Validation split (15%) prior to final selection.")

candidates = meta.get("candidate_models_validation", {})
comp_rows = []
for name, data in candidates.items():
    m = data["metrics_at_optimal_threshold"]
    comp_rows.append({
        "Model": name,
        "Val ROC-AUC": m["roc_auc"],
        "Val PR-AUC": m["pr_auc"],
        "Optimal Threshold": data["optimal_threshold"],
        "Val Precision": m["precision"],
        "Val Recall": m["recall"],
        "Val F1 Score": m["f1"],
    })

comp_df = pd.DataFrame(comp_rows)
st.dataframe(
    comp_df.style.highlight_max(subset=["Val ROC-AUC", "Val PR-AUC", "Val F1 Score"], color="#e0f2fe"),
    use_container_width=True,
    hide_index=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Interactive Decision Threshold Simulator
st.markdown("### 2. Decision Threshold Sensitivity Simulator (Held-Out Test Set)")
st.caption("Explore how adjusting the operating threshold impacts staff workload (False Positives) vs missed appointment capture (Recall).")

y_true = scored_test["no_show"].to_numpy()
y_probs = scored_test["estimated_no_show_prob"].to_numpy()

selected_thresh = st.slider(
    "Interactive Decision Threshold",
    min_value=0.05,
    max_value=0.60,
    value=float(meta.get("selected_operating_threshold", 0.170)),
    step=0.01,
    help="Move slider to simulate operational trade-offs on 16,580 untouched test appointments.",
)

# Live calculation at selected threshold
y_pred_live = (y_probs >= selected_thresh).astype(int)
p_live = precision_score(y_true, y_pred_live, zero_division=0)
r_live = recall_score(y_true, y_pred_live, zero_division=0)
f_live = f1_score(y_true, y_pred_live, zero_division=0)
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_true, y_pred_live).ravel()

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Test Recall (Sensitivity)", f"{r_live*100:.1f}%", f"{tp_l:,} of {tp_l+fn_l:,} no-shows captured")
with m2:
    render_metric_card("Test Precision", f"{p_live*100:.1f}%", f"{tp_l:,} true misses out of {tp_l+fp_l:,} flagged")
with m3:
    render_metric_card("Test F1-Score", f"{f_live:.4f}", f"Threshold = {selected_thresh:.3f}")
with m4:
    render_metric_card("Staff Review Queue", f"{tp_l + fp_l:,}", f"{((tp_l+fp_l)/len(y_true))*100:.1f}% of total test volume")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations: Confusion Matrix & ROC / PR Curves
r1, r2 = st.columns(2)

with r1:
    # Confusion Matrix Heatmap
    cm_matrix = np.array([[tn_l, fp_l], [fn_l, tp_l]])
    fig_cm = px.imshow(
        cm_matrix,
        text_auto=True,
        labels=dict(x="Predicted Attendance", y="Observed Ground Truth", color="Appointments"),
        x=["Attended (0)", "No-Show (1)"],
        y=["Attended (0)", "No-Show (1)"],
        color_continuous_scale=[[0, THEME["surface_elevated"]], [1, THEME["primary_teal"]]],
        title=f"Test Confusion Matrix at Threshold {selected_thresh:.3f}",
    )
    fig_cm.update_layout(coloraxis_showscale=False, **get_plotly_theme_layout())
    st.plotly_chart(fig_cm, use_container_width=True)

with r2:
    # ROC and PR curves
    tab_roc, tab_pr = st.tabs(["ROC Curve", "Precision-Recall Curve"])

    with tab_roc:
        roc_data = meta.get("roc_curve", {})
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc_data.get("fpr", []),
                y=roc_data.get("tpr", []),
                mode="lines",
                name="Production Model",
                line=dict(color=THEME["primary_teal"], width=3),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance Baseline (AUC=0.50)",
                line=dict(dash="dash", color=THEME["neutral"]),
            )
        )
        fig_roc.update_layout(
            title=f"ROC Curve (Held-out Test ROC-AUC = {meta['test_evaluation']['at_optimal_threshold']['roc_auc']:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            **get_plotly_theme_layout(),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_pr:
        pr_data = meta.get("pr_curve", {})
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=pr_data.get("recall", []),
                y=pr_data.get("precision", []),
                mode="lines",
                name="Production Model",
                line=dict(color=THEME["secondary_blue"], width=3),
            )
        )
        fig_pr.add_hline(
            y=0.207,
            line_dash="dash",
            line_color=THEME["warning"],
            annotation_text="Base No-Show Rate (~20.7%)",
        )
        fig_pr.update_layout(
            title=f"Precision-Recall Curve (Held-out Test PR-AUC = {meta['test_evaluation']['at_optimal_threshold']['pr_auc']:.4f})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            **get_plotly_theme_layout(),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

# Section 3: Probability Calibration Reliability Diagram
st.markdown("### 3. Probability Calibration (Reliability Diagram)")
st.caption("Verifies that a model-predicted probability corresponds to observed empirical attendance frequencies.")

cal_data = meta.get("calibration_curve", {})
fig_cal = go.Figure()
fig_cal.add_trace(
    go.Scatter(
        x=cal_data.get("prob_pred", []),
        y=cal_data.get("prob_true", []),
        mode="lines+markers",
        name="Calibrated Model",
        line=dict(color=THEME["primary_teal"], width=3),
        marker=dict(size=8, color=THEME["primary_teal"]),
    )
)
fig_cal.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration (Identity)",
        line=dict(dash="dash", color=THEME["neutral"]),
    )
)
fig_cal.update_layout(
    title=f"Reliability Diagram (Brier Score = {meta['test_evaluation']['at_optimal_threshold']['brier_score']:.4f})",
    xaxis_title="Mean Predicted Probability",
    yaxis_title="Observed True Fraction of No-Shows",
    **get_plotly_theme_layout(),
)
st.plotly_chart(fig_cal, use_container_width=True)

# Section 4: Key Feature Importances
st.markdown("### 4. Global Feature Importance Ranking")
st.caption("Relative factor contributions in the gradient boosted ensemble (non-causal associations).")

top_feats = meta.get("top_features", [])
if top_feats:
    feat_df = pd.DataFrame(top_feats).head(10)
    fig_imp = px.bar(
        feat_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Feature Importance Weights",
        color="importance",
        color_continuous_scale=[[0, THEME["surface_elevated"]], [1, THEME["primary_teal"]]],
    )
    fig_imp.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Relative Feature Importance",
        **get_plotly_theme_layout(exclude=["yaxis"]),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

