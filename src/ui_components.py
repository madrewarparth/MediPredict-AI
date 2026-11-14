"""
MediPredict AI - UI Component Library & Visual Theme System
Implements the 'Clinical Intelligence - Clinical Light' design system matching
the modern hospital operations aesthetic (Tailwind Slate/Sky-Blue palette).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.config import THEME, RISK_BANDS


def inject_custom_css():
    """Injects high-fidelity custom CSS matching the Clinical Light design system."""
    css = f"""
    <style>
    /* Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main Background & Font Styling */
    .stApp {{
        background-color: {THEME['app_bg']};
        color: {THEME['text_main']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['sidebar_bg']} !important;
        border-right: 1px solid {THEME['border']} !important;
    }}

    /* Card Panels */
    .mp-card {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }}

    .mp-card-elevated {{
        background-color: {THEME['surface_elevated']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}

    /* Metric Box */
    .mp-metric-box {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.25rem;
        text-align: left;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }}
    .mp-metric-box:hover {{
        border-color: #7dd3fc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }}
    .mp-metric-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {THEME['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }}
    .mp-metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {THEME['text_main']};
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }}
    .mp-metric-subtext {{
        font-size: 0.8rem;
        color: {THEME['text_muted']};
        margin-top: 0.5rem;
    }}

    /* Risk Badges */
    .mp-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .mp-badge-lower {{
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }}
    .mp-badge-moderate {{
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }}
    .mp-badge-higher {{
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }}

    /* Boundary Disclaimer Callouts */
    .mp-callout-info {{
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid {THEME['primary_teal']};
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0;
        color: {THEME['text_main']};
        font-size: 0.875rem;
    }}

    .mp-callout-warning {{
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid {THEME['warning']};
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0;
        color: {THEME['text_main']};
        font-size: 0.875rem;
    }}

    .mp-callout-governance {{
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0;
        color: #064e3b;
        font-size: 0.875rem;
    }}

    /* Buttons */
    button[kind="primary"] {{
        background-color: {THEME['primary_teal']} !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }}
    button[kind="primary"]:hover {{
        background-color: {THEME['teal_hover']} !important;
    }}

    /* Dataframe table styling */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        background: #ffffff;
    }}

    /* Headers */
    h1, h2, h3, h4 {{
        color: {THEME['text_main']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_header():
    """Renders product identity, logo mark, and operational disclaimer in sidebar."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid {THEME['border']}; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="background: {THEME['primary_teal']}; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #ffffff; font-size: 1.1rem; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">
                        +
                    </div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['text_main']}; letter-spacing: -0.01em; line-height: 1.1;">MediPredict AI</div>
                        <div style="font-size: 0.72rem; color: {THEME['text_muted']}; font-weight: 500;">Hospital Clinical Intelligence</div>
                    </div>
                </div>
                <div style="margin-top: 0.6rem; display: inline-flex; align-items: center; gap: 0.35rem; background: #ecfdf5; color: #065f46; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; border: 1px solid #a7f3d0;">
                    <span style="width: 6px; height: 6px; border-radius: 9999px; background: #059669;"></span>
                    <span>ML Decision Support</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "**Human-in-the-Loop Operations Support**\n"
            "Estimates appointment miss probability for reminder prioritization. "
            "Not a diagnostic or clinical triage system."
        )
        st.markdown(f"<div style='border-top: 1px solid {THEME['border']}; margin: 0.75rem 0;'></div>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", delta: str = None):
    """Renders a styled KPI card with Clinical Light design system aesthetics."""
    delta_html = ""
    if delta:
        delta_color = "#059669" if not delta.startswith("-") else "#d97706"
        bg_color = "#ecfdf5" if not delta.startswith("-") else "#fffbeb"
        border_color = "#a7f3d0" if not delta.startswith("-") else "#fde68a"
        delta_html = f"<span style='background: {bg_color}; border: 1px solid {border_color}; color: {delta_color}; font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.45rem; border-radius: 4px; margin-left: 0.4rem;'>{delta}</span>"

    html = f"""
    <div class="mp-metric-box">
        <div class="mp-metric-label">{label}</div>
        <div class="mp-metric-value">{value} {delta_html}</div>
        <div class="mp-metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(band_name: str) -> str:
    """Returns HTML for a color-coded risk band badge matching Clinical Light palette."""
    if "Lower" in band_name or "Low" in band_name:
        cls = "mp-badge-lower"
        dot = "#10b981"
    elif "Moderate" in band_name:
        cls = "mp-badge-moderate"
        dot = "#f59e0b"
    else:
        cls = "mp-badge-higher"
        dot = "#ef4444"
    return f"<span class='mp-badge {cls}'><span style='width: 6px; height: 6px; border-radius: 9999px; background: {dot};'></span>{band_name}</span>"


def render_governance_disclaimer():
    """Renders the standard clinical boundary notice required by project policy."""
    st.markdown(
        f"""
        <div class="mp-callout-governance">
            <strong>Operational Boundary Notice:</strong>
            MediPredict AI provides statistical estimates to support administrative reminder workflows. 
            Predictions are not facts and must <strong>never</strong> be used to diagnose patients, determine medical urgency, 
            deprioritize appointments, or deny access to healthcare services.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_theme_layout(title: str = None, exclude: list = None) -> dict:
    """Returns standardized layout dictionary for Plotly charts matching Clinical Light theme."""
    layout = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter, -apple-system, sans-serif"),
        title_font=dict(size=14, color="#0f172a"),
        xaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#0f172a", size=12),
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#0f172a", size=12),
        ),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#475569", size=11),
        ),
    )
    if title is not None:
        layout["title"] = title
    if exclude:
        for k in exclude:
            layout.pop(k, None)
    return layout
