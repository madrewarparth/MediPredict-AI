"""
MediPredict AI - UI Component Library & Visual Theme System
"Clinical Light" design system: one set of tokens, responsive layout rules,
reusable cards/badges/callouts and a shared Plotly style.
"""

import html
import json
from typing import Dict, Iterable, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import THEME, RISK_BANDS, PROCESSED_DATA_FILE, MODEL_METADATA_FILE

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- CSS
def inject_custom_css() -> None:
    """Injects the design system. Safe to call on every page run."""
    t = THEME
    css = f"""
    <style>

    :root {{
        --mp-bg: {t['app_bg']};
        --mp-card: {t['card_bg']};
        --mp-soft: {t['surface_elevated']};
        --mp-primary: {t['primary_teal']};
        --mp-primary-hover: {t['teal_hover']};
        --mp-text: {t['text_main']};
        --mp-muted: {t['text_muted']};
        --mp-border: {t['border']};
        --mp-radius: 14px;
        --mp-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 1px 3px rgba(15, 23, 42, .06);
    }}

    .stApp {{ background: var(--mp-bg); color: var(--mp-text); }}

    /* Keep the header (it holds the mobile sidebar toggle) but make it quiet */
    header[data-testid="stHeader"], header[data-testid="stAppHeader"], .stAppHeader {{
        background: rgba(246, 248, 247, .94) !important; backdrop-filter: blur(6px);
    }}
    [data-testid="stToolbar"] [data-testid="stToolbarActions"] {{ display: none; }}
    footer {{ visibility: hidden; }}

    .block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1280px; }}

    section[data-testid="stSidebar"] {{
        background: {t['sidebar_bg']};
        border-right: 1px solid var(--mp-border);
    }}

    h1, h2, h3, h4 {{ color: var(--mp-text); letter-spacing: -0.015em; }}
    h1 {{ font-weight: 700 !important; }}

    /* ---------- page header ---------- */
    .mp-eyebrow {{
        font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
        color: var(--mp-primary); margin-bottom: .35rem;
    }}
    .mp-title {{ font-size: clamp(1.5rem, 1.1rem + 1.4vw, 2.1rem) !important; font-weight: 700; line-height: 1.2 !important; padding: 0 !important;
        color: var(--mp-text); margin: 0 0 .55rem 0 !important; letter-spacing: -0.02em; }}
    .mp-subtitle {{ color: var(--mp-muted); font-size: .98rem; line-height: 1.55; max-width: 62ch; margin: 0; }}
    .mp-header {{ margin-bottom: 1.25rem; }}

    .mp-section-title {{ font-size: 1.08rem; font-weight: 650; color: var(--mp-text); margin: 1.6rem 0 .15rem 0; }}
    .mp-section-caption {{ color: var(--mp-muted); font-size: .86rem; margin: 0 0 .75rem 0; }}

    /* ---------- cards ---------- */
    .mp-card {{
        background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
        padding: 1.1rem 1.25rem; box-shadow: var(--mp-shadow); margin-bottom: .9rem;
    }}
    .mp-card h4 {{ margin: 0 0 .35rem 0; font-size: 1rem; }}
    .mp-card p {{ color: var(--mp-muted); font-size: .9rem; line-height: 1.55; margin: 0; }}

    .mp-kpi-grid {{ display: grid; gap: .8rem; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: .25rem 0 1rem 0; }}
    .mp-kpi {{
        background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
        padding: .95rem 1.05rem; box-shadow: var(--mp-shadow); min-width: 0;
        border-top: 3px solid var(--mp-accent, var(--mp-primary));
    }}
    .mp-kpi-label {{ font-size: .72rem; font-weight: 600; color: var(--mp-muted); text-transform: uppercase; letter-spacing: .05em; }}
    .mp-kpi-value {{ font-size: clamp(1.35rem, 1.1rem + .9vw, 1.75rem); font-weight: 700; color: var(--mp-text);
        font-variant-numeric: tabular-nums; line-height: 1.15; margin-top: .3rem; overflow-wrap: anywhere; }}
    .mp-kpi-sub {{ font-size: .8rem; color: var(--mp-muted); margin-top: .3rem; line-height: 1.35; }}

    /* backwards-compatible single metric card */
    .mp-metric-box {{ background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
        padding: .95rem 1.05rem; box-shadow: var(--mp-shadow); }}

    /* ---------- badges ---------- */
    .mp-badge {{ display: inline-flex; align-items: center; gap: .4rem; padding: .22rem .65rem; border-radius: 999px;
        font-size: .76rem; font-weight: 600; white-space: nowrap; border: 1px solid transparent; }}
    .mp-badge i {{ width: 7px; height: 7px; border-radius: 999px; display: inline-block; }}
    .mp-badge-lower {{ background: #ECFDF3; color: #166534; border-color: #BBF7D0; }}
    .mp-badge-moderate {{ background: #FFFBEB; color: #92400E; border-color: #FDE68A; }}
    .mp-badge-higher {{ background: #FEF2F2; color: #991B1B; border-color: #FECACA; }}
    .mp-badge-neutral {{ background: #F1F5F9; color: #334155; border-color: #E2E8F0; }}

    /* ---------- callouts ---------- */
    .mp-callout {{ border-radius: 12px; padding: .8rem 1rem; margin: .6rem 0 1rem 0; font-size: .875rem; line-height: 1.5;
        border: 1px solid; border-left-width: 4px; color: var(--mp-text); }}
    .mp-callout-info {{ background: #F0F9FF; border-color: #BAE6FD; border-left-color: #0284C7; }}
    .mp-callout-warning {{ background: #FFFBEB; border-color: #FDE68A; border-left-color: #D97706; }}
    .mp-callout-governance {{ background: #F0FDF9; border-color: #CCFBEF; border-left-color: var(--mp-primary); }}

    /* ---------- result hero (single prediction) ---------- */
    .mp-result {{ background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
        padding: 1.2rem 1.3rem; box-shadow: var(--mp-shadow); border-left: 5px solid var(--mp-accent, var(--mp-primary)); }}
    .mp-result-top {{ display: flex; flex-wrap: wrap; gap: .5rem; justify-content: space-between; align-items: center; }}
    .mp-result-prob {{ font-size: clamp(2.2rem, 1.6rem + 2.4vw, 3rem); font-weight: 700; line-height: 1; margin: .6rem 0 .25rem 0;
        font-variant-numeric: tabular-nums; }}
    .mp-meter {{ position: relative; height: 10px; border-radius: 999px; margin: .9rem 0 .35rem 0;
        background: linear-gradient(90deg, #DCFCE7 0 var(--mp-t), #FEF3C7 var(--mp-t) var(--mp-h), #FEE2E2 var(--mp-h) 100%); }}
    .mp-meter-fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--mp-accent); opacity: .9; }}
    .mp-meter-tick {{ position: absolute; top: -4px; width: 2px; height: 18px; background: var(--mp-text); border-radius: 2px; }}
    .mp-meter-legend {{ display: flex; justify-content: space-between; font-size: .74rem; color: var(--mp-muted); }}

    .mp-factor {{ display: grid; grid-template-columns: 1fr auto; gap: .2rem .75rem; align-items: center;
        padding: .6rem .8rem; border: 1px solid var(--mp-border); border-radius: 10px; margin-bottom: .45rem; background: var(--mp-card); }}
    .mp-factor-name {{ font-weight: 600; font-size: .88rem; }}
    .mp-factor-val {{ color: var(--mp-muted); font-size: .8rem; grid-column: 1 / 2; }}
    .mp-factor-delta {{ grid-row: 1 / span 2; grid-column: 2; font-weight: 700; font-size: .9rem; font-variant-numeric: tabular-nums; }}
    .mp-up {{ color: #B91C1C; }} .mp-down {{ color: #15803D; }}

    .mp-steps {{ display: grid; gap: .8rem; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
    .mp-step-n {{ width: 28px; height: 28px; border-radius: 8px; background: var(--mp-soft); color: var(--mp-primary);
        display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; margin-bottom: .5rem; }}

    /* ---------- widgets ---------- */
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
        background: var(--mp-primary); border: 1px solid var(--mp-primary); color: #fff; font-weight: 600; border-radius: 10px;
    }}
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{
        background: var(--mp-primary-hover); border-color: var(--mp-primary-hover);
    }}
    .stButton > button, .stDownloadButton > button {{ border-radius: 10px; }}
    [data-testid="stForm"] {{ background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius); box-shadow: var(--mp-shadow); }}
    [data-testid="stDataFrame"] {{ border: 1px solid var(--mp-border); border-radius: 12px; overflow: hidden; }}
    [data-testid="stPlotlyChart"] {{ background: var(--mp-card); border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
        box-shadow: var(--mp-shadow); padding: .35rem .25rem 0 .25rem; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: var(--mp-radius); }}
    [data-testid="stFileUploaderDropzone"] {{ border-radius: var(--mp-radius); background: var(--mp-card); }}

    /* ---------- responsive ---------- */
    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 3.8rem; }}
    }}
    @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .mp-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; }}
        .mp-kpi {{ padding: .75rem .8rem; }}
        .mp-card, .mp-result {{ padding: .95rem 1rem; }}
        .mp-subtitle {{ font-size: .92rem; }}
    }}
    @media (max-width: 380px) {{
        .mp-kpi-grid {{ grid-template-columns: 1fr; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------------------------------------------- components
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="mp-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="mp-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="mp-header">{eb}<h1 class="mp-title">{esc(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    cap = f'<p class="mp-section-caption">{caption}</p>' if caption else ""
    st.markdown(f'<div class="mp-section-title">{esc(title)}</div>{cap}', unsafe_allow_html=True)


def kpi_grid(items: Iterable[Dict[str, str]]) -> None:
    """items: dicts with label, value, optional sub and accent colour. Wraps responsively."""
    cards = []
    for it in items:
        accent = f' style="--mp-accent:{it["accent"]}"' if it.get("accent") else ""
        sub = f'<div class="mp-kpi-sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="mp-kpi"{accent}><div class="mp-kpi-label">{esc(it["label"])}</div>'
            f'<div class="mp-kpi-value">{esc(it["value"])}</div>{sub}</div>'
        )
    st.markdown(f'<div class="mp-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", delta: Optional[str] = None) -> None:
    """Single KPI card (kept for backwards compatibility; prefer kpi_grid)."""
    kpi_grid([{"label": label, "value": value if not delta else f"{value} {delta}", "sub": subtext}])


def callout(body_html: str, kind: str = "info") -> None:
    st.markdown(f'<div class="mp-callout mp-callout-{kind}">{body_html}</div>', unsafe_allow_html=True)


def render_risk_badge(band_name: str) -> str:
    """HTML pill for a risk band name."""
    name = str(band_name)
    if name == RISK_BANDS["LOWER"]["name"]:
        cls, dot = "mp-badge-lower", RISK_BANDS["LOWER"]["color"]
    elif name == RISK_BANDS["MODERATE"]["name"]:
        cls, dot = "mp-badge-moderate", RISK_BANDS["MODERATE"]["color"]
    elif name == RISK_BANDS["HIGHER"]["name"]:
        cls, dot = "mp-badge-higher", RISK_BANDS["HIGHER"]["color"]
    else:
        cls, dot = "mp-badge-neutral", THEME["neutral"]
    return f'<span class="mp-badge {cls}"><i style="background:{dot}"></i>{esc(name)}</span>'


def render_governance_disclaimer(compact: bool = True) -> None:
    """Standard non-clinical boundary notice."""
    if compact:
        callout(
            "<strong>Decision support only.</strong> Estimates help staff prioritise reminders. "
            "Never use them to diagnose, judge medical urgency, cancel, deprioritise or deny care.",
            "governance",
        )
    else:
        callout(
            "<strong>Operational boundary.</strong> MediPredict AI provides statistical estimates to support "
            "administrative reminder workflows. Predictions are not facts and must <strong>never</strong> be used to "
            "diagnose patients, determine medical urgency, deprioritise appointments, or deny access to care.",
            "governance",
        )


def synthetic_data_notice(meta: Optional[dict]) -> None:
    if meta and meta.get("data_is_synthetic"):
        callout(
            "<strong>Synthetic dataset.</strong> The bundled data is generated by <code>src/generate_data.py</code> "
            "in the Kaggle no-show schema. Patterns shown here reflect that generator, not a real hospital. "
            "Swap in real data and re-train for operational use.",
            "warning",
        )


def render_sidebar_header() -> None:
    """Brand mark at the top of the sidebar, above st.navigation's links."""
    from src.config import ASSETS_DIR

    st.logo(str(ASSETS_DIR / "logo.svg"), size="large", icon_image=str(ASSETS_DIR / "logo_icon.svg"))


# --------------------------------------------------------------------- charts
def get_plotly_theme_layout(title: Optional[str] = None, exclude: Optional[List[str]] = None) -> dict:
    """Base Plotly layout for the Clinical Light theme."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text_main"], family="Source Sans Pro, -apple-system, Segoe UI, sans-serif", size=12),
        title_font=dict(size=14, color=THEME["text_main"]),
        xaxis=dict(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0", linecolor="#E2E8F0",
                   tickfont=dict(color=THEME["text_muted"], size=11), title_font=dict(color=THEME["text_muted"], size=12)),
        yaxis=dict(gridcolor="#EEF2F6", zerolinecolor="#E2E8F0",
                   tickfont=dict(color=THEME["text_muted"], size=11), title_font=dict(color=THEME["text_muted"], size=12)),
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#475569", size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=THEME["border"], font=dict(color=THEME["text_main"])),
    )
    if title is not None:
        layout["title"] = dict(text=title, x=0.01, xanchor="left")
    for k in exclude or []:
        layout.pop(k, None)
    return layout


def style_fig(fig: go.Figure, title: Optional[str] = None, height: int = 360) -> go.Figure:
    fig.update_layout(**get_plotly_theme_layout(title), height=height)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


# --------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_metadata() -> Optional[dict]:
    if not MODEL_METADATA_FILE.exists():
        return None
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_engine():
    from src.predict import InferenceEngine

    return InferenceEngine()


def require_data() -> pd.DataFrame:
    df = load_processed_data()
    if df.empty:
        st.error("Processed dataset not found. Run `python -m src.train_model` (or `python -m src.data_preprocessing`).")
        st.stop()
    return df


def require_engine():
    try:
        return get_engine()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load the model: {e}")
        st.stop()
