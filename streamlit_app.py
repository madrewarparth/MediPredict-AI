"""
MediPredict AI - Executive Dashboard Portal (Clinical Light)
Hospital Appointment No-Show Prediction & Appointment Operations Intelligence
Organization: ALGIONS AI LABS PRIVATE LIMITED
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Page configuration


st.set_page_config(
    page_title="Helix MediPredict AI - Clinical Intelligence Portal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Seamless Full-Canvas CSS
st.markdown(
    """
    <style>
    /* Hide Streamlit default UI chrome to display exact Clinical Light design */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    /* Responsive Full-Canvas Layout for all viewports (Mobile, Tablet, Desktop) */
    html, body, [data-testid="stAppViewContainer"], .main, .main .block-container {
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        background-color: #f8faf9 !important;
        height: 100vh !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    iframe {
        width: 100% !important;
        height: 100vh !important;
        min-height: 100vh !important;
        border: none !important;
        background-color: #f8faf9 !important;
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the exact HTML dashboard provided
html_file = Path(__file__).resolve().parent / "assets" / "dashboard.html"

if html_file.exists():
    with open(html_file, "r", encoding="utf-8") as f:
        html_markup = f.read()
    # Render full-featured dashboard with Tailwind, fonts, and client-side interactions
    components.html(html_markup, height=2250, scrolling=True)
else:
    st.error("Dashboard template not found at assets/dashboard.html")
