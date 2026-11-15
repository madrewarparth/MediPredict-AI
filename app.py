"""
MediPredict AI - Streamlit entrypoint
Hospital appointment no-show prediction & reminder-prioritisation decision support.

Run:  streamlit run app.py
"""

import streamlit as st

from src.ui_components import inject_custom_css, render_sidebar_header

st.set_page_config(
    page_title="MediPredict AI",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_custom_css()
render_sidebar_header()

pages = {
    "": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Operations": [
        st.Page("app_pages/single_prediction.py", title="Score an appointment", icon=":material/person_search:", url_path="score"),
        st.Page("app_pages/bulk_prediction.py", title="Bulk review queue", icon=":material/upload_file:", url_path="bulk"),
    ],
    "Insights": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", url_path="overview"),
        st.Page("app_pages/analytics.py", title="Analytics", icon=":material/insights:", url_path="analytics"),
    ],
    "Model & governance": [
        st.Page("app_pages/model_performance.py", title="Model performance", icon=":material/monitoring:", url_path="model"),
        st.Page("app_pages/responsible_use.py", title="Data & responsible use", icon=":material/verified_user:", url_path="governance"),
    ],
}

nav = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.caption("Human-in-the-loop decision support. Not a diagnostic or clinical triage system.")

nav.run()
