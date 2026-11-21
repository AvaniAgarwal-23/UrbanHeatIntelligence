"""Urban Heat Island dashboard (Streamlit).

Run from the repository root:
    streamlit run dashboard/streamlit_app.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st

st.set_page_config(page_title="Urban Heat Island Analysis", page_icon="🌡️", layout="wide")

from dashboard.views import (  # noqa: E402
    city_deep_dive,
    cooling_simulator,
    explainability,
    generalization,
    ml_benchmarks,
    overview,
    rolling_window,
    spatial_grid,
)

pages = [
    st.Page(overview.render, title="Overview Map", url_path="overview", default=True),
    st.Page(city_deep_dive.render, title="City Deep Dive", url_path="city-deep-dive"),
    st.Page(spatial_grid.render, title="Spatial Grid", url_path="spatial-grid"),
    st.Page(ml_benchmarks.render, title="ML Benchmarks", url_path="ml-benchmarks"),
    st.Page(rolling_window.render, title="Rolling Window", url_path="rolling-window"),
    st.Page(generalization.render, title="Generalization", url_path="generalization"),
    st.Page(explainability.render, title="Explainability", url_path="explainability"),
    st.Page(cooling_simulator.render, title="Cooling Simulator", url_path="cooling-simulator"),
]

with st.sidebar:
    st.markdown("### Urban Heat Island Analysis")
    st.caption("42 cities, 2016 to 2025. Landsat 8/9, ERA5-Land and GHSL.")

st.navigation(pages).run()
