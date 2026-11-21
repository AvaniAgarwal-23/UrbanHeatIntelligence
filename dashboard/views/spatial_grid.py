"""Spatial Grid: hotspots on the sampled ~1 km grid of one city."""
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def render() -> None:
    ui.header(
        "Spatial Grid",
        "Sampled grid points at about 1 km (not native 30 m Landsat pixels). Colour shows UHI at each point.",
    )
    names = sorted(data.cities().keys())
    city = st.selectbox("City", names, index=names.index("Delhi"))
    df = data.grid(city).dropna(subset=["uhi"])

    c1, c2 = st.columns(2)
    years = ["All"] + sorted(df["year"].unique().tolist())
    year = c1.selectbox("Year", years)
    months = ["All"] + list(range(1, 13))
    month = c2.selectbox("Month", months)
    if year != "All":
        df = df[df["year"] == year]
    if month != "All":
        df = df[df["month"] == month]
    if df.empty:
        st.warning("No valid grid points for this selection (cloud cover can hide a whole month).")
        return

    sample = df.sample(min(len(df), 6000), random_state=42)
    c1, c2, c3 = st.columns(3)
    c1.metric("Grid points", f"{len(df):,}")
    c2.metric("Mean UHI", f"{df['uhi'].mean():.2f} K")
    c3.metric("Hottest 5% start at", f"{df['uhi'].quantile(0.95):.2f} K")

    fig = px.scatter_map(
        sample, lat="latitude", lon="longitude", color="uhi", range_color=(-5, 5),
        color_continuous_scale=ui.SCALE, zoom=9, map_style="open-street-map", height=520,
        hover_data={"uhi": ":.2f", "ndvi": ":.2f", "ndbi": ":.2f", "population_density": ":.0f", "latitude": False, "longitude": False},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar_title="UHI (K)")
    st.plotly_chart(fig, width="stretch")
    st.caption("A random sample of up to 6,000 points is drawn for speed. The colour scale is capped at plus or minus 5 K.")

    left, right = st.columns(2)
    with left:
        hist = px.histogram(df, x="uhi", nbins=60, title="Distribution of UHI over grid points")
        hist.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(hist, width="stretch")
    with right:
        var = st.selectbox("Compare UHI with", ["ndvi", "ndbi", "ndwi", "population_density"], key="gridvar")
        sc = px.scatter(sample, x=var, y="uhi", opacity=0.35, title=f"UHI against {var} (association only)")
        sc.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(sc, width="stretch")
