"""City Deep Dive: ten year monthly profile of one city."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render() -> None:
    ui.header(
        "City Deep Dive",
        "Ten year monthly profile (2016 to 2025): urban and rural surface temperature, UHI and seasonal pattern.",
    )
    monthly = data.monthly()
    city = st.selectbox("City", sorted(monthly["city"].unique()), index=sorted(monthly["city"].unique()).index("Delhi"))
    df = monthly[monthly["city"] == city].sort_values("date")
    row = df.iloc[0]
    st.caption(f"{city} | {row['region']} | climate group: {row['climate_group']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean UHI", f"{df['uhi'].mean():.2f} K")
    c2.metric("Largest monthly UHI", f"{df['uhi'].max():.2f} K")
    c3.metric("Months with no thermal data", f"{df['uhi'].isna().mean() * 100:.1f}%")
    c4.metric("Mean population density", f"{df['population_density'].mean():,.0f} per km2")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["lst"], name="Urban LST", mode="lines", connectgaps=False))
    fig.add_trace(go.Scatter(x=df["date"], y=df["rural_lst"], name="Rural LST", mode="lines", connectgaps=False))
    fig.update_layout(
        title="Land surface temperature (degrees C). Gaps are months hidden by cloud.",
        height=340, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="LST (C)",
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        uhi = px.bar(df, x="date", y="uhi", title="Monthly UHI (K)", labels={"uhi": "UHI (K)", "date": ""})
        uhi.add_hline(y=0, line_dash="dot")
        uhi.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(uhi, width="stretch")
    with right:
        seasonal = df.groupby("month")["uhi"].agg(["mean", "std"]).reindex(range(1, 13)).reset_index()
        season = px.bar(
            seasonal, x=seasonal["month"].map(lambda m: MONTHS[m - 1]), y="mean", error_y="std",
            title="Average UHI by month of year (bars show spread)", labels={"x": "", "mean": "Mean UHI (K)"},
        )
        season.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(season, width="stretch")

    st.subheader("Surface and weather drivers")
    pick = st.selectbox(
        "Variable", ["ndvi", "ndbi", "ndwi", "air_temperature", "relative_humidity", "total_precipitation", "wind_speed", "solar_radiation"]
    )
    line = px.line(df, x="date", y=pick, title=f"{pick} over time")
    line.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(line, width="stretch")

    with st.expander("Show the monthly table"):
        st.dataframe(
            df[["year", "month", "lst", "rural_lst", "uhi", "ndvi", "ndbi", "ndwi", "air_temperature", "relative_humidity"]],
            hide_index=True, width="stretch",
        )
