"""Overview Map: all 42 cities coloured by mean UHI, with the heat risk ranking."""
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def render() -> None:
    ui.header(
        "Overview Map",
        "42 cities (24 Indian, 18 international), monthly data from 2016 to 2025. "
        "UHI = urban land surface temperature minus a 10 km rural reference ring.",
    )
    risk = data.risk()
    metrics = data.report("model_metrics")
    champion = metrics["Hist_Gradient_Boosting"]["test_2025"]

    region = st.radio("Region", ["All", "Indian", "International"], horizontal=True)
    view = risk if region == "All" else risk[risk["region"] == region]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities shown", len(view))
    c2.metric("Mean UHI", f"{view['mean_uhi'].mean():.2f} K")
    top = view.sort_values("vulnerability_score", ascending=False).iloc[0]
    c3.metric("Highest risk score", top["city"], f"{top['vulnerability_score']:.1f}")
    c4.metric("Champion test R2 (2025)", f"{champion['r2']:.3f}")

    fig = px.scatter_map(
        view,
        lat="latitude",
        lon="longitude",
        color="mean_uhi",
        size="vulnerability_score",
        size_max=26,
        hover_name="city",
        hover_data={
            "climate_group": True,
            "mean_uhi": ":.2f",
            "vulnerability_score": ":.1f",
            "risk_tier": True,
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale=ui.SCALE,
        zoom=1 if region != "Indian" else 3.4,
        center=None if region != "Indian" else {"lat": 22.5, "lon": 80},
        map_style="open-street-map",
        height=520,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar_title="Mean UHI (K)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Bubble size shows the heat risk score. Negative mean UHI means the sampled urban surface "
        "was cooler than its rural ring during the daytime overpass, not that the city is safe from heat."
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Heat risk ranking")
        table = view.sort_values("vulnerability_score", ascending=False)[
            ["city", "region", "climate_group", "mean_uhi", "mean_pop_density", "vulnerability_score", "risk_tier"]
        ].rename(
            columns={
                "mean_uhi": "Mean UHI (K)",
                "mean_pop_density": "Population density",
                "vulnerability_score": "Risk score",
                "risk_tier": "Tier",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch", height=360)
    with right:
        st.subheader("Cities per risk tier")
        tiers = view["risk_tier"].value_counts().reindex(
            ["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"], fill_value=0
        )
        bar = px.bar(x=tiers.index, y=tiers.values, labels={"x": "Tier", "y": "Cities"})
        bar.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(bar, width="stretch")

    with st.expander("How the risk score works"):
        st.markdown(
            "Risk = 100 x (0.50 x UHI percentile + 0.30 x population percentile + 0.20 x canopy deficit percentile). "
            "Percentiles are taken inside each climate group, so a city that is alone in its group gets 100 "
            "by construction. Read it as a screening tool, not a precise order."
        )
