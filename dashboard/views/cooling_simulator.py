"""Cooling Simulator: what if greening, cool roofs and water changed NDVI, NDBI and NDWI."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui


@st.cache_data(show_spinner="Running the simulation for all cities...")
def run_all(d_ndvi: float, d_ndbi: float, d_ndwi: float):
    df = data.simulator().simulate_all_cities(delta_ndvi=d_ndvi, delta_ndbi=d_ndbi, delta_ndwi=d_ndwi)
    reg = {c: v["region"] for c, v in data.cities().items()}
    df["region"] = df["city"].map(reg)
    return df


@st.cache_data(show_spinner=False)
def run_city(city: str, d_ndvi: float, d_ndbi: float, d_ndwi: float):
    return data.simulator().simulate_city_intervention(city, delta_ndvi=d_ndvi, delta_ndbi=d_ndbi, delta_ndwi=d_ndwi)


def render() -> None:
    ui.header(
        "Cooling Simulator",
        "A what if planner. Change vegetation, built up surface and water, and see how the model's UHI estimate for 2025 changes.",
    )
    ui.estimate_note()

    c1, c2, c3 = st.columns(3)
    d_ndvi = c1.slider("Green canopy (change in NDVI)", 0.0, 0.30, 0.15, 0.01)
    d_ndbi = c2.slider("Cool roofs and pavements (change in NDBI)", -0.30, 0.0, -0.15, 0.01)
    d_ndwi = c3.slider("Water and wetlands (change in NDWI)", 0.0, 0.20, 0.05, 0.01)

    df = run_all(d_ndvi, d_ndbi, d_ndwi)
    helped = int((df["mean_cooling_celsius"] > 0).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Average estimated cooling", f"{df['mean_cooling_celsius'].mean():.3f} K")
    best = df.iloc[0]
    m2.metric("Largest estimated cooling", best["city"], f"{best['mean_cooling_celsius']:.2f} K")
    m3.metric("Cities with estimated cooling", f"{helped} of {len(df)}")

    fig = px.bar(df.sort_values("mean_cooling_celsius"), x="mean_cooling_celsius", y="city", orientation="h", color="region", height=900,
                 labels={"mean_cooling_celsius": "Estimated mean cooling (K), negative means warming", "city": ""})
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.warning(
        "For several desert cities (for example Cairo, Jaipur and Jodhpur) the model predicts slight warming. "
        "Physically this is hard to believe. The model learned a pooled pattern from all climates and does not know the cause, "
        "so these numbers are estimates and not the measured effect of a real project."
    )

    st.subheader("One city in detail")
    city = st.selectbox("City", sorted(df["city"]), index=sorted(df["city"]).index("Kolkata"))
    sim = run_city(city, d_ndvi, d_ndbi, d_ndwi)
    mb = sim["monthly_breakdown"]
    months = [r["month"] for r in mb]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=months, y=[r["baseline_uhi"] for r in mb], name="Baseline UHI", mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=months, y=[r["counterfactual_uhi"] for r in mb], name="With changes", mode="lines+markers"))
    fig2.update_layout(height=340, xaxis_title="Month of 2025", yaxis_title="Predicted UHI (K)", margin=dict(l=0, r=0, t=10, b=0))
    fig2.update_xaxes(dtick=1)
    st.plotly_chart(fig2, width="stretch")
    st.caption(sim["policy_summary"] + " This is an association learned by the model, not a guaranteed outcome.")
