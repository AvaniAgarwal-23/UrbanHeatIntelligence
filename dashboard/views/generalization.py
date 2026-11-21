"""Generalization: how well the model predicts cities it has never seen."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def render() -> None:
    ui.header(
        "Generalization to Unseen Cities",
        "A model that works on cities it was trained on may fail on a new city. These tests check that.",
    )
    gen = data.report("generalization")
    metrics = data.report("model_metrics")
    seen = {
        "Ridge": metrics["Ridge_Regression"]["test_2025"],
        "Random Forest": metrics["Random_Forest"]["test_2025"],
        "HistGradientBoosting": metrics["Hist_Gradient_Boosting"]["test_2025"],
    }

    rows = [{"Setting": "Seen cities (2025 test)", "Model": k, "RMSE": v["rmse"], "R2": v["r2"]} for k, v in seen.items()]
    rows += [{"Setting": "Unseen cities (6 city groups)", "Model": k, "RMSE": v["rmse"], "R2": v["r2"]} for k, v in gen["leave_cities_out"].items()]
    for setting, vals in gen["cross_region"].items():
        rows += [{"Setting": setting, "Model": k, "RMSE": v["rmse"], "R2": v["r2"]} for k, v in vals.items() if isinstance(v, dict)]
    df = pd.DataFrame(rows)
    df["Model"] = df["Model"].replace({"Mean baseline": "Predict the mean"})

    plot = df[df["RMSE"] < 5]  # Ridge India to International (RMSE 7.5) would squash the chart
    fig = px.bar(plot, x="Setting", y="RMSE", color="Model", barmode="group", height=380, labels={"RMSE": "RMSE (K)"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Ridge trained on India and tested on international cities has an RMSE of 7.48 K and is left out of the chart.")
    st.dataframe(df.round(3), hide_index=True, width="stretch")

    hgb = gen["leave_cities_out"]["HistGradientBoosting"]
    st.warning(
        f"On cities seen in training HistGradientBoosting reaches R2 {seen['HistGradientBoosting']['r2']:.2f}. "
        f"On cities it has never seen this drops to {hgb['r2']:.2f} (RMSE {hgb['rmse']:.2f} K), only a little better than "
        "predicting the training mean. Treat the model as a tool for studying known cities over time, not for new cities."
    )

    st.subheader("Per city error for unseen cities (HistGradientBoosting)")
    left, right = st.columns(2)
    best = pd.DataFrame(gen["per_city_rmse_hgb"]["best5"]).rename(columns={"city": "City", "rmse": "RMSE (K)"})[["City", "RMSE (K)"]]
    worst = pd.DataFrame(gen["per_city_rmse_hgb"]["worst5"]).rename(columns={"city": "City", "rmse": "RMSE (K)"})[["City", "RMSE (K)"]]
    left.markdown("**Easiest five**")
    left.dataframe(best.round(2), hide_index=True)
    right.markdown("**Hardest five**")
    right.dataframe(worst.round(2), hide_index=True)
    st.caption(f"HistGradientBoosting beat the mean baseline in {gen['n_cities_hgb_beats_mean']} of {gen['n_cities']} cities.")
