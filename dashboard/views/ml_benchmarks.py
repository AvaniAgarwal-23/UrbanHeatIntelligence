"""ML Benchmarks: model comparison on the 2024 validation and 2025 test years."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def render() -> None:
    ui.header(
        "ML Benchmarks",
        "Chronological split: train 2016 to 2023, validation 2024, untouched test 2025. Errors are in kelvin.",
    )
    metrics = data.report("model_metrics")
    rows = []
    for key, label in data.MODEL_LABELS.items():
        for split, name in (("validation_2024", "Validation 2024"), ("test_2025", "Test 2025")):
            m = metrics[key][split]
            rows.append({"Model": label, "Split": name, "RMSE": m["rmse"], "MAE": m["mae"], "R2": m["r2"]})
    df = pd.DataFrame(rows)

    best = metrics["best_model"].replace("_", " ")
    st.success(f"Champion model (lowest 2024 validation RMSE): {best}")
    metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True)
    fig = px.bar(df, x="Model", y=metric, color="Split", barmode="group", height=360)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(
        "The 2024 scores are much worse than 2025 for every model, so a lower 2025 error should not be read "
        "as a real improvement. A single year can be easier or harder to predict. See the Rolling Window page."
    )

    st.subheader("Performance by climate group (test 2025)")
    strat = pd.DataFrame(data.report("climate_stratified_risk")["climate_stratified_performance"])
    strat = strat.rename(columns={"climate_group": "Climate group", "sample_count": "Records", "rmse": "RMSE", "mae": "MAE", "r2": "R2",
                                  "mean_observed_uhi": "Observed UHI", "mean_predicted_uhi": "Predicted UHI"})
    st.dataframe(strat, hide_index=True, width="stretch")
    st.caption("The model works well for cold snowy and Mediterranean cities and poorly for Tropical Monsoon and hot desert groups. "
               "A negative R2 means the model is worse than predicting the group mean.")

    st.subheader("One month ahead air temperature forecast (test 2025)")
    fb = data.report("weather_forecast_benchmarks")
    bench = pd.DataFrame(fb["benchmarks"]).T.reset_index().rename(columns={"index": "Method", "rmse": "RMSE", "mae": "MAE", "r2": "R2"})
    bench["Method"] = bench["Method"].str.replace("_", " ")
    st.dataframe(bench, hide_index=True, width="stretch")
    st.warning(
        f"The machine learning forecaster did not beat the climatology baseline (skill {fb['skill_score_vs_climatology']:.3f}). "
        "For air temperature, the average of earlier years for the same month is hard to beat."
    )
