"""Rolling Window: stability of the models over four chronological folds."""
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def render() -> None:
    ui.header(
        "Rolling Window Validation",
        "Four folds, each with five training years and one validation year. The 2025 test year is never used here.",
    )
    traj = data.rolling_csv("rolling_temporal_trajectory").copy()
    traj["Model"] = traj["model"].map(data.MODEL_LABELS)

    fig = px.line(traj, x="validation_year", y="rmse", color="Model", markers=True, labels={"validation_year": "Validation year", "rmse": "RMSE (K)"}, height=360)
    fig.update_xaxes(dtick=1)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("2024 is the hardest year for every model and 2023 the easiest.")

    st.subheader("Stability summary")
    summary = data.rolling_csv("rolling_stability_summary").copy()
    summary["model"] = summary["model"].map(data.MODEL_LABELS)
    st.dataframe(
        summary[["model", "mean_rmse", "std_rmse", "mean_mae", "mean_r2", "mean_mbe", "pct_folds_outperform_baseline"]].rename(
            columns={"model": "Model", "mean_rmse": "Mean RMSE", "std_rmse": "Std RMSE", "mean_mae": "Mean MAE",
                     "mean_r2": "Mean R2", "mean_mbe": "Mean bias", "pct_folds_outperform_baseline": "% folds beating Ridge"}),
        hide_index=True, width="stretch",
    )
    st.info("HistGradientBoosting and Random Forest are almost tied (mean RMSE 1.740 and 1.741 K) and both beat Ridge in every fold.")

    st.subheader("Fixed split against rolling folds")
    comp = data.rolling_csv("fixed_vs_rolling_comparison").copy()
    comp["model"] = comp["model"].map(data.MODEL_LABELS)
    st.dataframe(
        comp[["model", "fixed_val_2024_rmse", "fixed_test_2025_rmse", "rolling_mean_rmse", "rolling_std_rmse", "rank_fixed_test", "rank_rolling_mean"]],
        hide_index=True, width="stretch",
    )

    st.subheader("Where the error is largest")
    tab_a, tab_b, tab_c = st.tabs(["By region", "By climate group", "By season"])
    with tab_a:
        reg = data.rolling_csv("rolling_regional_robustness")
        agg = reg.groupby(["region", "model"], as_index=False)[["rmse", "mae", "r2"]].mean()
        agg["model"] = agg["model"].map(data.MODEL_LABELS)
        st.dataframe(agg.round(3), hide_index=True, width="stretch")
    with tab_b:
        clim = data.rolling_csv("rolling_climate_robustness")
        clim = clim[clim["model"] == "Hist_Gradient_Boosting"].sort_values("rmse", ascending=False)
        st.dataframe(clim[["climate_group", "sample_count", "rmse", "mae", "r2"]].round(3), hide_index=True, width="stretch")
    with tab_c:
        sea = data.rolling_csv("rolling_seasonal_robustness")
        sea = sea[sea["model"] == "Hist_Gradient_Boosting"].groupby("season", as_index=False)[["rmse", "mae", "r2"]].mean()
        st.dataframe(sea.round(3), hide_index=True, width="stretch")
