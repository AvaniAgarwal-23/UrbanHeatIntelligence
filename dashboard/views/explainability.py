"""Explainability: SHAP feature attributions and a test of how reliable they are."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data, ui


def feature_group(name: str) -> str:
    n = name.lower()
    if "pixel_count" in n or "landsat_obs" in n or "population_epoch" in n:
        return "Data coverage"
    if n in ("latitude", "longitude"):
        return "Location"
    if n.startswith("month_"):
        return "Season"
    if any(t in n for t in ("ndvi", "ndbi", "ndwi", "built_veg", "veg_moisture")):
        return "Surface indices"
    if "pop" in n:
        return "Population"
    return "Weather"


def render() -> None:
    ui.header(
        "Explainability",
        "TreeSHAP attributions for the HistGradientBoosting champion on the 2025 test records.",
    )
    shap_df = data.shap_values()
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]

    scope = st.radio("Scope", ["Global", "India", "International", "Climate group", "Single city"], horizontal=True)
    view = shap_df
    label = "all 42 cities"
    if scope == "India":
        view, label = shap_df[shap_df["region"] == "Indian"], "Indian cities"
    elif scope == "International":
        view, label = shap_df[shap_df["region"] == "International"], "international cities"
    elif scope == "Climate group":
        group = st.selectbox("Climate group", sorted(shap_df["climate_group"].unique()))
        view, label = shap_df[shap_df["climate_group"] == group], group
    elif scope == "Single city":
        city = st.selectbox("City", sorted(shap_df["city"].unique()))
        view, label = shap_df[shap_df["city"] == city], city

    imp = view[shap_cols].abs().mean().sort_values(ascending=False)
    imp.index = [c.replace("shap_", "") for c in imp.index]
    st.caption(f"{len(view)} records, {label}.")

    left, right = st.columns([3, 2])
    with left:
        top = imp.head(15).iloc[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Mean absolute SHAP (K)", "y": ""}, height=460)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with right:
        share = imp.groupby(imp.index.map(feature_group)).sum()
        share = share / share.sum()
        pie = px.pie(values=share.values, names=share.index, hole=0.5, height=460, title="Share of total attribution")
        pie.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(pie, width="stretch")
    st.caption("Data coverage (valid pixel counts) and location take a large share. That says the model uses them to tell cities apart, not that they physically drive UHI.")

    st.subheader("How does one feature change the prediction?")
    feats = [c for c in imp.index if c in data.test_features().columns]
    feat = st.selectbox("Feature", feats[:20])
    merged = view[["city", "year", "month", "shap_" + feat]].merge(
        data.test_features()[["city", "year", "month", feat]], on=["city", "year", "month"], how="inner"
    )
    dep = px.scatter(merged, x=feat, y="shap_" + feat, color="city" if scope == "Single city" else None, opacity=0.6, height=340,
                     labels={"shap_" + feat: "SHAP value (K)"})
    dep.add_hline(y=0, line_dash="dot")
    dep.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(dep, width="stretch")

    if scope == "Single city":
        st.subheader("Explain one month")
        month = st.selectbox("Month of 2025", sorted(view["month"].unique()))
        rec = view[view["month"] == month].iloc[0]
        contrib = rec[shap_cols].astype(float)
        contrib.index = [c.replace("shap_", "") for c in contrib.index]
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        c1, c2 = st.columns(2)
        c1.metric("Observed UHI", f"{rec['uhi']:.2f} K")
        c2.metric("Predicted UHI", f"{rec['predicted_uhi']:.2f} K")
        wf = px.bar(x=contrib.values, y=contrib.index, orientation="h", labels={"x": "Push on predicted UHI (K)", "y": ""}, height=380)
        wf.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(wf, width="stretch")

    st.divider()
    st.subheader("Can we trust these explanations?")
    fid = data.report("shap_fidelity")
    d = fid["deletion"]
    rows = []
    for k in ("1", "3", "5", "10"):
        rows += [
            {"Features removed": int(k), "Which features": "Top ranked by SHAP", "RMSE (K)": d[k]["top_k_rmse"]},
            {"Features removed": int(k), "Which features": "Random", "RMSE (K)": d[k]["random_k_rmse"]},
            {"Features removed": int(k), "Which features": "Lowest ranked", "RMSE (K)": d[k]["bottom_k_rmse"]},
        ]
    dele = px.bar(pd.DataFrame(rows), x="Features removed", y="RMSE (K)", color="Which features", barmode="group", height=340,
                  title="Deletion test: removing SHAP's top features hurts the model most")
    dele.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(dele, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Additivity error", f"{fid['additivity']['max_abs_err']:.0e} K")
    c2.metric("Agreement with permutation importance", f"{fid['perm_vs_shap']['spearman']:.2f}")
    c3.metric("Stability 2024 vs 2025", f"{fid['stability_2024_vs_2025']['spearman']:.2f}")
    c4.metric("Stability India vs international", f"{fid['stability_india_vs_intl']['spearman']:.2f}")
    ab = fid["ablation"]
    abl = pd.DataFrame(
        [{"Retrained without": k.replace("no_", "").replace("_", " + ") if k != "all" else "nothing (full model)", "Features": v["n_feat"], "Test RMSE (K)": v["test_rmse"], "Test R2": v["test_r2"]} for k, v in ab.items()]
    )
    st.dataframe(abl.round(3), hide_index=True, width="stretch")
    st.info(
        "SHAP describes the fitted model faithfully: it adds up exactly, removing top ranked features hurts most, and the ranking is stable. "
        "But features are correlated. Removing all weather features does not make the model worse, although weather holds the biggest SHAP share. "
        "So SHAP shows how this model used its inputs, not how much each factor matters in the real world."
    )
