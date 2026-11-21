"""Cached data loaders shared by every Streamlit view.

The dashboard only reads files that the Python pipeline already produced
(Parquet tables, JSON reports and trained models). It never retrains a model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # src.utils.config reads config/config.yaml relative to the repo root

import pandas as pd
import streamlit as st
import yaml

MONTHLY = ROOT / "data" / "processed" / "monthly_level"
PIXEL = ROOT / "data" / "processed" / "pixel_level"
SPLITS = ROOT / "data" / "processed" / "splits"
ROLLING = ROOT / "data" / "processed" / "rolling_results"
REPORTS = ROOT / "reports"

MODEL_LABELS = {
    "Ridge_Regression": "Ridge",
    "Random_Forest": "Random Forest",
    "Hist_Gradient_Boosting": "HistGradientBoosting",
}


def grid_file_key(city: str) -> str:
    """'Los Angeles' -> 'losangeles' (file naming used for the grid Parquet files)."""
    return str(city).lower().replace(" ", "")


@st.cache_data(show_spinner=False)
def cities() -> dict:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["cities"]


def _add_region(df: pd.DataFrame) -> pd.DataFrame:
    reg = {c: v["region"] for c, v in cities().items()}
    df["region"] = df["city"].map(reg).fillna("Unknown")
    return df


@st.cache_data(show_spinner=False)
def monthly() -> pd.DataFrame:
    df = pd.read_parquet(MONTHLY / "uhi_monthly_clean.parquet")
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return _add_region(df)


@st.cache_data(show_spinner=False)
def risk() -> pd.DataFrame:
    return _add_region(pd.read_parquet(MONTHLY / "uhi_city_risk_scores.parquet"))


@st.cache_data(show_spinner=False)
def shap_values() -> pd.DataFrame:
    return _add_region(pd.read_parquet(MONTHLY / "uhi_shap_values_2025.parquet"))


@st.cache_data(show_spinner=False)
def test_features() -> pd.DataFrame:
    return pd.read_parquet(SPLITS / "test_2025.parquet")


@st.cache_data(show_spinner=False)
def grid(city: str) -> pd.DataFrame:
    return pd.read_parquet(PIXEL / f"uhi_grid_{grid_file_key(city)}.parquet")


@st.cache_data(show_spinner=False)
def report(name: str) -> dict:
    with open(REPORTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def rolling_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROLLING / f"{name}.csv")


@st.cache_resource(show_spinner="Loading the champion model...")
def simulator():
    from src.models.simulator import CounterfactualSimulator

    return CounterfactualSimulator()
