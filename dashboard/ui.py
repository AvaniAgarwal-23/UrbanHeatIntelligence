"""Small shared UI helpers."""
import streamlit as st

# Viridis based palette, colour blind safe like the React dashboard.
SCALE = "Viridis"


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def estimate_note() -> None:
    st.info(
        "Model based estimate for academic exploration. The results show associations "
        "learned from data, not proven causes, and are not a heat warning system."
    )
