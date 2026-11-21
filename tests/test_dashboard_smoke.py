"""Smoke tests: every Streamlit view must render without raising an exception."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEWS = [
    "overview", "city_deep_dive", "spatial_grid", "ml_benchmarks",
    "rolling_window", "generalization", "explainability", "cooling_simulator",
]


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders(view):
    script = (
        "import sys, os\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        f"os.chdir(r'{ROOT}')\n"
        f"from dashboard.views import {view}\n"
        f"{view}.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
