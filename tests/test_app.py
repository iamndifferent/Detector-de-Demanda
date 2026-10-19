"""Prueba de humo de la interfaz: la app arranca con los datos de ejemplo sin errores."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_app_runs_with_sample_data():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.endswith("Predictor de demanda")
