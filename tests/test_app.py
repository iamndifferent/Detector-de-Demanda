"""Prueba de humo: la aplicación arranca con los datos de ejemplo."""

from pathlib import Path

import pytest

# Si Streamlit no está instalado, omitir esta prueba sin impedir el resto de tests.
streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_app_runs_with_sample_data() -> None:
    """Comprueba que la app principal arranca sin excepciones."""
    app_test = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app_test.exception
    assert app_test.title[0].value.endswith("Predictor de demanda")
