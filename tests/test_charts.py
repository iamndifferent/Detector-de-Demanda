import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from forecaster.charts import order_chart, overview_chart, product_chart  # noqa: E402
from forecaster.config import FREQUENCIES  # noqa: E402
from forecaster.models import forecast_series  # noqa: E402
from forecaster.replenishment import suggest_order  # noqa: E402
from forecaster.summary import ProductResult  # noqa: E402

WEEKLY = FREQUENCIES["Semanal"]


def _result(name, seed):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2024-01-01", periods=40, freq="W-SUN")
    history = pd.Series(100 + rng.normal(0, 5, 40), index=index)
    fc = forecast_series(history.to_numpy(), 4, WEEKLY.season, WEEKLY.window)
    future = pd.period_range(start=index[-1] + 1, periods=4)
    return ProductResult(name, history, fc, suggest_order(fc.mean, fc.sigma, 2, 0.95), future)


def test_product_chart_has_band_history_and_forecast():
    r = _result("A", 1)
    fig = product_chart(r.history, r.forecast, r.future, "semana")
    assert len(fig.data) == 4  # límite superior, límite inferior, ventas reales, previsión


def test_overview_chart_shows_at_most_four_products():
    results = [_result(f"P{i}", i) for i in range(6)]
    assert len(overview_chart(results, "semana").data) == 4 * 4
    assert len(overview_chart(results[:2], "semana").data) == 2 * 4


def test_order_chart_has_one_bar_series():
    fig = order_chart([_result("A", 1), _result("B", 2)])
    assert len(fig.data) == 1
