import numpy as np
import pandas as pd
import pytest

from forecaster.cleaning import clean_sales
from forecaster.config import FREQUENCIES
from forecaster.models import forecast_series
from forecaster.readiness import assess_data
from forecaster.replenishment import suggest_order
from forecaster.summary import (
    ProductResult,
    build_summary,
    expected_change,
    recent_trend,
    trend_short,
    trend_text,
)

WEEKLY = FREQUENCIES["Semanal"]


def _result(name, values, stock=0.0, horizon=4, covered=2):
    index = pd.period_range("2024-01-01", periods=len(values), freq="W-SUN")
    history = pd.Series(values, index=index, dtype=float)
    fc = forecast_series(history.to_numpy(), horizon, WEEKLY.season, WEEKLY.window)
    future = pd.period_range(start=index[-1] + 1, periods=horizon)
    suggestion = suggest_order(fc.mean, fc.sigma, covered, 0.95, stock)
    return ProductResult(name, history, fc, suggestion, future)


def _report():
    df = pd.DataFrame([["2024-01-01", "A", "1"], ["2024-06-30", "A", "1"]],
                      columns=["fecha", "producto", "cantidad"])  # fmt: skip
    return clean_sales(df, "fecha", "producto", "cantidad").report


def _summary(results, stock_provided=False, lead_days=7):
    series = {r.name: r.history for r in results}
    return build_summary(
        _report(), assess_data(series, WEEKLY), WEEKLY, results,
        horizon=4, lead_days=lead_days, extra_days=7, service_level=0.95,
        stock_provided=stock_provided,
    )  # fmt: skip


def test_trend_detection_and_labels():
    rising = np.array([10, 10, 10, 10, 20, 20, 20, 20.0])
    assert recent_trend(rising, 4) == 1.0
    assert trend_text(1.0) == "sube un 100%" and trend_short(1.0) == "↑ 100%"
    assert trend_text(-0.25) == "baja un 25%" and trend_short(-0.25) == "↓ 25%"
    assert trend_text(0.05) == "se mantiene estable" and trend_short(0.05) == "→ estable"
    assert np.isnan(recent_trend(np.array([1.0, 2.0, 3.0]), 4))  # historial insuficiente
    assert "no tiene datos" in trend_text(float("nan"))


def test_expected_change_compares_with_the_same_length_window():
    history = np.array([10.0] * 8)
    assert expected_change(history, np.full(4, 12.0), 4) == pytest.approx(0.2)
    assert np.isnan(expected_change(np.zeros(8), np.full(4, 5.0), 4))


def test_summary_explains_each_point_and_lists_actions():
    rng = np.random.default_rng(1)
    results = [
        _result("Croissant", 190 + rng.normal(0, 8, 40)),
        _result("Té verde", np.linspace(30, 12, 40) + rng.normal(0, 1, 40)),
    ]
    summary = _summary(results)
    titles = [p.title for p in summary.points]
    assert titles == ["Tus datos", "Cómo van tus ventas", "Qué prevemos", "Cuánto pedir",
                      "Qué tan fiable es", "Lo que la app no sabe"]  # fmt: skip
    text = " ".join(p.text for p in summary.points)
    assert "Croissant" in text and "Té verde baja" in text
    assert "no descuenta tu stock" in text  # no se indicó stock
    actions = " ".join(summary.actions)
    assert "Escribe tu stock actual" in actions
    assert "Descarga la tabla de compras" in actions
    assert "cada semana" in actions


def test_urgent_products_are_flagged_when_stock_is_below_lead_time_demand():
    rng = np.random.default_rng(2)
    results = [_result("Pan", 100 + rng.normal(0, 3, 40), stock=20.0)]
    summary = _summary(results, stock_provided=True)
    assert any(a.startswith("Prioridad: Pan") for a in summary.actions)
    assert not any("Escribe tu stock" in a for a in summary.actions)


def test_enough_stock_means_no_order_needed():
    rng = np.random.default_rng(3)
    results = [_result("Pan", 100 + rng.normal(0, 3, 40), stock=5000.0)]
    summary = _summary(results, stock_provided=True)
    assert any("no hace falta pedir" in p.text for p in summary.points)


def test_short_history_gets_a_cautious_headline_and_advice():
    results = [_result("Nuevo", [3.0, 4, 5, 4, 6])]
    summary = _summary(results)
    assert "pocos" in summary.headline
    assert any("más historial" in a for a in summary.actions)


def test_product_names_cannot_inject_markdown():
    rng = np.random.default_rng(4)
    results = [_result("[gratis](http://x.com)", 50 + rng.normal(0, 2, 40))]
    text = " ".join(p.text for p in _summary(results).points)
    assert "\\[gratis\\]" in text and "[gratis](" not in text
