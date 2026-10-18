import numpy as np
import pytest

from forecaster.models import forecast_series


def _weekly_pattern(n, seed=0):
    rng = np.random.default_rng(seed)
    pattern = np.array([30, 10, 15, 20, 25, 60, 70.0])
    return np.maximum(0, np.tile(pattern, n // 7 + 1)[:n] + rng.normal(0, 2, n))


def test_seasonal_series_beats_the_baseline_clearly():
    fc = forecast_series(_weekly_pattern(280), horizon=14, season=7, window=7)
    assert fc.best_wape < 0.15
    assert fc.best_wape < fc.baseline_wape
    assert fc.improvement > 0.3
    assert fc.reliability == "Alta"


def test_forecast_shape_and_interval_ordering():
    fc = forecast_series(_weekly_pattern(120), horizon=10, season=7, window=7)
    assert len(fc.mean) == len(fc.lower) == len(fc.upper) == 10
    assert np.all(fc.lower <= fc.mean) and np.all(fc.mean <= fc.upper)
    assert np.all(fc.lower >= 0)


def test_short_history_still_returns_a_cautious_forecast():
    fc = forecast_series(np.array([3.0, 5, 4, 6, 5, 4]), horizon=4, season=52, window=4)
    assert np.all(fc.mean >= 0)
    assert any("orientativa" in w for w in fc.warnings)
    assert fc.reliability == "Baja"


def test_all_zero_series_does_not_crash():
    fc = forecast_series(np.zeros(30), horizon=5, season=7, window=7)
    assert np.allclose(fc.mean, 0) and fc.sigma == 0
    assert any("sin ventas" in w or "intermitente" in w for w in fc.warnings)


def test_constant_series_has_no_uncertainty():
    fc = forecast_series(np.full(40, 8.0), horizon=4, season=7, window=7)
    assert np.allclose(fc.mean, 8.0) and fc.sigma == pytest.approx(0.0, abs=1e-6)


def test_intermittent_demand_is_flagged():
    y = np.zeros(60)
    y[::9] = 5
    assert any("intermitente" in w for w in forecast_series(y, 4, 7, 7).warnings)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        forecast_series(np.array([1.0]), 3, 7, 7)
    with pytest.raises(ValueError):
        forecast_series(np.arange(20.0), 0, 7, 7)
