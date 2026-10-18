import numpy as np

from forecaster.smoothing import holt_damped, holt_winters, ses


def test_ses_on_constant_series_forecasts_the_constant():
    mean, fitted = ses(np.full(30, 12.0), 5)
    assert np.allclose(mean, 12.0)
    assert len(fitted) == 30 and np.isnan(fitted[0])


def test_holt_follows_a_linear_trend():
    y = 10 + 2 * np.arange(40.0)
    mean, _ = holt_damped(y, 3)
    assert abs(mean[0] - 90.0) / 90.0 < 0.05
    assert mean[0] < mean[1] < mean[2]


def test_holt_winters_reproduces_a_weekly_pattern():
    pattern = np.array([10, -5, 0, 8, -3, -6, -4.0])
    y = 100 + np.tile(pattern, 10)
    mean, _ = holt_winters(y, 7, 7)
    assert np.allclose(mean, 100 + pattern, atol=1.0)


def test_holt_winters_needs_two_full_cycles():
    assert holt_winters(np.arange(13.0), 3, 7) is None
