import numpy as np

from forecaster.evaluation import rolling_origin_wape, wape


def test_wape_basic_and_zero_denominator():
    assert wape(np.array([10.0, 10.0]), np.array([8.0, 13.0])) == 0.25
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_rolling_origin_is_zero_for_a_perfect_model():
    y = np.full(20, 5.0)
    perfect = lambda train, h: (np.full(h, 5.0), np.full(len(train), 5.0))  # noqa: E731
    assert rolling_origin_wape(y, perfect, 3, 3) == 0.0


def test_rolling_origin_is_nan_when_model_cannot_fit():
    assert np.isnan(rolling_origin_wape(np.arange(20.0), lambda train, h: None, 3, 2))
