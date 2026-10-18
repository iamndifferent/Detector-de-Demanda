import numpy as np
import pytest

from forecaster.replenishment import cover_periods_for, suggest_order


def test_order_quantity_example():
    sug = suggest_order(np.array([10.0, 10.0, 10.0]), sigma=2.0, cover_periods=2,
                        service_level=0.95, on_hand=5.0)  # fmt: skip
    assert sug.periods_covered == 2 and sug.demand == 20.0
    assert sug.safety_stock == pytest.approx(1.6449 * 2 * 2**0.5, rel=1e-3)
    assert sug.order_qty == 20  # 20 + 4,65 - 5 = 19,65 -> 20


def test_enough_stock_means_no_order():
    assert suggest_order(np.array([5.0, 5.0]), 1.0, 2, 0.95, on_hand=100).order_qty == 0


def test_cover_is_capped_by_forecast_length():
    assert suggest_order(np.array([4.0, 4.0]), 0.0, 10, 0.9).periods_covered == 2


def test_invalid_arguments_raise():
    for bad in (1.0, 0.3):
        with pytest.raises(ValueError):
            suggest_order(np.array([1.0]), 1.0, 1, bad)
    with pytest.raises(ValueError):
        suggest_order(np.array([1.0]), 1.0, 0, 0.9)
    with pytest.raises(ValueError):
        suggest_order(np.array([1.0]), 1.0, 1, 0.9, on_hand=-1)


def test_cover_periods_conversion():
    assert cover_periods_for(7, 7, 7) == 2
    assert cover_periods_for(0, 0, 7) == 1
    assert cover_periods_for(10, 0, 7) == 2
    assert cover_periods_for(30, 0, 30.44) == 1
