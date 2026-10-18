import pandas as pd
import pytest

from forecaster.cleaning import clean_sales
from forecaster.config import FREQUENCIES
from forecaster.readiness import assess_data, file_checks

WEEKLY = FREQUENCIES["Semanal"]


def _series(*lengths):
    idx = lambda n: pd.period_range("2024-01-01", periods=n, freq="W-SUN")  # noqa: E731
    return {f"P{i}": pd.Series(range(1, n + 1), index=idx(n), dtype=float)
            for i, n in enumerate(lengths)}  # fmt: skip


def test_levels_follow_the_amount_of_history():
    assert assess_data(_series(5), WEEKLY).level == "Insuficiente"
    assert assess_data(_series(8), WEEKLY).level == "Justo"
    assert assess_data(_series(25), WEEKLY).level == "Justo"
    assert assess_data(_series(26), WEEKLY).level == "Suficiente"


def test_assessment_details():
    a = assess_data(_series(110, 6, 30), WEEKLY)
    assert a.longest == 110 and a.products == 3 and a.products_usable == 2
    assert a.seasonality and a.periods_missing == 0
    short = assess_data(_series(10), WEEKLY)
    assert not short.seasonality and short.periods_missing == 16


def test_empty_series_is_rejected():
    with pytest.raises(ValueError):
        assess_data({}, WEEKLY)


def _report(rows):
    df = pd.DataFrame(rows, columns=["fecha", "producto", "cantidad"])
    return clean_sales(df, "fecha", "producto", "cantidad").report


def test_checks_flag_short_history_as_error_and_good_history_as_ok():
    report = _report([["2024-01-01", "A", "1"], ["2024-01-08", "A", "2"]])
    bad = file_checks(report, assess_data(_series(4), WEEKLY), WEEKLY)
    assert any(c.status == "error" and "insuficiente" in c.text for c in bad)
    good = file_checks(report, assess_data(_series(120), WEEKLY), WEEKLY)
    assert not any(c.status == "error" for c in good)
    assert any("Historial suficiente" in c.text for c in good)
    assert any("detectar temporadas" in c.text and c.status == "ok" for c in good)


def test_checks_warn_when_many_rows_are_invalid():
    rows = [["2024-01-01", "A", "1"]] + [["no es fecha", "A", "1"]] * 4
    checks = file_checks(_report(rows), assess_data(_series(30), WEEKLY), WEEKLY)
    assert any(c.status == "error" and "20%" in c.text for c in checks)
