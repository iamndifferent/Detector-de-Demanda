import numpy as np
import pandas as pd
import pytest

from forecaster.cleaning import build_series, cap_outliers, clean_sales, parse_quantity
from forecaster.config import FREQUENCIES
from forecaster.loader import DataError


def _raw(rows):
    return pd.DataFrame(rows, columns=["fecha", "producto", "cantidad"])


def test_parse_quantity_handles_european_format():
    out = parse_quantity(pd.Series(["1.234,5", "12,5", "7", "abc", None]))
    assert out.iloc[:3].tolist() == [1234.5, 12.5, 7.0]
    assert out.iloc[3:].isna().all()


def test_clean_sales_reports_and_fixes_problems():
    df = _raw([
        ["01/03/2024", "A", "5"],
        ["fecha rara", "A", "3"],
        ["02/03/2024", "", "4"],
        ["03/03/2024", "A", "x"],
        ["04/03/2024", "A", "-2"],
    ])  # fmt: skip
    result = clean_sales(df, "fecha", "producto", "cantidad")
    rep = result.report
    assert (rep.rows_in, rep.rows_used) == (5, 2)
    assert (rep.invalid_dates, rep.invalid_products, rep.invalid_quantities) == (1, 1, 1)
    assert rep.negative_quantities == 1
    assert result.data["qty"].tolist() == [5.0, 0.0]  # el -2 pasa a 0
    assert result.data["date"].iloc[0] == pd.Timestamp("2024-03-01")  # día/mes/año


def test_dayfirst_can_be_disabled():
    df = _raw([["03/04/2024", "A", "1"]])
    assert clean_sales(df, "fecha", "producto", "cantidad", dayfirst=False).data[
        "date"
    ].iloc[0] == pd.Timestamp("2024-03-04")


def test_iso_dates_are_never_swapped_by_dayfirst():
    df = _raw([["2024-01-08", "A", "1"], ["08/01/2024", "A", "1"], ["2024-12-31", "A", "1"]])
    for dayfirst in (True, False):
        dates = clean_sales(df, "fecha", "producto", "cantidad", dayfirst=dayfirst).data["date"]
        assert dates.iloc[0] == pd.Timestamp("2024-01-08")
        assert dates.iloc[2] == pd.Timestamp("2024-12-31")
    day_first = clean_sales(df, "fecha", "producto", "cantidad", dayfirst=True).data["date"]
    assert day_first.iloc[1] == pd.Timestamp("2024-01-08")  # 08/01 = 8 de enero


def test_duplicates_are_kept_unless_requested():
    df = _raw([["2024-03-01", "A", "5"], ["2024-03-01", "A", "5"]])
    kept = clean_sales(df, "fecha", "producto", "cantidad")
    removed = clean_sales(df, "fecha", "producto", "cantidad", drop_duplicates=True)
    assert len(kept.data) == 2 and kept.report.duplicates == 1 and kept.report.messages
    assert len(removed.data) == 1


def test_no_valid_rows_raises():
    with pytest.raises(DataError):
        clean_sales(_raw([["nada", "A", "1"]]), "fecha", "producto", "cantidad")


def test_same_column_twice_raises():
    with pytest.raises(DataError):
        clean_sales(_raw([["2024-01-01", "A", "1"]]), "fecha", "fecha", "cantidad")


def test_build_series_fills_gaps_and_trims_leading_zeros():
    df = _raw([
        ["2024-01-01", "A", "5"],  # lunes de la semana 1
        ["2024-01-22", "A", "7"],  # semana 4 (las semanas 2 y 3 sin ventas)
        ["2024-01-15", "B", "2"],  # B empieza en la semana 3
        ["2024-01-28", "A", "1"],  # domingo: la semana 4 está completa
    ])  # fmt: skip
    data = clean_sales(df, "fecha", "producto", "cantidad").data
    series, dropped = build_series(data, FREQUENCIES["Semanal"])
    assert not dropped
    assert series["A"].tolist() == [5.0, 0.0, 0.0, 8.0]
    assert series["B"].tolist() == [2.0, 0.0]  # sin ceros iniciales


def test_build_series_drops_incomplete_last_week():
    df = _raw([["2024-01-01", "A", "5"], ["2024-01-08", "A", "6"], ["2024-01-10", "A", "9"]])
    data = clean_sales(df, "fecha", "producto", "cantidad").data  # el 10/01 es miércoles
    series, dropped = build_series(data, FREQUENCIES["Semanal"])
    assert dropped and series["A"].tolist() == [5.0]
    series_all, dropped_all = build_series(data, FREQUENCIES["Semanal"], drop_incomplete=False)
    assert not dropped_all and series_all["A"].tolist() == [5.0, 15.0]


def test_cap_outliers_limits_spikes_but_not_intermittent_series():
    spiky = np.array([10, 11, 9, 10, 12, 10, 500.0])
    capped, n = cap_outliers(spiky)
    assert n == 1 and capped.max() < 50
    intermittent = np.array([0, 0, 0, 0, 0, 0, 40.0])
    same, n0 = cap_outliers(intermittent)
    assert n0 == 0 and same.max() == 40
