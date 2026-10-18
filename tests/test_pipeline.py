import numpy as np

from forecaster.cleaning import build_series, clean_sales
from forecaster.config import FREQUENCIES
from forecaster.loader import guess_columns, read_csv
from forecaster.models import forecast_series
from forecaster.replenishment import cover_periods_for, suggest_order
from forecaster.samples import generate_sample_sales, sample_csv_bytes, template_csv_bytes


def test_sample_data_is_reproducible_and_well_formed():
    a, b = generate_sample_sales(seed=1), generate_sample_sales(seed=1)
    assert a.equals(b)
    assert list(a.columns) == ["fecha", "producto", "cantidad"]
    assert a["producto"].nunique() == 7 and (a["cantidad"] > 0).all()
    assert len(template_csv_bytes().splitlines()) == 13  # cabecera + 12 filas


def test_end_to_end_from_csv_to_purchase_suggestion():
    raw = read_csv(sample_csv_bytes())
    cols = guess_columns(list(raw.columns))
    cleaned = clean_sales(raw, cols["date"], cols["product"], cols["quantity"])
    assert cleaned.report.rows_used == cleaned.report.rows_in and not cleaned.report.messages

    freq = FREQUENCIES["Semanal"]
    series, dropped = build_series(cleaned.data, freq)
    assert dropped and len(series) == 7
    assert len(series["Vela aromática"]) < len(series["Pan de masa madre"])  # lanzamiento tardío

    history = series["Helado artesanal"].to_numpy()
    fc = forecast_series(history, 8, freq.season, freq.window)
    assert fc.best_wape < fc.baseline_wape  # el helado es muy estacional
    covered = cover_periods_for(14, 7, freq.days)
    sug = suggest_order(fc.mean, fc.sigma, covered, 0.95, on_hand=0)
    assert sug.order_qty > 0 and np.isfinite(sug.safety_stock)


def test_cafe_example_has_the_documented_errors():
    from forecaster.samples import generate_cafe_example

    raw = read_csv(generate_cafe_example())
    assert list(raw.columns) == ["fecha", "producto", "cantidad"]
    report = clean_sales(raw, "fecha", "producto", "cantidad").report
    assert (report.rows_in, report.rows_used, report.products) == (581, 578, 3)
    assert (report.invalid_dates, report.invalid_products, report.invalid_quantities) == (1, 1, 1)
    assert report.negative_quantities == 1
