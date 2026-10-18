import numpy as np

from forecaster.formatting import fmt, md_escape, pct, signed_pct


def test_spanish_number_format():
    assert fmt(1234.5, 1) == "1.234,5"
    assert fmt(1234567) == "1.234.567"
    assert fmt(0.5, 2) == "0,50"


def test_percentages_handle_nan():
    assert pct(0.054) == "5%"
    assert pct(float("nan")) == "—"
    assert signed_pct(0.08) == "+8%" and signed_pct(-0.05) == "-5%"
    assert signed_pct(-0.001) == "0%" and signed_pct(0.003) == "0%"
    assert signed_pct(np.nan) == "—"


def test_markdown_in_product_names_is_neutralised():
    text = md_escape("[clic](http://x.com) *negrita* $5 Café (kg)")
    assert "\\[clic\\](http://x.com)" in text
    assert "\\*negrita\\*" in text and "\\$5" in text and "Café" in text
