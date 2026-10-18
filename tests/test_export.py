import pandas as pd

from forecaster.export import sanitize_cell, to_csv_bytes


def test_formula_like_text_is_neutralised():
    assert sanitize_cell("=SUM(A1)") == "'=SUM(A1)"
    assert sanitize_cell("@cmd") == "'@cmd"
    assert sanitize_cell("Café") == "Café"
    assert sanitize_cell(-5.0) == -5.0  # los números negativos no son texto


def test_csv_is_excel_friendly_and_sanitised():
    df = pd.DataFrame({"Producto": ["=1+1", "Té"], "Pedir": [-3, 4]})
    text = to_csv_bytes(df).decode("utf-8-sig")
    assert text.splitlines()[0] == "Producto,Pedir"
    assert "'=1+1" in text and "Té" in text
