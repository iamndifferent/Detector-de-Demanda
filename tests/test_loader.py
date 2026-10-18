import pytest

from forecaster.config import MAX_FILE_BYTES
from forecaster.loader import DataError, guess_columns, read_csv


def test_reads_semicolon_latin1_file():
    raw = "fecha;producto;cantidad\n01/02/2024;Café;3,5\n02/02/2024;Té;2\n".encode("latin-1")
    df = read_csv(raw)
    assert df.shape == (2, 3)
    assert df.loc[0, "producto"] == "Café"
    assert df.loc[0, "cantidad"] == "3,5"  # se lee como texto; la limpieza lo convierte


def test_rejects_empty_file():
    with pytest.raises(DataError):
        read_csv(b"   \n")


def test_rejects_oversized_file():
    with pytest.raises(DataError):
        read_csv(b"a,b,c\n" + b"1,2,3\n" * (MAX_FILE_BYTES // 6 + 10))


def test_rejects_too_few_columns():
    with pytest.raises(DataError):
        read_csv(b"fecha,cantidad\n2024-01-01,3\n")


def test_guesses_columns_in_spanish_and_english():
    assert guess_columns(["Fecha", "Producto", "Cantidad"]) == {
        "date": "Fecha", "product": "Producto", "quantity": "Cantidad",
    }  # fmt: skip
    assert guess_columns(["sku", "units", "date"]) == {
        "date": "date", "product": "sku", "quantity": "units",
    }  # fmt: skip


def test_guess_falls_back_to_column_order():
    assert guess_columns(["a", "b", "c"]) == {"date": "a", "product": "b", "quantity": "c"}
