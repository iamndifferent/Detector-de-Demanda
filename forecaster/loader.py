"""Lectura segura de archivos CSV de ventas."""

from __future__ import annotations

import csv
import io

import pandas as pd

from .config import MAX_FILE_BYTES, MAX_ROWS


class DataError(ValueError):
    """Problema con los datos del usuario; el mensaje se muestra tal cual en pantalla."""


_DATE_HINTS = ("fecha", "date", "dia", "día", "day", "time")
_PRODUCT_HINTS = ("producto", "product", "sku", "item", "articulo", "artículo", "nombre", "name")
_QUANTITY_HINTS = ("cantidad", "quantity", "qty", "unidades", "units", "ventas", "sales", "vendido")


def read_csv(source: bytes | io.BufferedIOBase) -> pd.DataFrame:
    """Lee un CSV (cualquier separador habitual, UTF-8 o Latin-1) como texto sin interpretar."""
    raw = source if isinstance(source, bytes) else source.read()
    if len(raw) > MAX_FILE_BYTES:
        raise DataError(f"El archivo supera el máximo de {MAX_FILE_BYTES // 1_048_576} MB.")
    if not raw.strip():
        raise DataError("El archivo está vacío.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")  # nunca falla; cubre exportaciones antiguas de Excel

    try:
        sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","

    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True, nrows=MAX_ROWS + 1
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"No se pudo leer el CSV: {exc}") from exc

    if len(df) > MAX_ROWS:
        raise DataError(f"El archivo tiene más de {MAX_ROWS:,} filas; divídelo en partes.")
    if df.shape[1] < 3:
        raise DataError("Hacen falta al menos 3 columnas: fecha, producto y cantidad.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_columns(columns: list[str]) -> dict[str, str]:
    """Propone qué columna es la fecha, el producto y la cantidad según su nombre."""
    remaining = list(columns)
    guess: dict[str, str] = {}
    for key, hints in (
        ("date", _DATE_HINTS),
        ("product", _PRODUCT_HINTS),
        ("quantity", _QUANTITY_HINTS),
    ):
        match = next((c for c in remaining if any(h in c.lower() for h in hints)), None)
        if match is not None:
            guess[key] = match
            remaining.remove(match)
    for key in ("date", "product", "quantity"):
        if key not in guess:
            guess[key] = remaining.pop(0)
    return guess
