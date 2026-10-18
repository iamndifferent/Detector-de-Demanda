"""Exportación segura de resultados a CSV."""

from __future__ import annotations

import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(value):
    """Evita la inyección de fórmulas: Excel ejecuta celdas de texto que empiezan por = + - @."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV en UTF-8 con BOM (para que Excel respete las tildes) y celdas de texto saneadas."""
    safe = df.copy()
    for column in safe.columns:
        if safe[column].dtype == object or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(sanitize_cell)
    return safe.to_csv(index=False).encode("utf-8-sig")
