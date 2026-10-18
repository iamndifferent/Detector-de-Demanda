"""Limpieza de ventas y conversión a series temporales por producto."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MAX_PRODUCTS, Frequency
from .loader import DataError


@dataclass
class CleaningReport:
    """Resumen de lo que se encontró y se corrigió, para mostrarlo al usuario."""

    rows_in: int
    rows_used: int
    invalid_dates: int
    invalid_products: int
    invalid_quantities: int
    negative_quantities: int
    duplicates: int
    duplicates_removed: bool
    products: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    messages: list[str] = field(default_factory=list)


@dataclass
class CleaningResult:
    data: pd.DataFrame  # columnas: date, product, qty
    report: CleaningReport


def parse_quantity(values: pd.Series) -> pd.Series:
    """Convierte texto a número. Si hay coma, se asume formato europeo ("1.234,5")."""
    text = values.fillna("").astype(str).str.strip()
    has_comma = text.str.contains(",", regex=False)
    european = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text.where(~has_comma, european), errors="coerce")


def _parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    """Interpreta fechas de formato mixto.

    Las fechas ISO (2024-03-05) se leen siempre como año-mes-día: aplicarles "día primero"
    las dejaría con mes y día intercambiados sin avisar. El resto (05/03/2024) sigue la
    preferencia del usuario.
    """
    text = values.fillna("").astype(str).str.strip()
    is_iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    try:
        iso = pd.to_datetime(text.where(is_iso), errors="coerce", format="ISO8601", utc=True)
        other = pd.to_datetime(
            text.where(~is_iso), errors="coerce", dayfirst=dayfirst, format="mixed", utc=True
        )
    except (ValueError, TypeError) as exc:
        raise DataError(f"No se pudieron interpretar las fechas: {exc}") from exc
    return iso.fillna(other).dt.tz_convert(None)


def clean_sales(
    df: pd.DataFrame,
    date_col: str,
    product_col: str,
    qty_col: str,
    *,
    dayfirst: bool = True,
    drop_duplicates: bool = False,
) -> CleaningResult:
    """Valida y normaliza una tabla de ventas con una fila por venta (o por día y producto)."""
    chosen = (date_col, product_col, qty_col)
    for col in chosen:
        if col not in df.columns:
            raise DataError(f"No existe la columna «{col}» en el archivo.")
    if len(set(chosen)) < 3:
        raise DataError("Elige tres columnas distintas para fecha, producto y cantidad.")

    rows_in = len(df)
    dates = _parse_dates(df[date_col], dayfirst)
    products = df[product_col].fillna("").astype(str).str.strip()
    qty = parse_quantity(df[qty_col])

    bad_date, bad_product, bad_qty = dates.isna(), products.eq(""), qty.isna()
    valid = ~(bad_date | bad_product | bad_qty)
    data = pd.DataFrame(
        {"date": dates[valid].dt.normalize(), "product": products[valid], "qty": qty[valid]}
    ).reset_index(drop=True)
    if data.empty:
        raise DataError(
            "No quedó ninguna fila válida. Revisa las columnas elegidas y el formato de fecha."
        )

    negatives = int((data["qty"] < 0).sum())
    data["qty"] = data["qty"].clip(lower=0.0)  # las devoluciones no se descuentan: se ponen a 0
    duplicates = int(data.duplicated().sum())
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)

    n_products = int(data["product"].nunique())
    if n_products > MAX_PRODUCTS:
        raise DataError(f"Hay {n_products:,} productos; el máximo es {MAX_PRODUCTS:,}.")

    messages: list[str] = []
    if int(bad_date.sum()) > 0.5 * rows_in:
        messages.append(
            "Más de la mitad de las fechas no se pudieron leer. "
            "Comprueba la columna elegida y el formato día/mes/año."
        )
    if negatives:
        messages.append(f"{negatives} cantidades negativas (devoluciones) se trataron como 0.")
    if duplicates and not drop_duplicates:
        messages.append(
            f"Hay {duplicates} filas idénticas. Se mantienen porque pueden ser ventas reales; "
            "puedes eliminarlas en las opciones."
        )

    report = CleaningReport(
        rows_in=rows_in,
        rows_used=len(data),
        invalid_dates=int(bad_date.sum()),
        invalid_products=int(bad_product.sum()),
        invalid_quantities=int(bad_qty.sum()),
        negative_quantities=negatives,
        duplicates=duplicates,
        duplicates_removed=drop_duplicates,
        products=n_products,
        date_min=data["date"].min(),
        date_max=data["date"].max(),
        messages=messages,
    )
    return CleaningResult(data=data, report=report)


def build_series(
    data: pd.DataFrame, freq: Frequency, drop_incomplete: bool = True
) -> tuple[dict[str, pd.Series], bool]:
    """Agrupa las ventas por periodo y devuelve una serie por producto.

    - Los periodos sin ventas se rellenan con 0.
    - Cada producto empieza en su primera venta (así un lanzamiento reciente no arrastra ceros).
    - El último periodo se descarta si aún no ha terminado (p. ej. una semana a medias).
    """
    periods = data["date"].dt.to_period(freq.code)
    full = pd.period_range(periods.min(), periods.max(), freq=freq.code)
    dropped = False
    if drop_incomplete and len(full) > 1 and full[-1].end_time.normalize() > data["date"].max():
        full = full[:-1]
        dropped = True

    table = (
        data.assign(period=periods)
        .groupby(["period", "product"])["qty"]
        .sum()
        .unstack("product")
        .reindex(full)
        .fillna(0.0)
    )
    series: dict[str, pd.Series] = {}
    for product in table.columns:
        column = table[product]
        sold = np.flatnonzero(column.to_numpy() > 0)
        if sold.size:
            series[str(product)] = column.iloc[sold[0] :]
    return series, dropped


def cap_outliers(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Limita los picos extremos (Q3 + 3·IQR). No toca series con demanda intermitente."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return values, 0
    upper = q3 + 3 * iqr
    return np.minimum(values, upper), int((values > upper).sum())
