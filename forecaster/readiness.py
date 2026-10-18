"""Comprobaciones sobre el archivo subido: ¿hay datos suficientes y de buena calidad?"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import CleaningReport
from .config import MIN_POINTS, Frequency
from .formatting import fmt


@dataclass(frozen=True)
class DataAssessment:
    level: str  # "Insuficiente", "Justo" o "Suficiente"
    longest: int  # periodos del producto con más historial
    products: int
    products_usable: int  # productos con historial para usar suavizado (>= MIN_POINTS)
    seasonality: bool  # ¿hay al menos dos ciclos completos para detectar temporadas?
    periods_missing: int  # periodos que faltan para llegar al historial recomendado


@dataclass(frozen=True)
class Check:
    status: str  # "ok", "info", "warn" o "error"
    text: str


def assess_data(series: dict[str, pd.Series], freq: Frequency) -> DataAssessment:
    if not series:
        raise ValueError("No hay series que evaluar.")
    lengths = [len(s) for s in series.values()]
    longest = max(lengths)
    if longest < MIN_POINTS:
        level = "Insuficiente"
    elif longest < freq.recommended:
        level = "Justo"
    else:
        level = "Suficiente"
    return DataAssessment(
        level=level,
        longest=longest,
        products=len(lengths),
        products_usable=sum(n >= MIN_POINTS for n in lengths),
        seasonality=longest >= 2 * freq.season,
        periods_missing=max(0, freq.recommended - longest),
    )


def file_checks(report: CleaningReport, assessment: DataAssessment, freq: Frequency) -> list[Check]:
    """Lista de comprobaciones para mostrar al usuario justo después de subir el archivo."""
    plural = freq.plural
    checks = [
        Check(
            "ok",
            f"Archivo leído: {fmt(report.rows_in)} filas y {report.products} productos, "
            f"del {report.date_min:%d/%m/%Y} al {report.date_max:%d/%m/%Y}.",
        )
    ]

    valid_share = report.rows_used / report.rows_in if report.rows_in else 0.0
    if report.duplicates_removed:
        valid_share = (report.rows_used + report.duplicates) / report.rows_in
    if valid_share >= 0.98:
        checks.append(Check("ok", "Formato correcto: casi todas las filas son válidas."))
    elif valid_share >= 0.8:
        checks.append(
            Check(
                "warn",
                f"Solo el {valid_share:.0%} de las filas es válido: revisa fechas y cantidades.",
            )
        )
    else:
        checks.append(
            Check(
                "error",
                f"Solo el {valid_share:.0%} de las filas es válido: comprueba que elegiste "
                "bien las columnas y el formato de fecha.",
            )
        )

    if assessment.level == "Insuficiente":
        checks.append(
            Check(
                "error",
                f"Historial insuficiente: el producto con más datos tiene {assessment.longest} "
                f"{plural} y hacen falta al menos {MIN_POINTS}. Prueba una agrupación más fina "
                "o sube un archivo con más fechas.",
            )
        )
    elif assessment.level == "Justo":
        checks.append(
            Check(
                "warn",
                f"Historial justo: {assessment.longest} {plural}. Con {freq.recommended} o más "
                f"la previsión sería más fiable (faltan {assessment.periods_missing}).",
            )
        )
    else:
        checks.append(
            Check(
                "ok",
                f"Historial suficiente: {assessment.longest} {plural} "
                f"(recomendado: {freq.recommended}).",
            )
        )

    if assessment.products_usable < assessment.products:
        low = assessment.products - assessment.products_usable
        checks.append(
            Check(
                "warn",
                f"{low} de {assessment.products} productos tienen menos de {MIN_POINTS} {plural} "
                "de historial: su previsión es solo orientativa.",
            )
        )
    else:
        checks.append(Check("ok", "Todos los productos tienen historial para poder prever."))

    if assessment.seasonality:
        checks.append(Check("ok", "Hay historial para detectar temporadas (picos repetidos)."))
    else:
        checks.append(
            Check(
                "info",
                f"Aún no hay historial para detectar temporadas: harían falta {2 * freq.season} "
                f"{plural}.",
            )
        )
    return checks
