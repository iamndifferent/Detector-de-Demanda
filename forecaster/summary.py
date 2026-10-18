"""Resumen en lenguaje llano: qué se analizó y qué debería hacer el cliente a continuación."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cleaning import CleaningReport
from .config import MIN_POINTS, Frequency
from .formatting import fmt, md_escape, pct, signed_pct
from .models import Forecast
from .readiness import DataAssessment
from .replenishment import OrderSuggestion, cover_periods_for

TREND_THRESHOLD = 0.10  # cambios de ±10 % o menos se consideran "estable"
REFRESH = {"Diaria": "cada semana", "Semanal": "cada semana", "Mensual": "cada mes"}


@dataclass
class ProductResult:
    """Todo lo calculado para un producto."""

    name: str
    history: pd.Series
    forecast: Forecast
    suggestion: OrderSuggestion
    future: pd.PeriodIndex


@dataclass(frozen=True)
class Point:
    title: str
    text: str


@dataclass(frozen=True)
class BusinessSummary:
    headline: str
    points: list[Point]
    actions: list[str]


def recent_trend(values: np.ndarray, window: int) -> float:
    """Cambio relativo entre los últimos `window` periodos y los `window` anteriores."""
    if len(values) < 2 * window:
        return float("nan")
    previous = values[-2 * window : -window].mean()
    if previous == 0:
        return float("nan")
    return float(values[-window:].mean() / previous - 1)


def trend_text(change: float) -> str:
    if not np.isfinite(change):
        return "no tiene datos suficientes para ver su tendencia"
    if change > TREND_THRESHOLD:
        return f"sube un {pct(change)}"
    if change < -TREND_THRESHOLD:
        return f"baja un {pct(-change)}"
    return "se mantiene estable"


def trend_short(change: float) -> str:
    """Versión corta de la tendencia para celdas de tabla."""
    if not np.isfinite(change):
        return "—"
    if change > TREND_THRESHOLD:
        return f"↑ {pct(change)}"
    if change < -TREND_THRESHOLD:
        return f"↓ {pct(-change)}"
    return "→ estable"


def expected_change(history: np.ndarray, mean: np.ndarray, horizon: int) -> float:
    """Previsión de los próximos `horizon` periodos frente a los `horizon` últimos reales."""
    h = min(horizon, len(mean), len(history))
    recent = history[-h:].sum()
    if recent == 0:
        return float("nan")
    return float(mean[:h].sum() / recent - 1)


def build_summary(
    report: CleaningReport,
    assessment: DataAssessment,
    freq: Frequency,
    results: list[ProductResult],
    *,
    horizon: int,
    lead_days: float,
    extra_days: float,
    service_level: float,
    stock_provided: bool,
) -> BusinessSummary:
    if not results:
        raise ValueError("No hay resultados que resumir.")
    plural = freq.plural
    n = len(results)
    reliable = [r for r in results if r.forecast.reliability == "Alta"]
    cautious = [r for r in results if r.forecast.reliability != "Alta"]
    to_order = sorted(
        (r for r in results if r.suggestion.order_qty > 0),
        key=lambda r: r.suggestion.order_qty,
        reverse=True,
    )
    order_list = ", ".join(
        f"{md_escape(r.name)} ({fmt(r.suggestion.order_qty)})" for r in to_order[:5]
    )
    if len(to_order) > 5:
        order_list += f" y {len(to_order) - 5} más (mira la tabla)"

    # --- Titular ---
    if assessment.level == "Insuficiente":
        headline = (
            "Tus datos todavía son pocos: tómate las cifras como una orientación, "
            "no como una previsión firme."
        )
    elif len(reliable) == n:
        headline = (
            "Buenas noticias: la previsión del producto analizado es fiable."
            if n == 1
            else f"Buenas noticias: las previsiones de los {n} productos analizados son fiables."
        )
    else:
        headline = (
            f"Analizamos {n} productos: {len(reliable)} con previsión fiable y "
            f"{len(cautious)} para mirar con cautela."
        )

    # --- Puntos analizados ---
    level_text = {
        "Insuficiente": (
            f"Es un historial corto: el producto con más datos tiene {assessment.longest} "
            f"{plural} y hacen falta al menos {MIN_POINTS}. Las cifras son solo orientativas."
        ),
        "Justo": (
            f"Alcanza para una previsión básica ({assessment.longest} {plural}); con "
            f"{freq.recommended} o más sería más fiable."
        ),
        "Suficiente": f"Es un historial suficiente ({assessment.longest} {plural})."
        + (" Incluso permite detectar temporadas." if assessment.seasonality else ""),
    }[assessment.level]
    points = [
        Point(
            "Tus datos",
            f"Leímos {fmt(report.rows_used)} ventas de {report.products} productos entre "
            f"el {report.date_min:%d/%m/%Y} y el {report.date_max:%d/%m/%Y}. {level_text}",
        )
    ]

    trends = [
        f"{md_escape(r.name)} {trend_text(recent_trend(r.history.to_numpy(), freq.window))}"
        for r in results[:5]
    ]
    points.append(
        Point(
            "Cómo van tus ventas",
            f"Comparando el periodo reciente ({freq.window} {plural}) con el anterior: "
            + "; ".join(trends)
            + ".",
        )
    )

    h = min(horizon, len(results[0].forecast.mean))
    outlook = []
    for r in results[:3]:
        total = r.forecast.mean[:h].sum()
        change = expected_change(r.history.to_numpy(), r.forecast.mean, horizon)
        extra = f", {signed_pct(change)} frente al periodo anterior"
        outlook.append(
            f"{md_escape(r.name)}: unas {fmt(total)} en el periodo previsto de {h} {plural}"
            + (extra if np.isfinite(change) else "")
        )
    points.append(
        Point("Qué prevemos", "; ".join(outlook) + ". El gráfico muestra el rango probable.")
    )

    if to_order:
        order_text = (
            f"Para cubrir {fmt(lead_days)} días de entrega más {fmt(extra_days)} de margen, con "
            f"un nivel de servicio del {service_level:.0%}, te sugerimos pedir: {order_list}."
        )
        if not stock_provided:
            order_text += " Esto no descuenta tu stock actual porque no lo indicaste."
    else:
        order_text = "Con el stock que indicaste no hace falta pedir ahora ningún producto."
    points.append(Point("Cuánto pedir", order_text))

    reliability_text = []
    if reliable:
        names = ", ".join(md_escape(r.name) for r in reliable)
        reliability_text.append(f"Puedes fiarte más de: {names}.")
    if cautious:
        reliability_text.append(
            f"Con cautela: {', '.join(md_escape(r.name) for r in cautious)} "
            "(poco historial, ventas irregulares o error de validación alto)."
        )
    points.append(Point("Qué tan fiable es", " ".join(reliability_text)))
    points.append(
        Point(
            "Lo que la app no sabe",
            "No conoce promociones, cambios de precio ni festivos futuros. Si alguna vez te "
            "quedaste sin stock, tus ventas pasadas son menores que la demanda real.",
        )
    )

    # --- Qué hacer ahora ---
    lead_periods = cover_periods_for(lead_days, 0, freq.days)
    urgent = [
        md_escape(r.name)
        for r in results
        if 0 < r.suggestion.on_hand < r.forecast.mean[:lead_periods].sum()
    ]
    actions: list[str] = []
    if to_order:
        actions.append(
            f"Revisa el pedido sugerido: {order_list}. Ajústalo con tu stock real antes de "
            "comprar."
        )
    if urgent:
        actions.append(
            f"Prioridad: {', '.join(urgent)}. Tu stock actual no llega para el plazo de entrega "
            f"({fmt(lead_days)} días); pide primero estos."
        )
    if not stock_provided:
        actions.append(
            "Escribe tu stock actual en la tabla «Stock actual» para que las cantidades "
            "descuenten lo que ya tienes."
        )
    if cautious:
        actions.append(
            f"No decidas solo con el número en {', '.join(md_escape(r.name) for r in cautious)}: "
            "compáralo con tu experiencia y pide cantidades prudentes."
        )
    if assessment.level == "Insuficiente":
        actions.append(
            f"Sube un archivo con más historial: necesitas al menos {MIN_POINTS} {plural} "
            f"(mejor {freq.recommended}) o prueba una agrupación más fina."
        )
    elif assessment.level == "Justo":
        actions.append(
            f"Sigue registrando tus ventas: con {assessment.periods_missing} {plural} más "
            f"llegarás a {freq.recommended} y las previsiones serán más fiables."
        )
    elif not assessment.seasonality:
        actions.append(
            f"Cuando reúnas {2 * freq.season} {plural} de datos, la app podrá detectar "
            "temporadas (picos de verano, Navidad…)."
        )
    actions.append(f"Vuelve a subir tus ventas {REFRESH[freq.name]} para actualizar las cifras.")
    actions.append("Descarga la tabla de compras (CSV) y úsala de base para tu pedido.")
    return BusinessSummary(headline=headline, points=points, actions=actions)
