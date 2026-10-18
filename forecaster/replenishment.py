"""Sugerencia de compra: cuánto pedir para no quedarse sin stock."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class OrderSuggestion:
    periods_covered: int
    demand: float  # ventas previstas durante los periodos cubiertos
    safety_stock: float  # colchón para absorber la incertidumbre de la previsión
    on_hand: float
    order_qty: int


def cover_periods_for(lead_days: float, extra_days: float, days_per_period: float) -> int:
    """Convierte plazo de entrega + días extra de cobertura en número de periodos (mínimo 1)."""
    if lead_days < 0 or extra_days < 0:
        raise ValueError("Los días no pueden ser negativos.")
    return max(1, math.ceil(round((lead_days + extra_days) / days_per_period, 9)))


def suggest_order(
    mean: np.ndarray,
    sigma: float,
    cover_periods: int,
    service_level: float,
    on_hand: float = 0.0,
) -> OrderSuggestion:
    """Pedido = demanda prevista + stock de seguridad − stock actual (nunca negativo).

    Stock de seguridad = z · σ · √periodos, donde z sale del nivel de servicio deseado
    (p. ej. 95 % → z ≈ 1,64) y σ es el error típico de la previsión por periodo.
    """
    if not 0.5 <= service_level < 1:
        raise ValueError("El nivel de servicio debe estar entre 0,5 y 1 (sin incluir el 1).")
    if cover_periods < 1:
        raise ValueError("Hay que cubrir al menos 1 periodo.")
    if on_hand < 0:
        raise ValueError("El stock actual no puede ser negativo.")

    periods = min(cover_periods, len(mean))
    demand = float(np.sum(mean[:periods]))
    safety = NormalDist().inv_cdf(service_level) * sigma * math.sqrt(periods)
    needed = demand + safety - on_hand
    return OrderSuggestion(
        periods_covered=periods,
        demand=demand,
        safety_stock=safety,
        on_hand=on_hand,
        order_qty=math.ceil(round(max(0.0, needed), 6)),
    )
