"""Métricas de error y validación contra el pasado reciente."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Un modelo recibe (historial, horizonte) y devuelve (previsión, valores ajustados) o None.
ModelFn = Callable[[np.ndarray, int], "tuple[np.ndarray, np.ndarray] | None"]


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Error porcentual ponderado: suma de errores absolutos / suma de ventas reales.

    Es más robusto que el MAPE cuando hay periodos con 0 ventas. Devuelve NaN si no hubo ventas.
    """
    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def rolling_origin_wape(y: np.ndarray, model: ModelFn, horizon: int, folds: int) -> float:
    """Valida un modelo "viajando al pasado": entrena con los datos hasta un punto, predice
    los siguientes `horizon` periodos y compara con lo que realmente se vendió. Repite
    `folds` veces desplazando el punto y devuelve el WAPE conjunto.
    """
    n = len(y)
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for k in range(folds, 0, -1):
        split = n - k * horizon
        result = model(y[:split], horizon)
        if result is None:
            return float("nan")
        actual.append(y[split : split + horizon])
        predicted.append(np.clip(result[0], 0, None))
    return wape(np.concatenate(actual), np.concatenate(predicted))
