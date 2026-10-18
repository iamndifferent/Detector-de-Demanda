"""Suavizado exponencial implementado con NumPy y SciPy (sin statsmodels).

Cada función recibe el historial `y` y el horizonte `h`, y devuelve una tupla
(previsión, valores ajustados), donde los ajustados son la predicción a un paso de cada
periodo histórico (NaN cuando no hay predicción posible). Devuelven None si no se puede ajustar.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_NAN = float("nan")


def _sse(values: list[float], fitted: list[float], start: int) -> float:
    return sum((v - f) ** 2 for v, f in zip(values[start:], fitted[start:], strict=True))


def _optimise(objective, starts: list[list[float]], bounds: list[tuple[float, float]]):
    best = None
    for x0 in starts:
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def _finish(mean: np.ndarray, fitted: list[float]):
    fitted_array = np.asarray(fitted, dtype=float)
    if not np.all(np.isfinite(mean)):
        return None
    return mean, fitted_array


# --- Suavizado exponencial simple: nivel sin tendencia ni temporada --------------------------


def _ses_pass(values: list[float], alpha: float) -> tuple[list[float], float]:
    level = values[0]
    fitted = [_NAN]
    for v in values[1:]:
        fitted.append(level)
        level = alpha * v + (1 - alpha) * level
    return fitted, level


def ses(y: np.ndarray, h: int):
    values = [float(v) for v in y]
    if len(values) < 3:
        return None
    alpha = _optimise(
        lambda p: _sse(values, _ses_pass(values, p[0])[0], 1),
        [[0.2], [0.6]],
        [(0.01, 0.99)],
    )[0]
    fitted, level = _ses_pass(values, alpha)
    return _finish(np.full(h, level), fitted)


# --- Holt con tendencia amortiguada: nivel + tendencia que se va apagando --------------------


def _holt_pass(values: list[float], alpha: float, beta: float, phi: float):
    level = values[0]
    k = min(4, len(values) - 1)
    trend = (values[k] - values[0]) / k if k else 0.0
    fitted = [_NAN]
    for v in values[1:]:
        predicted = level + phi * trend
        fitted.append(predicted)
        new_level = alpha * v + (1 - alpha) * predicted
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    return fitted, level, trend


def holt_damped(y: np.ndarray, h: int):
    values = [float(v) for v in y]
    if len(values) < 4:
        return None
    alpha, beta, phi = _optimise(
        lambda p: _sse(values, _holt_pass(values, *p)[0], 1),
        [[0.3, 0.1, 0.9], [0.6, 0.05, 0.95]],
        [(0.01, 0.99), (0.01, 0.99), (0.8, 0.98)],
    )
    fitted, level, trend = _holt_pass(values, alpha, beta, phi)
    damping = np.cumsum(phi ** np.arange(1, h + 1))
    return _finish(level + damping * trend, fitted)


# --- Holt-Winters aditivo: nivel + patrón repetitivo (temporada) -----------------------------


def _hw_pass(values: list[float], alpha: float, gamma: float, m: int):
    first = np.asarray(values[: 2 * m], dtype=float).reshape(2, m)
    level = float(first[0].mean())
    seasonal = (first - first.mean(axis=1, keepdims=True)).mean(axis=0).tolist()
    fitted: list[float] = []
    for t, v in enumerate(values):
        s = seasonal[t % m]
        fitted.append(level + s)
        new_level = alpha * (v - s) + (1 - alpha) * level
        seasonal[t % m] = gamma * (v - level) + (1 - gamma) * s
        level = new_level
    return fitted, level, seasonal


def holt_winters(y: np.ndarray, h: int, m: int):
    """Requiere al menos dos ciclos completos (2·m periodos) para estimar la temporada."""
    values = [float(v) for v in y]
    n = len(values)
    if n < 2 * m:
        return None
    alpha, gamma = _optimise(
        lambda p: _sse(values, _hw_pass(values, p[0], p[1], m)[0], 0),
        [[0.2, 0.1], [0.5, 0.3]],
        [(0.01, 0.99), (0.01, 0.99)],
    )
    fitted, level, seasonal = _hw_pass(values, alpha, gamma, m)
    mean = np.array([level + seasonal[(n + k) % m] for k in range(h)])
    return _finish(mean, fitted)
