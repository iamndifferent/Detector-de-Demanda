"""Selección automática del mejor modelo por validación y cálculo del rango probable."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BASELINE_MODEL, INTERVAL_Z, MAX_FOLDS, MIN_POINTS, MIN_TRAIN
from .evaluation import ModelFn, rolling_origin_wape
from .smoothing import holt_damped, holt_winters, ses


@dataclass
class Forecast:
    model: str
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigma: float  # desviación típica del error a un paso (base del stock de seguridad)
    scores: dict[str, float]  # WAPE de validación de cada modelo evaluado
    n_obs: int
    warnings: list[str] = field(default_factory=list)

    @property
    def best_wape(self) -> float:
        return self.scores.get(self.model, float("nan"))

    @property
    def baseline_wape(self) -> float:
        return self.scores.get(BASELINE_MODEL, float("nan"))

    @property
    def improvement(self) -> float:
        """Cuánto menos error tiene el modelo elegido que la media móvil (0.2 = 20 % menos)."""
        base, best = self.baseline_wape, self.best_wape
        if not (np.isfinite(base) and np.isfinite(best)) or base == 0:
            return float("nan")
        return 1 - best / base

    @property
    def reliability(self) -> str:
        wape = self.best_wape
        if not np.isfinite(wape) or self.n_obs < MIN_POINTS:
            return "Baja"
        if wape <= 0.25:
            return "Alta"
        return "Media" if wape <= 0.5 else "Baja"


# --- Modelos sencillos de referencia ---------------------------------------------------------


def _last_value(y: np.ndarray, h: int):
    return np.full(h, y[-1]), np.concatenate([[np.nan], y[:-1]])


def _moving_average(window: int) -> ModelFn:
    def model(y: np.ndarray, h: int):
        w = min(window, len(y))
        fitted = np.full(len(y), np.nan)
        if len(y) > w:
            sums = np.convolve(y, np.ones(w), mode="valid")
            fitted[w:] = sums[:-1] / w
        return np.full(h, y[-w:].mean()), fitted

    return model


def _seasonal_naive(m: int) -> ModelFn:
    def model(y: np.ndarray, h: int):
        if len(y) < m + 1:
            return None
        fitted = np.full(len(y), np.nan)
        fitted[m:] = y[:-m]
        last_cycle = y[-m:]
        return np.array([last_cycle[k % m] for k in range(h)]), fitted

    return model


def build_candidates(n: int, season: int, window: int) -> dict[str, ModelFn]:
    """Modelos que tiene sentido probar según cuánto historial hay."""
    candidates: dict[str, ModelFn] = {
        BASELINE_MODEL: _moving_average(window),
        "Último valor": _last_value,
    }
    if n >= season + 1:
        candidates["Estacional ingenuo"] = _seasonal_naive(season)
    if n >= MIN_POINTS:
        candidates["Suavizado exponencial"] = ses
        candidates["Suavizado con tendencia"] = holt_damped
    if n >= max(MIN_POINTS, 2 * season):
        candidates["Suavizado estacional"] = lambda y, h: holt_winters(y, h, season)
    return candidates


def forecast_series(y: np.ndarray, horizon: int, season: int, window: int) -> Forecast:
    """Prevé los próximos `horizon` periodos de una serie de ventas.

    Prueba varios modelos contra el pasado reciente y se queda con el de menor error. En caso
    de empate gana el más simple (la media móvil se evalúa primero).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 periodos de historial.")
    if horizon < 1:
        raise ValueError("El horizonte debe ser de al menos 1 periodo.")

    notes: list[str] = []
    candidates = build_candidates(n, season, window)

    validation_len = max(1, min(horizon, n // 4))
    folds = min(MAX_FOLDS, (n - MIN_TRAIN) // validation_len)
    scores: dict[str, float] = {}
    if folds >= 1:
        for name, model in candidates.items():
            score = rolling_origin_wape(y, model, validation_len, folds)
            if np.isfinite(score):
                scores[name] = score
    else:
        notes.append("No hay historial suficiente para validar los modelos: se usa la media móvil.")

    chosen = min(scores, key=scores.get) if scores else BASELINE_MODEL
    result = candidates[chosen](y, horizon)
    if result is None:  # un modelo que falla al ajustarse no debe tumbar la previsión
        chosen = BASELINE_MODEL
        result = candidates[chosen](y, horizon)
    mean, fitted = result
    mean = np.clip(mean, 0, None)

    residuals = (y - fitted)[np.isfinite(fitted)]
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else float(np.std(y))
    lower = np.clip(mean - INTERVAL_Z * sigma, 0, None)
    upper = mean + INTERVAL_Z * sigma

    if n < MIN_POINTS:
        notes.append(f"Solo hay {n} periodos de historial: la previsión es orientativa.")
    elif n < 2 * season:
        notes.append(
            f"Con menos de {2 * season} periodos no se puede detectar la estacionalidad "
            "(patrones que se repiten por temporada)."
        )
    if np.mean(y == 0) > 0.5:
        notes.append(
            "Más de la mitad de los periodos no tuvo ventas (demanda intermitente): "
            "la incertidumbre es alta."
        )
    if scores.get(chosen, 0.0) > 0.5:
        notes.append("El error de validación supera el 50 %: úsala solo como referencia.")

    return Forecast(
        model=chosen,
        mean=mean,
        lower=lower,
        upper=upper,
        sigma=sigma,
        scores=scores,
        n_obs=n,
        warnings=notes,
    )
