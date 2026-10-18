"""Constantes y ajustes compartidos por toda la aplicación."""

from __future__ import annotations

from dataclasses import dataclass

# Límites de entrada (protegen la app frente a archivos enormes o malformados).
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ROWS = 200_000
MAX_PRODUCTS = 2_000

# Parámetros del modelado.
MIN_POINTS = 8  # periodos mínimos para usar suavizado exponencial
MIN_TRAIN = 5  # periodos mínimos de entrenamiento en cada prueba de validación
MAX_FOLDS = 3  # número máximo de pruebas de validación por producto
INTERVAL_Z = 1.2816  # rango probable del 80 %
BASELINE_MODEL = "Media móvil"  # el método "de toda la vida" con el que se compara


@dataclass(frozen=True)
class Frequency:
    """Cómo se agrupan las ventas y qué significa una "temporada" en cada caso."""

    name: str
    code: str  # alias de periodo de pandas
    unit: str
    plural: str
    season: int  # periodos que dura un ciclo estacional
    window: int  # ventana de la media móvil
    days: float  # días que dura un periodo (para convertir plazos de entrega)
    default_horizon: int
    max_horizon: int
    label: str  # cabecera de la columna de fechas en las tablas
    recommended: int  # periodos de historial recomendados para una previsión fiable


FREQUENCIES: dict[str, Frequency] = {
    f.name: f
    for f in (
        Frequency("Diaria", "D", "día", "días", 7, 7, 1.0, 14, 60, "Día", 60),
        Frequency(
            "Semanal", "W-SUN", "semana", "semanas", 52, 4, 7.0, 8, 26, "Semana que empieza", 26
        ),
        Frequency("Mensual", "M", "mes", "meses", 12, 3, 30.44, 6, 12, "Mes", 24),
    )
}
