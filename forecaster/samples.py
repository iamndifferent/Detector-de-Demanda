"""Datos de ejemplo sintéticos (no corresponden a ningún negocio real)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# nombre: (demanda diaria base, efecto fin de semana, amplitud anual, día del pico,
#          crecimiento anual, día de lanzamiento)
_PRODUCTS = {
    "Café en grano 1 kg": (12.0, 0.25, 0.10, 20, 0.12, 0),
    "Pan de masa madre": (30.0, 0.35, 0.05, 1, 0.05, 0),
    "Helado artesanal": (15.0, 0.30, 0.80, 200, 0.08, 0),
    "Bufanda de lana": (3.0, 0.10, 0.90, 5, 0.00, 0),
    "Cuaderno A5": (6.0, -0.30, 0.60, 255, 0.05, 0),
    "Funda de móvil": (0.4, 0.00, 0.00, 1, 0.00, 0),
    "Vela aromática": (5.0, 0.20, 0.30, 350, 0.60, 500),
}


def generate_sample_sales(
    seed: int = 42, days: int = 1095, start: str = "2023-01-01"
) -> pd.DataFrame:
    """Tres años de ventas diarias con estacionalidad semanal y anual, una promoción puntual,
    un producto de demanda intermitente y un lanzamiento tardío."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    weekend = (dates.dayofweek >= 5).astype(float)

    frames = []
    for name, (base, weekly, yearly, peak, growth, launch) in _PRODUCTS.items():
        season = 1 + yearly * np.cos(2 * np.pi * (dates.dayofyear - peak) / 365.25)
        rate = base * (1 + weekly * weekend) * season * (1 + growth * t / 365)
        if name == "Café en grano 1 kg":
            rate = np.where((t >= 700) & (t < 704), rate * 3, rate)  # promoción puntual
        qty = rng.poisson(np.clip(rate, 0.01, None))
        keep = (qty > 0) & (t >= launch)
        frames.append(
            pd.DataFrame(
                {
                    "fecha": dates[keep].strftime("%Y-%m-%d"),
                    "producto": name,
                    "cantidad": qty[keep],
                }
            )
        )
    sales = pd.concat(frames, ignore_index=True)
    return sales.sort_values(["fecha", "producto"], ignore_index=True)


def sample_csv_bytes() -> bytes:
    return generate_sample_sales().to_csv(index=False).encode("utf-8")


def template_csv_bytes() -> bytes:
    """Plantilla mínima para que el usuario vea el formato esperado."""
    return generate_sample_sales().head(12).to_csv(index=False).encode("utf-8")


def generate_cafe_example(seed: int = 7) -> bytes:
    """CSV de una cafetería tal y como lo exportaría Excel en español: separado por `;`,
    fechas día/mes/año, decimales con coma, codificación Latin-1 y 4 filas con errores a propósito
    (fecha imposible, devolución, producto vacío y cantidad no numérica)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-03", "2025-09-14", freq="D")  # 28 semanas completas
    rows = []
    for d in dates:
        weekend = d.dayofweek >= 5
        coffee = max(0.0, rng.normal(2.2 * (1.3 if weekend else 1.0), 0.5))  # kg
        croissants = rng.poisson(24 * (1.5 if weekend else 1.0))
        tea = rng.poisson(max(0.5, 4 - 0.012 * (d - dates[0]).days))
        stamp = d.strftime("%d/%m/%Y")
        rows.append((stamp, "Café molido (kg)", f"{coffee:.1f}".replace(".", ",")))
        rows.append((stamp, "Croissant", str(croissants)))
        if tea:
            rows.append((stamp, "Té verde", str(tea)))
    rows += [
        ("31/02/2025", "Croissant", "5"),
        ("10/06/2025", "Croissant", "-3"),
        ("11/06/2025", "", "4"),
        ("12/06/2025", "Té verde", "abc"),
    ]
    frame = pd.DataFrame(rows, columns=["fecha", "producto", "cantidad"])
    return frame.to_csv(sep=";", index=False).encode("latin-1")
