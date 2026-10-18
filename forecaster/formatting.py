"""Formato de números al estilo español: punto para los miles y coma para los decimales."""

from __future__ import annotations

import numpy as np


def fmt(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.0%}"


def signed_pct(value: float) -> str:
    """Porcentaje con signo; los cambios que redondean a 0 se muestran como «0%», sin signo."""
    if not np.isfinite(value):
        return "—"
    rounded = round(value * 100)
    return "0%" if rounded == 0 else f"{rounded:+d}%"


_MD_SPECIAL = "\\`*_[]#|<>~$"


def md_escape(text: str) -> str:
    """Neutraliza el formato markdown de un texto del usuario (enlaces, negritas, fórmulas)."""
    return "".join("\\" + c if c in _MD_SPECIAL else c for c in text)
