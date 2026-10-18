"""Gráficos (Plotly) con los datos del cliente."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .formatting import fmt
from .models import Forecast
from .summary import ProductResult

GREEN = "#1F6F5C"
INK = "#14201C"
BAND = "rgba(31,111,92,0.18)"


def _forecast_traces(fc: Forecast, future: pd.PeriodIndex, show_legend: bool) -> list[go.Scatter]:
    x = future.to_timestamp()
    return [
        go.Scatter(x=x, y=fc.upper, mode="lines", line={"width": 0},
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=x, y=fc.lower, mode="lines", line={"width": 0}, fill="tonexty",
                   fillcolor=BAND, name="Rango probable (80 %)", legendgroup="band",
                   showlegend=show_legend, hoverinfo="skip"),
    ]  # fmt: skip


def product_chart(history: pd.Series, fc: Forecast, future: pd.PeriodIndex, unit: str) -> go.Figure:
    """Ventas reales, previsión y rango probable de un producto."""
    fig = go.Figure()
    for trace in _forecast_traces(fc, future, True):
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(x=history.index.to_timestamp(), y=history.to_numpy(), mode="lines",
                             name="Ventas reales", line={"color": INK, "width": 1.8}))  # fmt: skip
    fig.add_trace(go.Scatter(x=future.to_timestamp(), y=fc.mean, mode="lines+markers",
                             name="Previsión",
                             line={"color": GREEN, "width": 2.4, "dash": "dash"}))  # fmt: skip
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.12},
        yaxis_title=f"Unidades por {unit}",
        hovermode="x unified",
    )
    return fig


def overview_chart(results: list[ProductResult], unit: str, max_products: int = 4) -> go.Figure:
    """Un panel por producto (los más vendidos) con su historial y su previsión."""
    chosen = results[:max_products]
    fig = make_subplots(
        rows=len(chosen), cols=1, shared_xaxes=True,
        subplot_titles=[r.name for r in chosen], vertical_spacing=0.12,
    )  # fmt: skip
    for i, r in enumerate(chosen, start=1):
        first = i == 1
        for trace in _forecast_traces(r.forecast, r.future, first):
            fig.add_trace(trace, row=i, col=1)
        fig.add_trace(go.Scatter(x=r.history.index.to_timestamp(), y=r.history.to_numpy(),
                                 mode="lines", name="Ventas reales", legendgroup="real",
                                 showlegend=first, line={"color": INK, "width": 1.6}),
                      row=i, col=1)  # fmt: skip
        fig.add_trace(go.Scatter(x=r.future.to_timestamp(), y=r.forecast.mean, mode="lines",
                                 name="Previsión", legendgroup="forecast", showlegend=first,
                                 line={"color": GREEN, "width": 2.2, "dash": "dash"}),
                      row=i, col=1)  # fmt: skip
    fig.update_layout(
        template="plotly_white",
        height=210 * len(chosen) + 90,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "y": 1.0 + 0.35 / len(chosen)},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=f"por {unit}")
    return fig


def order_chart(results: list[ProductResult]) -> go.Figure:
    """Barras horizontales con la cantidad sugerida a pedir (la mayor arriba)."""
    ordered = sorted(results, key=lambda r: r.suggestion.order_qty)
    quantities = [r.suggestion.order_qty for r in ordered]
    fig = go.Figure(
        go.Bar(
            x=quantities,
            y=[r.name for r in ordered],
            orientation="h",
            text=[fmt(q) for q in quantities],
            textposition="outside",
            marker_color=GREEN,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=70 * len(ordered) + 110,
        margin={"l": 10, "r": 30, "t": 20, "b": 10},
        xaxis_title="Cantidad sugerida a pedir",
    )
    return fig
