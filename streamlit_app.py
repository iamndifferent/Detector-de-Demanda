"""Predictor de demanda para negocios pequeños: interfaz de Streamlit."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from forecaster import __version__
from forecaster.charts import order_chart, overview_chart, product_chart
from forecaster.cleaning import build_series, cap_outliers, clean_sales
from forecaster.config import (
    FREQUENCIES,
    MAX_FILE_BYTES,
    MAX_PRODUCTS,
    MAX_ROWS,
    MIN_POINTS,
    Frequency,
)
from forecaster.export import to_csv_bytes
from forecaster.formatting import fmt, pct
from forecaster.loader import DataError, guess_columns, read_csv
from forecaster.models import Forecast, forecast_series
from forecaster.readiness import assess_data, file_checks
from forecaster.replenishment import cover_periods_for, suggest_order
from forecaster.samples import generate_cafe_example, sample_csv_bytes, template_csv_bytes
from forecaster.summary import ProductResult, build_summary, recent_trend, trend_short

st.set_page_config(
    page_title="Predictor de demanda",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "Predictor de demanda: previsión estadística de ventas "
            "y recomendaciones de inventario."
        )
    },
)

ICONS = {"ok": "✅", "info": "ℹ️", "warn": "⚠️", "error": "❌"}


# --- Funciones con caché ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_raw(data: bytes) -> pd.DataFrame:
    return read_csv(data)


@st.cache_data(show_spinner="Limpiando datos…")
def prepare(data, date_col, product_col, qty_col, dayfirst, drop_duplicates):
    return clean_sales(
        load_raw(data),
        date_col,
        product_col,
        qty_col,
        dayfirst=dayfirst,
        drop_duplicates=drop_duplicates,
    )


@st.cache_data(show_spinner=False, max_entries=512)
def cached_forecast(values: tuple, horizon: int, season: int, window: int) -> Forecast:
    return forecast_series(np.asarray(values, dtype=float), horizon, season, window)


def compute_results(
    names: list[str],
    series: dict[str, pd.Series],
    stock: dict[str, float],
    cfg: Frequency,
    horizon: int,
    covered: int,
    service: float,
) -> list[ProductResult]:
    results = []
    for name in names:
        history = series[name]
        if len(history) < 2:
            continue
        values = tuple(history.to_numpy().tolist())
        fc = cached_forecast(values, horizon, cfg.season, cfg.window)
        suggestion = suggest_order(fc.mean, fc.sigma, covered, service, float(stock.get(name, 0.0)))
        future = pd.period_range(start=history.index[-1] + 1, periods=horizon)
        results.append(ProductResult(name, history, fc, suggestion, future))
    return results


def show_requirements() -> None:
    """Condiciones del archivo y cantidad de datos necesaria, antes de subir nada."""
    st.markdown(
        f"""
**Formato del archivo**
- Un archivo **CSV** (`.csv`) de hasta **{MAX_FILE_BYTES // 1_048_576} MB**, con un máximo de
  {fmt(MAX_ROWS)} filas y {fmt(MAX_PRODUCTS)} productos distintos.
- La **primera fila** contiene los nombres de las columnas.
- **Tres columnas**: la **fecha** de la venta, el **producto** y la **cantidad** vendida.
  Una fila por venta, o una por día y producto. El orden de las filas da igual.
- **Fechas**: `2025-03-01` o `01/03/2025` (día/mes/año; puedes cambiarlo en las opciones).
- **Cantidades**: números, con punto o con coma decimal (`2.5` o `2,5`). Sin unidades ni
  símbolos (no escribas «5 kg» ni «12 €»).
- **Productos**: escríbelos siempre igual («Café» y «cafe» contarían como dos productos).
- Separador `,` `;` o tabulador, y codificación UTF-8 o Latin-1 (la detecta sola).
- **Evita**: filas de totales, celdas combinadas, varias tablas en una hoja y filas vacías.
- Desde Excel: *Archivo → Guardar como → CSV UTF-8 (delimitado por comas)*.

**Cuántos datos hacen falta** (según cómo agrupes las ventas)
"""
    )
    rows = [
        "| Agrupación | Mínimo para prever | Recomendado | Para detectar temporadas |",
        "|---|---|---|---|",
    ]
    for f in FREQUENCIES.values():
        rows.append(
            f"| {f.name} | {MIN_POINTS} {f.plural} | {f.recommended} {f.plural} "
            f"| {2 * f.season} {f.plural} |"
        )
    st.markdown("\n".join(rows))
    st.markdown(
        """
- Los periodos cuentan desde la primera venta de cada producto. Si no hay ventas un día, no hace
  falta que aparezca ninguna fila: se interpreta como 0.
- Con menos del mínimo la app te avisa de que las cifras son solo orientativas.
- «Temporadas» son picos que se repiten (verano, Navidad…): la app necesita ver al menos
  **dos** ciclos completos para detectarlos.

**Ejemplo de archivo correcto**
```
fecha;producto;cantidad
03/03/2025;Café molido (kg);2,2
03/03/2025;Croissant;31
04/03/2025;Croissant;29
```
"""
    )


# --- Cabecera --------------------------------------------------------------------------------

st.title("📈 Predictor de demanda")
st.write(
    "Sube tus ventas y descubre cuánto venderás las próximas semanas y cuánto te conviene pedir. "
    "Gratis, en español y pensado para negocios pequeños."
)
with st.expander("Privacidad y límites"):
    st.markdown(
        "- El código de esta app **no guarda tus archivos**: se procesan en memoria durante la "
        "sesión.\n"
        "- Si la usas en una versión alojada por terceros, no subas datos personales de clientes. "
        "Para datos sensibles, ejecútala en tu ordenador (hay instrucciones en el README).\n"
        "- Es una **estimación estadística**, no una garantía: no conoce promociones futuras, "
        "cambios de precio ni roturas de stock pasadas."
    )

# --- Entrada principal de datos ---------------------------------------------------------------

st.subheader("1. Empieza aquí")
st.write(
    "Prueba la aplicación con datos de ejemplo o sube tus propias ventas "
    "para calcular la demanda y preparar tu próximo pedido."
)

source = st.radio(
    "¿Qué datos quieres analizar?",
    ["Datos de ejemplo", "Ejemplo de cafetería", "Subir mi CSV"],
    index=0,
)

uploaded = None
if source == "Subir mi CSV":
    uploaded = st.file_uploader(
        "Selecciona tu archivo de ventas",
        type=["csv"],
        help=(
            f"CSV de hasta {MAX_FILE_BYTES // 1_048_576} MB "
            "con fecha, producto y cantidad."
        ),
    )

st.download_button(
    "Descargar plantilla de ventas",
    template_csv_bytes(),
    file_name="plantilla_ventas.csv",
    mime="text/csv",
)
st.caption("Puedes empezar con el ejemplo y después cargar tus propios datos.")

# --- Configuración de la previsión -------------------------------------------------------------

with st.expander("2. Ajustes de previsión y compras", expanded=False):
    st.markdown("**Periodo de análisis**")
    freq_name = st.selectbox("Agrupar las ventas por", list(FREQUENCIES), index=1)
    cfg = FREQUENCIES[freq_name]
    horizon = st.slider(
        f"¿Cuántos {cfg.plural} quieres prever?",
        1,
        cfg.max_horizon,
        cfg.default_horizon,
    )

    st.markdown("**Limpieza de los datos**")
    dayfirst = st.checkbox("Las fechas van en formato día/mes/año", value=True)
    drop_incomplete = st.checkbox(
        f"Ignorar el último {cfg.unit} si está incompleto",
        value=True,
    )
    drop_dups = st.checkbox("Eliminar filas idénticas", value=False)
    cap_on = st.checkbox("Limitar picos extremos de ventas", value=False)

    st.markdown("**Reposición de inventario**")
    lead_days = st.number_input(
        "Plazo de entrega (días)",
        min_value=0,
        max_value=180,
        value=7,
    )
    extra_days = st.number_input(
        "Días extra de cobertura",
        min_value=0,
        max_value=180,
        value=7,
    )
    service = st.select_slider(
        "Nivel de servicio",
        options=[0.90, 0.95, 0.97, 0.99],
        value=0.95,
        format_func=lambda value: f"{value:.0%}",
        help=(
            "Nivel de servicio objetivo utilizado para calcular "
            "el stock de seguridad."
        ),
    )

st.caption(f"Versión {__version__}")

# --- Requisitos y datos de entrada -----------------------------------------------------------

awaiting_upload = source == "Subir mi CSV" and uploaded is None
with st.expander(
    "Requisitos del archivo y cantidad de datos necesarios", expanded=awaiting_upload
):
    show_requirements()

if source == "Datos de ejemplo":
    data = sample_csv_bytes()
elif source == "Ejemplo de cafetería":
    data = generate_cafe_example()
elif uploaded is not None:
    data = uploaded.getvalue()
else:
    st.info("Cuando tu archivo cumpla los requisitos, súbelo en la pantalla principal.")
    st.stop()

try:
    raw = load_raw(data)
except DataError as exc:
    st.error(str(exc))
    st.stop()

columns = list(raw.columns)
guess = guess_columns(columns)
with st.expander("Columnas del archivo", expanded=source == "Subir mi CSV"):
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Fecha", columns, index=columns.index(guess["date"]))
    product_col = c2.selectbox("Producto", columns, index=columns.index(guess["product"]))
    qty_col = c3.selectbox("Cantidad vendida", columns, index=columns.index(guess["quantity"]))

try:
    result = prepare(data, date_col, product_col, qty_col, dayfirst, drop_dups)
except DataError as exc:
    st.error(str(exc))
    st.stop()

report = result.report
for message in report.messages:
    st.warning(message)

series, dropped = build_series(result.data, cfg, drop_incomplete)
if not series:
    st.error("Después de agrupar no queda ningún producto con ventas. Revisa los datos.")
    st.stop()
if dropped:
    st.caption(f"Se ignoró el último {cfg.unit} porque todavía no había terminado.")
if cap_on:
    capped_total = 0
    for name, s in series.items():
        values, n_capped = cap_outliers(s.to_numpy())
        series[name] = pd.Series(values, index=s.index)
        capped_total += n_capped
    st.caption(f"Se limitaron {capped_total} picos extremos de ventas.")

# --- Verificación del archivo ----------------------------------------------------------------

assessment = assess_data(series, cfg)
st.subheader("Verificación de tu archivo")
for check in file_checks(report, assessment, cfg):
    st.markdown(f"{ICONS[check.status]} {check.text}")
st.caption(f"Nivel de tus datos: **{assessment.level}**")

ranking = sorted(series, key=lambda p: series[p].sum(), reverse=True)
covered = cover_periods_for(lead_days, extra_days, cfg.days)
eff_horizon = max(horizon, covered)
data_id = hashlib.blake2b(data, digest_size=8).hexdigest()
stock_by_name: dict[str, float] = {}

tab_summary, tab_detail, tab_how = st.tabs(["Resumen", "Detalle por producto", "Cómo funciona"])

# --- Pestaña 1: resumen para el cliente ------------------------------------------------------

with tab_summary:
    max_n = min(30, len(ranking))
    if max_n > 3:
        top_n = st.slider(
            "¿Cuántos productos analizar? (los más vendidos)", 3, max_n, min(5, max_n)
        )
    else:
        top_n = max_n
    top = ranking[:top_n]
    with st.expander("Opcional: indica tu stock actual para afinar los pedidos"):
        stock_df = st.data_editor(
            pd.DataFrame({"Producto": top, "Stock actual": 0.0}),
            disabled=["Producto"],
            hide_index=True,
            key=f"stock_{data_id}_{top_n}",
            column_config={"Stock actual": st.column_config.NumberColumn(min_value=0.0, step=1.0)},
        )
    stock_by_name = dict(zip(stock_df["Producto"], stock_df["Stock actual"].fillna(0.0)))
    stock_provided = any(v > 0 for v in stock_by_name.values())

    with st.spinner("Calculando previsiones…"):
        results = compute_results(top, series, stock_by_name, cfg, eff_horizon, covered, service)

    if not results:
        st.warning("Ningún producto tiene historial suficiente para hacer una previsión.")
    else:
        summary = build_summary(
            report,
            assessment,
            cfg,
            results,
            horizon=horizon,
            lead_days=lead_days,
            extra_days=extra_days,
            service_level=service,
            stock_provided=stock_provided,
        )
        st.success(summary.headline)

        st.subheader("Qué analizamos")
        for point in summary.points:
            st.markdown(f"**{point.title}.** {point.text}")

        st.subheader("Qué hacer ahora")
        for i, action in enumerate(summary.actions, start=1):
            st.markdown(f"{i}. {action}")

        st.subheader("Tus ventas y la previsión")
        st.plotly_chart(
            overview_chart(results, cfg.unit),
            width="stretch",
            config={"responsive": True, "displaylogo": False},
        )
        st.caption(
            "Línea negra: tus ventas reales. Línea verde discontinua: la previsión. Zona verde: "
            "rango probable (80 %). Se muestran los 4 productos más vendidos; el icono de la "
            "cámara del gráfico lo descarga como imagen."
        )

        st.subheader("Cuánto pedir de cada producto")
        st.plotly_chart(
            order_chart(results),
            width="stretch",
            config={"responsive": True, "displaylogo": False},
        )
        table = pd.DataFrame(
            [
                {
                    "Producto": r.name,
                    "Tendencia reciente": trend_short(
                        recent_trend(r.history.to_numpy(), cfg.window)
                    ),
                    "Modelo": r.forecast.model,
                    "Fiabilidad": r.forecast.reliability,
                    "Error de validación (%)": round(r.forecast.best_wape * 100)
                    if np.isfinite(r.forecast.best_wape)
                    else None,
                    "Ventas previstas": round(r.suggestion.demand, 1),
                    "Stock de seguridad": round(r.suggestion.safety_stock, 1),
                    "Stock actual": r.suggestion.on_hand,
                    "Pedir": r.suggestion.order_qty,
                }
                for r in results
            ]
        )
        visible_columns = [
            "Producto",
            "Tendencia reciente",
            "Fiabilidad",
            "Ventas previstas",
            "Stock actual",
            "Pedir",
        ]
        st.dataframe(table[visible_columns], hide_index=True, width="stretch")
        st.download_button(
            "Descargar tabla de compras (CSV)",
            to_csv_bytes(table),
            "resumen_compras.csv",
            "text/csv",
        )

# --- Pestaña 2: detalle de un producto -------------------------------------------------------

with tab_detail:
    product = st.selectbox("Producto", ranking, help="Ordenados de más a menos vendidos.")
    history = series[product]
    if len(history) < 2:
        st.warning("Este producto tiene muy poco historial para hacer una previsión.")
    else:
        if eff_horizon > horizon:
            st.caption(
                f"Se prevén {eff_horizon} {cfg.plural} para cubrir el plazo de entrega."
            )
        fc = cached_forecast(
            tuple(history.to_numpy().tolist()), eff_horizon, cfg.season, cfg.window
        )
        future = pd.period_range(start=history.index[-1] + 1, periods=eff_horizon)

        k1, k2, k3 = st.columns(3)
        k1.metric("Modelo elegido", fc.model)
        k2.metric(
            "Error en la validación",
            pct(fc.best_wape),
            help=(
                "Cuánto se equivocó este modelo al 'adivinar' tus ventas recientes (WAPE). "
                "Menos es mejor."
            ),
        )
        k3.metric(
            "Mejora frente a la media móvil",
            pct(fc.improvement),
            help=(
                "Reducción de error respecto al método simple de promediar las últimas ventas."
            ),
        )
        st.caption(f"Fiabilidad estimada: **{fc.reliability}**.")
        for note in fc.warnings:
            st.warning(note)

        st.plotly_chart(
            product_chart(history, fc, future, cfg.unit),
            width="stretch",
            config={"responsive": True, "displaylogo": False},
        )

        st.subheader("¿Cuánto pedir?")
        on_hand = st.number_input(
            "Stock actual (unidades)", 0.0, value=float(stock_by_name.get(product, 0.0)), step=1.0
        )
        sug = suggest_order(fc.mean, fc.sigma, covered, service, on_hand)
        s1, s2 = st.columns(2)
        s1.metric(
            f"Ventas previstas ({sug.periods_covered} {cfg.plural})",
            fmt(sug.demand, 1),
        )
        s2.metric("Stock de seguridad", fmt(sug.safety_stock, 1))

        s3, s4 = st.columns(2)
        s3.metric("Stock actual", fmt(on_hand, 1))
        s4.metric("Cantidad sugerida a pedir", fmt(sug.order_qty))
        st.caption(
            "Pedido = ventas previstas + stock de seguridad − stock actual. "
            f"Con un nivel de servicio del {service:.0%}."
        )

        forecast_table = pd.DataFrame(
            {
                cfg.label: future.start_time.strftime("%d/%m/%Y"),
                "Previsión": fc.mean.round(1),
                "Mínimo probable": fc.lower.round(1),
                "Máximo probable": fc.upper.round(1),
            }
        )
        st.dataframe(forecast_table, hide_index=True)
        st.download_button(
            "Descargar previsión (CSV)",
            to_csv_bytes(forecast_table.assign(Producto=product)),
            "prevision.csv",
            "text/csv",
        )
        with st.expander("¿Cómo se eligió el modelo?"):
            ranking_models = pd.DataFrame(
                {"Modelo": list(fc.scores), "Error de validación": list(fc.scores.values())}
            ).sort_values("Error de validación")
            ranking_models["Error de validación"] = ranking_models["Error de validación"].map(pct)
            st.dataframe(ranking_models, hide_index=True)
            st.write(
                "Cada modelo se probó prediciendo tus ventas recientes sin mirarlas. "
                "Gana el que menos se equivocó."
            )

# --- Pestaña 3: explicación ------------------------------------------------------------------

with tab_how:
    st.markdown(
        """
**1. Verificación.** Se comprueba que el archivo tenga el formato correcto y suficientes datos.

**2. Limpieza.** Se leen las fechas, se descartan filas inválidas y se agrupan las ventas por
día, semana o mes. Los periodos sin ventas cuentan como 0.

**3. Modelos.** Se prueban métodos sencillos (media móvil, último valor, repetir la temporada
anterior) y tres variantes de suavizado exponencial (nivel, tendencia y temporada).

**4. Validación.** Cada modelo intenta predecir tus ventas recientes sin verlas. Se elige el que
menos se equivoca (WAPE) y se compara con la media móvil, el método más habitual.

**5. Compra.** Pedido = ventas previstas durante el plazo de entrega y la cobertura elegida +
stock de seguridad − stock actual.

**Glosario**
- **Error de validación (WAPE)**: porcentaje medio en que el modelo falló al predecir tus ventas
  recientes. 5 % es muy bueno; más del 50 % es poco fiable.
- **Rango probable**: zona en la que caerán las ventas aproximadamente 8 de cada 10 veces.
- **Stock de seguridad**: colchón extra para no quedarte sin producto si vendes más de lo previsto.
- **Nivel de servicio**: probabilidad de no quedarte sin stock durante el plazo de entrega.
- **Fiabilidad**: Alta si el error de validación es de hasta el 25 %, Media hasta el 50 %, Baja
  por encima o si hay muy poco historial.

**Límites.** Con poco historial la previsión es orientativa; si hubo roturas de stock, las
ventas pasadas subestiman la demanda real; no se conocen promociones ni cambios de precio futuros.
"""
    )

