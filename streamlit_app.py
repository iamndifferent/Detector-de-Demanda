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
    initial_sidebar_state="collapsed",
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


# --- Interfaz principal: sencilla por defecto y adaptable a móvil ------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.35rem, 3.2vw, 2rem);
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 0.75rem 0.75rem 1.5rem;
        }

        h1 {
            font-size: 1.7rem;
            line-height: 1.2;
        }

        h2 {
            font-size: 1.3rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Predictor de demanda")
st.subheader("Planifica mejor tus compras")
st.write(
    "Convierte tus ventas en una previsión sencilla y descubre cuánto podrías necesitar "
    "pedir. No hace falta saber de estadística."
)

with st.expander("Privacidad y límites", expanded=False):
    st.markdown(
        "- El código de esta app procesa el archivo en memoria durante la sesión; no lo guarda "
        "deliberadamente.\n"
        "- No subas datos personales de clientes ni información confidencial.\n"
        "- Las previsiones son estimaciones estadísticas, no garantías. No conocen promociones, "
        "cambios de precios ni roturas de stock que no estén reflejadas en el historial."
    )

# Paso 1: elegir el origen. La demo está claramente identificada como ficticia.
st.subheader("1 · Empieza aquí")
source = st.radio(
    "¿Qué quieres analizar?",
    ["Probar con datos de ejemplo", "Subir mis ventas (CSV)"],
    horizontal=True,
    help="Puedes probar la aplicación sin subir ningún archivo.",
)

uploaded = None
if source == "Probar con datos de ejemplo":
    example_kind = st.selectbox(
        "Elige una demostración",
        ["Ventas de ejemplo", "Cafetería (incluye errores para probar la limpieza)"],
        index=0,
    )
    if example_kind == "Cafetería (incluye errores para probar la limpieza)":
        data = generate_cafe_example()
    else:
        data = sample_csv_bytes()
    st.info(
        "Estás viendo datos ficticios de demostración. Para analizar tu negocio, selecciona "
        "«Subir mis ventas (CSV)»."
    )
else:
    uploaded = st.file_uploader(
        "Sube el archivo CSV de tus ventas",
        type=["csv"],
        help=(
            f"Máximo {MAX_FILE_BYTES // 1_048_576} MB. Debe incluir fecha, producto y cantidad."
        ),
    )
    st.download_button(
        "Descargar plantilla CSV",
        template_csv_bytes(),
        file_name="plantilla_ventas.csv",
        mime="text/csv",
        help="Usa esta plantilla si no sabes cómo organizar las columnas.",
    )
    st.caption("Desde Excel: Guardar como → CSV UTF-8. No subas una hoja de cálculo XLSX.")
    with st.expander("¿Qué debe contener el CSV?", expanded=uploaded is None):
        show_requirements()
    if uploaded is None:
        st.info("Cuando selecciones tu archivo, aquí aparecerá el análisis.")
        st.stop()
    data = uploaded.getvalue()

# Paso 2: valores recomendados; las opciones técnicas quedan ocultas inicialmente.
st.subheader("2 · Ajustes (puedes dejar los valores iniciales)")
with st.expander("Personalizar previsión, limpieza y pedido", expanded=False):
    st.caption(
        "Los valores iniciales sirven para empezar. Cambia estos ajustes solo si sabes que "
        "tu negocio necesita otra frecuencia o plazo de entrega."
    )
    st.markdown("**Previsión de ventas**")
    freq_name = st.selectbox(
        "Agrupar las ventas por",
        list(FREQUENCIES),
        index=1,
        help="Semanal es una opción cómoda para la mayoría de los pequeños negocios.",
    )
    cfg = FREQUENCIES[freq_name]
    horizon = st.slider(
        f"Cuántos {cfg.plural} prever",
        min_value=1,
        max_value=cfg.max_horizon,
        value=cfg.default_horizon,
    )

    st.divider()
    st.markdown("**Plazo para preparar el pedido**")
    lead_days = st.number_input(
        "Días que tarda el proveedor en entregar",
        min_value=0,
        max_value=180,
        value=7,
        help="Por ejemplo, escribe 3 si normalmente recibes el pedido en tres días.",
    )
    extra_days = st.number_input(
        "Días adicionales de margen",
        min_value=0,
        max_value=180,
        value=7,
        help="Un margen para cubrir posibles retrasos o ventas superiores a lo esperado.",
    )
    service = st.select_slider(
        "Nivel de protección del stock",
        options=[0.90, 0.95, 0.97, 0.99],
        value=0.95,
        format_func=lambda value: f"{value:.0%}",
        help=(
            "Un porcentaje mayor añade más stock de seguridad y reduce el riesgo de quedarte "
            "sin existencias, pero puede aumentar el pedido."
        ),
    )

    st.divider()
    st.markdown("**Limpieza de datos (normalmente no hace falta cambiarlo)**")
    dayfirst = st.checkbox("Interpretar fechas como día/mes/año", value=True)
    drop_incomplete = st.checkbox(
        f"Ignorar el último {cfg.unit} si todavía no ha terminado",
        value=True,
    )
    drop_dups = st.checkbox("Eliminar filas completamente duplicadas", value=False)
    cap_on = st.checkbox(
        "Limitar picos de ventas muy extremos",
        value=False,
        help="Actívalo solo si hay picos aislados que no representan la demanda habitual.",
    )

try:
    raw = load_raw(data)
except DataError as exc:
    st.error(f"No hemos podido leer el archivo: {exc}")
    st.info("Descarga la plantilla CSV y copia ahí las columnas fecha, producto y cantidad.")
    st.stop()

columns = list(raw.columns)
guess = guess_columns(columns)
with st.expander("Revisar las columnas detectadas", expanded=False):
    st.caption(
        "La app intenta reconocerlas automáticamente. Si el resultado es incorrecto, "
        "cámbialas aquí."
    )
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Fecha de venta", columns, index=columns.index(guess["date"]))
    product_col = c2.selectbox(
        "Nombre del producto", columns, index=columns.index(guess["product"])
    )
    qty_col = c3.selectbox("Cantidad vendida", columns, index=columns.index(guess["quantity"]))

# Los valores por defecto también se usan cuando el usuario no abre la sección avanzada.
# Se recuperan de los widgets de Streamlit cuando ya están definidos arriba.
try:
    result = prepare(data, date_col, product_col, qty_col, dayfirst, drop_dups)
except DataError as exc:
    st.error(f"Hay un problema con las columnas o con los datos: {exc}")
    st.info("Abre «Revisar las columnas detectadas» y comprueba fecha, producto y cantidad.")
    st.stop()

report = result.report
series, dropped = build_series(result.data, cfg, drop_incomplete)
if not series:
    st.error(
        "No se han encontrado productos con ventas utilizables. "
        "Revisa el archivo y las columnas."
    )
    st.stop()

if dropped:
    st.caption(f"Se ha ignorado el último {cfg.unit} porque todavía no había terminado.")
if cap_on:
    capped_total = 0
    for name, product_series in series.items():
        values, n_capped = cap_outliers(product_series.to_numpy())
        series[name] = pd.Series(values, index=product_series.index)
        capped_total += n_capped
    if capped_total:
        st.caption(f"Se han limitado {capped_total} picos extremos de ventas.")

assessment = assess_data(series, cfg)
with st.expander(
    f"Comprobación del archivo · calidad del historial: {assessment.level}",
    expanded=False,
):
    for message in report.messages:
        st.warning(message)
    for check in file_checks(report, assessment, cfg):
        st.markdown(f"{ICONS[check.status]} {check.text}")

ranking = sorted(series, key=lambda product_name: series[product_name].sum(), reverse=True)
covered = cover_periods_for(lead_days, extra_days, cfg.days)
eff_horizon = max(horizon, covered)
data_id = hashlib.blake2b(data, digest_size=8).hexdigest()
stock_by_name: dict[str, float] = {}

tab_summary, tab_detail, tab_how = st.tabs(
    ["Resumen y pedido", "Ver un producto", "Ayuda"]
)

# Pestaña principal: primero la recomendación que necesita el usuario.
with tab_summary:
    st.subheader("3 · Elige los productos que quieres revisar")
    max_n = min(30, len(ranking))
    top_n = min(5, max_n)
    if max_n > 5:
        with st.expander("Incluir más productos (por defecto, los 5 más vendidos)"):
            options = list(range(5, max_n + 1, 5))
            if max_n not in options:
                options.append(max_n)
            top_n = st.selectbox(
                "Número de productos",
                options=options,
                index=0,
                format_func=lambda value: f"{value} productos",
            )
    top = ranking[:top_n]

    with st.expander("Añadir el stock que tienes ahora (recomendado)", expanded=False):
        st.write(
            "Indica las unidades disponibles de cada producto. Con este dato la recomendación "
            "calcula cuánto necesitas comprar, en lugar de suponer que no tienes existencias."
        )
        stock_df = st.data_editor(
            pd.DataFrame({"Producto": top, "Unidades disponibles": 0.0}),
            disabled=["Producto"],
            hide_index=True,
            width="stretch",
            key=f"stock_{data_id}_{top_n}",
            column_config={
                "Unidades disponibles": st.column_config.NumberColumn(
                    min_value=0.0,
                    step=1.0,
                    help="Cuántas unidades tienes disponibles ahora mismo.",
                )
            },
        )
    stock_by_name = dict(
        zip(stock_df["Producto"], stock_df["Unidades disponibles"].fillna(0.0))
    )
    stock_provided = any(value > 0 for value in stock_by_name.values())

    with st.spinner("Analizando tus ventas y calculando las recomendaciones…"):
        results = compute_results(
            top, series, stock_by_name, cfg, eff_horizon, covered, service
        )

    if not results:
        st.warning("No hay suficiente historial para generar previsiones de estos productos.")
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
        st.subheader("Tu recomendación de compra")
        if not stock_provided:
            st.info(
                "Todavía no has indicado existencias. Para calcular cuánto comprar de verdad, "
                "abre «Añadir el stock que tienes ahora». Hasta entonces, el pedido supone "
                "que hay 0 unidades disponibles."
            )

        for product_result in results:
            trend = trend_short(
                recent_trend(product_result.history.to_numpy(), cfg.window)
            )
            with st.container(border=True):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown(f"**{product_result.name}**")
                    st.caption(
                        f"Demanda prevista: {fmt(product_result.suggestion.demand, 1)} · "
                        f"Stock indicado: {fmt(product_result.suggestion.on_hand, 1)}"
                    )
                    st.caption(
                        f"Tendencia: {trend} · Fiabilidad estimada: "
                        f"{product_result.forecast.reliability}"
                    )
                with right:
                    st.metric("Comprar aprox.", fmt(product_result.suggestion.order_qty))

        with st.expander("Ver tabla completa y descargar resultados", expanded=False):
            table = pd.DataFrame(
                [
                    {
                        "Producto": item.name,
                        "Tendencia reciente": trend_short(
                            recent_trend(item.history.to_numpy(), cfg.window)
                        ),
                        "Modelo": item.forecast.model,
                        "Fiabilidad": item.forecast.reliability,
                        "Error de validación (%)": round(item.forecast.best_wape * 100)
                        if np.isfinite(item.forecast.best_wape)
                        else None,
                        "Demanda prevista": round(item.suggestion.demand, 1),
                        "Stock de seguridad": round(item.suggestion.safety_stock, 1),
                        "Stock actual": item.suggestion.on_hand,
                        "Pedir": item.suggestion.order_qty,
                    }
                    for item in results
                ]
            )
            visible_columns = [
                "Producto",
                "Demanda prevista",
                "Stock actual",
                "Pedir",
                "Fiabilidad",
            ]
            st.dataframe(
                table[visible_columns],
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Descargar informe completo (CSV)",
                to_csv_bytes(table),
                file_name="resumen_compras.csv",
                mime="text/csv",
            )

        with st.expander("Ver gráficos de ventas y pedido", expanded=False):
            st.plotly_chart(
                overview_chart(results, cfg.unit),
                width="stretch",
                config={"responsive": True, "displaylogo": False},
            )
            st.caption(
                "Se comparan las ventas históricas con la previsión. La banda sombreada representa "
                "el rango probable estimado por el modelo."
            )
            st.plotly_chart(
                order_chart(results),
                width="stretch",
                config={"responsive": True, "displaylogo": False},
            )

        with st.expander("Qué significa el resultado y qué hacer después", expanded=False):
            for point in summary.points:
                st.markdown(f"**{point.title}.** {point.text}")
            st.markdown("**Siguientes pasos**")
            for action_number, action in enumerate(summary.actions, start=1):
                st.markdown(f"{action_number}. {action}")

# Pestaña de detalle individual.
with tab_detail:
    st.write(
        "Selecciona un producto para ver su historial, la previsión y el cálculo del pedido."
    )
    product = st.selectbox(
        "Producto",
        ranking,
        help="Los productos aparecen ordenados de mayor a menor venta total.",
    )
    history = series[product]
    if len(history) < 2:
        st.warning("Este producto tiene muy poco historial para generar una previsión fiable.")
    else:
        if eff_horizon > horizon:
            st.caption(
                f"La previsión cubre {eff_horizon} {cfg.plural} para atender el plazo de entrega "
                "y el margen adicional."
            )
        fc = cached_forecast(
            tuple(history.to_numpy().tolist()), eff_horizon, cfg.season, cfg.window
        )
        future = pd.period_range(start=history.index[-1] + 1, periods=eff_horizon)

        metric_left, metric_right = st.columns(2)
        metric_left.metric("Modelo seleccionado", fc.model)
        metric_right.metric(
            "Error al validar el modelo",
            pct(fc.best_wape),
            help=(
                "Mide el error al predecir ventas históricas que el modelo no había visto. "
                "Menos es mejor."
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

        st.subheader("Calcular el pedido de este producto")
        on_hand = st.number_input(
            "Unidades que tienes disponibles",
            min_value=0.0,
            value=float(stock_by_name.get(product, 0.0)),
            step=1.0,
        )
        suggestion = suggest_order(fc.mean, fc.sigma, covered, service, on_hand)
        result_left, result_right = st.columns(2)
        result_left.metric("Demanda durante la cobertura", fmt(suggestion.demand, 1))
        result_right.metric("Stock de seguridad", fmt(suggestion.safety_stock, 1))
        result_left, result_right = st.columns(2)
        result_left.metric("Stock actual", fmt(on_hand, 1))
        result_right.metric("Comprar aprox.", fmt(suggestion.order_qty))
        st.caption(
            "Cálculo orientativo: demanda prevista + stock de seguridad − unidades disponibles."
        )

        with st.expander("Ver previsión por periodo y descargarla"):
            forecast_table = pd.DataFrame(
                {
                    cfg.label: future.start_time.strftime("%d/%m/%Y"),
                    "Previsión": fc.mean.round(1),
                    "Mínimo probable": fc.lower.round(1),
                    "Máximo probable": fc.upper.round(1),
                }
            )
            st.dataframe(forecast_table, hide_index=True, width="stretch")
            st.download_button(
                "Descargar previsión CSV",
                to_csv_bytes(forecast_table.assign(Producto=product)),
                file_name="prevision.csv",
                mime="text/csv",
            )

        with st.expander("Cómo se ha calculado esta previsión"):
            ranking_models = pd.DataFrame(
                {
                    "Modelo": list(fc.scores),
                    "Error de validación": list(fc.scores.values()),
                }
            ).sort_values("Error de validación")
            ranking_models["Error de validación"] = ranking_models["Error de validación"].map(pct)
            st.dataframe(ranking_models, hide_index=True, width="stretch")
            st.write(
                "Se prueban varios métodos con ventas históricas y se selecciona el que comete "
                "menos error en la validación. La fiabilidad no es una garantía de acierto futuro."
            )

# Pestaña de ayuda en lenguaje sencillo.
with tab_how:
    st.subheader("Cómo utilizar la aplicación")
    st.markdown(
        """
1. **Empieza con los datos de ejemplo** para ver cómo funciona, o elige **Subir mis ventas (CSV)**.
2. **Revisa las columnas** si el archivo usa nombres poco habituales: fecha, producto y cantidad.
3. **Añade tu stock actual** en la pestaña «Resumen y pedido» para que el programa
   descuente lo que ya tienes.
4. **Consulta «Comprar aprox.»** para ver una recomendación de unidades por producto.
5. **Descarga el CSV** si quieres guardar el informe o compartirlo.

**Consejo:** si no entiendes un ajuste, deja el valor inicial.
Los ajustes avanzados no son necesarios para probar la aplicación.
        """
    )
    with st.expander("Formato del archivo CSV"):
        show_requirements()
    with st.expander("Qué significan los términos técnicos"):
        st.markdown(
            "- **Demanda prevista:** cantidad que el modelo estima que venderás en los periodos "
            "futuros.\n"
            "- **Stock de seguridad:** margen adicional para reducir el riesgo de quedarte sin "
            "producto.\n"
            "- **Error de validación (WAPE):** medida de cuánto se equivocó el modelo al predecir "
            "ventas históricas que no utilizó para ajustar esa predicción.\n"
            "- **Fiabilidad estimada:** etiqueta orientativa basada en el historial y el error; "
            "no es una garantía estadística."
        )

st.divider()
st.caption(
    f"Predictor de demanda · Versión {__version__} · "
    "Previsiones orientativas, no garantías."
)



