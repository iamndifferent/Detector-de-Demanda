# 📈 Predictor de demanda para negocios pequeños

[![CI](https://github.com/iamndifferent/demand-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/iamndifferent/Detector-de-Demanda/actions/workflows/ci.yml)
[![Abrir en Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/iamndifferent/Detector-de-Demanda)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detector-de-demanda.streamlit.app/)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-green.svg)](LICENSE)

Sube un CSV con tus ventas y la app te dice **cuánto venderás** en las próximas semanas y
**cuánto te conviene pedir** al proveedor. Está pensada para comercios, cafeterías, vendedores
online y artesanos que hoy deciden sus compras "a ojo" o con una hoja de cálculo.

> **Demo en vivo:** `https://detector-de-demanda.streamlit.app/` · incluye datos de ejemplo para probarla sin
> subir nada.

## Qué hace

- **Te dice qué necesita tu archivo antes de subirlo**: formato y cantidad mínima de datos.
- **Verifica tu archivo** al subirlo (✅ ⚠️ ❌) y te explica qué corrigió: fechas ilegibles,
  comas decimales, devoluciones, duplicados y picos extremos.
- **Prevé la demanda** por producto (diaria, semanal o mensual) con un rango probable del 80 %.
- **Elige el mejor modelo solo**: prueba 6 métodos contra tus ventas recientes y se queda con el
  que menos se equivoca, comparándolo con la media móvil.
- **Calcula el pedido**: ventas previstas durante el plazo de entrega + stock de seguridad −
  stock actual, según el nivel de servicio que quieras.
- **Avisa cuando no se fía**: poco historial, demanda intermitente o error de validación alto.
- **Resume en lenguaje llano** qué analizó y **qué hacer ahora** (qué pedir, qué vigilar, cuándo
  volver a subir datos).
- **Dibuja gráficos con tus datos**: ventas reales, previsión y rango probable de tus productos
  principales, y cantidades a pedir.
- **Exporta** previsiones y resumen de compras a CSV (con protección frente a inyección de
  fórmulas en Excel).

## Pruébala

### Opción 1 · GitHub Codespaces (sin instalar nada)
1. Pulsa **Abrir en Codespaces** arriba (o *Code → Codespaces → Create*).
2. Espera a que termine la instalación; la app se abre sola en el puerto 8501.

### Opción 2 · Docker
```bash
git clone https://github.com/iamndifferent/Detector-de-Demanda.git
cd Detector-de-Demanda
docker compose up --build   # abre http://localhost:8501
```

### Opción 3 · Python en local
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Opción 4 · Publicarla tú (gratis)
En [share.streamlit.io](https://share.streamlit.io) elige este repositorio y el archivo
`streamlit_app.py`. No necesita variables secretas.

## Requisitos del archivo

**Formato**
- Un archivo **CSV** de hasta **5 MB**, con un máximo de 200.000 filas y 2.000 productos.
- Primera fila con los nombres de las columnas. Tres columnas: **fecha, producto y cantidad**
  (una fila por venta, o por día y producto; el orden da igual).
- Fechas como `2025-03-01` o `01/03/2025` (día/mes/año). Cantidades numéricas, con punto o coma
  decimal, sin unidades ni símbolos. Productos escritos siempre igual.
- Separador `,` `;` o tabulador; codificación UTF-8 o Latin-1 (se detectan solas).
- Evita filas de totales, celdas combinadas y varias tablas en una hoja. Desde Excel:
  *Guardar como → CSV UTF-8*.

```csv
fecha;producto;cantidad
03/03/2025;Café molido (kg);2,2
03/03/2025;Croissant;31
```

**Cuántos datos hacen falta**

| Agrupación | Mínimo para prever | Recomendado | Para detectar temporadas |
|---|---|---|---|
| Diaria | 8 días | 60 días | 14 días |
| Semanal | 8 semanas | 26 semanas | 104 semanas |
| Mensual | 8 meses | 24 meses | 24 meses |

Con menos del mínimo la app avisa de que las cifras son solo orientativas. Hay una plantilla
descargable en la propia app y dos ejemplos en [`data/sample/`](data/sample/):
`ventas_ejemplo.csv` (datos limpios de 3 años) y `ejemplo_cafeteria.csv` (con 4 errores a
propósito: fecha imposible, devolución, producto vacío y cantidad no numérica).

## Qué informe obtienes

Con `ejemplo_cafeteria.csv` (agrupado por semana, 4 semanas de previsión, 7 días de entrega,
7 de margen y 95 % de nivel de servicio), la app muestra:

**1. Verificación del archivo**
- ✅ Archivo leído: 581 filas y 3 productos, del 03/03/2025 al 14/09/2025.
- ✅ Formato correcto: casi todas las filas son válidas.
- ✅ Historial suficiente: 28 semanas (recomendado: 26).
- ✅ Todos los productos tienen historial para poder prever.
- ℹ️ Aún no hay historial para detectar temporadas: harían falta 104 semanas.

**2. Resumen** (extracto)
> Buenas noticias: las previsiones de los 3 productos analizados son fiables.
>
> **Cómo van tus ventas.** Comparando el periodo reciente (4 semanas) con el anterior:
> Croissant se mantiene estable; Té verde baja un 33%; Café molido (kg) se mantiene estable.
>
> **Cuánto pedir.** Para cubrir 7 días de entrega más 7 de margen, con un nivel de servicio
> del 95%, te sugerimos pedir: Croissant (376), Café molido (kg) (34), Té verde (28).

**3. Qué hacer ahora**
1. Revisa el pedido sugerido y ajústalo con tu stock real antes de comprar.
2. Cuando reúnas 104 semanas de datos, la app podrá detectar temporadas.
3. Vuelve a subir tus ventas cada semana para actualizar las cifras.
4. Descarga la tabla de compras (CSV) y úsala de base para tu pedido.

**4. Gráficos**: un panel por producto con ventas reales, previsión y rango probable, y barras
con la cantidad a pedir de cada uno.

## Cómo funciona

```
CSV → lectura segura → limpieza → series por producto → verificación de datos
                                                                  ↓
              6 modelos → validación → previsión → rango probable + sugerencia de compra
                                                                  ↓
                                          resumen en lenguaje llano + gráficos
```

Detalle en [`docs/metodologia.md`](docs/metodologia.md). Resumen de las decisiones técnicas:

- **Suavizado exponencial propio** (nivel, tendencia amortiguada y Holt-Winters aditivo) con
  NumPy y SciPy, en lugar de añadir `statsmodels`: instalación ligera para Codespaces y
  Streamlit Cloud, y el código es legible y está probado.
- **Validación de origen móvil y WAPE**: el modelo no se elige por cómo ajusta el pasado, sino
  por cómo habría predicho tramos recientes que no vio. El WAPE aguanta periodos con 0 ventas.
- **Siempre contra una referencia**: se muestra cuánto mejora (o no) respecto a la media móvil.
- **Fallos amables**: los problemas con los datos del usuario son errores con mensaje claro
  (`DataError`), no trazas de Python.

## Límites (léelos antes de fiarte de un número)

- **Ventas no son demanda**: si te quedaste sin stock, las ventas pasadas subestiman lo que
  realmente se habría vendido.
- Con **menos de dos ciclos** (p. ej. dos años en semanal) no se puede aprender estacionalidad.
- No conoce **promociones, precios ni festivos futuros**.
- El rango probable es una aproximación estadística, no una garantía.

## Seguridad y privacidad

El código no guarda los archivos subidos y limita tamaño, filas y productos. Modelo de amenazas
en [`docs/threat-model.md`](docs/threat-model.md); cómo reportar una vulnerabilidad en
[`SECURITY.md`](SECURITY.md). En versiones alojadas por terceros, no subas datos personales.

## Estructura

```
streamlit_app.py        interfaz
forecaster/
  loader.py             lectura segura del CSV y detección de columnas
  cleaning.py           limpieza, informe y series por producto
  readiness.py          cuántos datos hay y comprobaciones del archivo (✅ ⚠️ ❌)
  summary.py            resumen en lenguaje llano y acciones recomendadas
  charts.py             gráficos con Plotly
  formatting.py         números en formato español y escape de texto del usuario
  smoothing.py          suavizado exponencial (NumPy/SciPy)
  models.py             candidatos, validación y selección del modelo
  evaluation.py         WAPE y validación de origen móvil
  replenishment.py      stock de seguridad y cantidad a pedir
  export.py             CSV seguro
  samples.py            datos sintéticos de ejemplo (limpios y con errores a propósito)
tests/                  pruebas automáticas (pytest)
docs/                   metodología y modelo de amenazas
```

## Desarrollo

```bash
pip install -r requirements-dev.txt
pytest -q        # pruebas
ruff check .     # estilo
```

La integración continua ejecuta ambos en Python 3.11 y 3.12 en cada cambio.

## Ideas para el futuro

- Calendario de festivos y marcado de promociones como variables conocidas.
- Detección de roturas de stock para corregir la demanda censurada.
- Conexión con el gestor de inventario y el análisis de marketing del resto del portafolio.

## Licencia

[MIT](LICENSE)
