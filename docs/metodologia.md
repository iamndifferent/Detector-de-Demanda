# Metodología

## 1. Preparación de los datos
- Las fechas ISO (`2024-03-05`) se leen siempre como año-mes-día; las demás (`05/03/2024`) siguen
  la preferencia día/mes/año del usuario.
- Cantidades con coma se interpretan como formato europeo (`1.234,5`).
- Las devoluciones (cantidades negativas) se tratan como 0, no se descuentan.
- Los periodos sin ventas valen 0; cada producto empieza en su primera venta; el último periodo
  se descarta si aún no terminó.

## 1 bis. Verificación del archivo
Antes de prever se evalúa cuánto historial hay (periodos del producto con más datos):
**Insuficiente** (< 8), **Justo** (8 hasta el recomendado: 60 días, 26 semanas o 24 meses) y
**Suficiente**. También se comprueba qué porcentaje de filas es válido y si hay al menos dos
ciclos completos (7, 52 o 12 periodos) para poder detectar temporadas.

## 2. Modelos candidatos
| Modelo | Idea | Cuándo se prueba |
|---|---|---|
| Media móvil (referencia) | Promedio de las últimas ventas | Siempre |
| Último valor | Repite el último periodo | Siempre |
| Estacional ingenuo | Repite el mismo periodo del ciclo anterior | Historial > 1 ciclo |
| Suavizado exponencial | Nivel que se adapta con el tiempo | ≥ 8 periodos |
| Suavizado con tendencia | Nivel + tendencia amortiguada (Holt) | ≥ 8 periodos |
| Suavizado estacional | Nivel + patrón repetitivo (Holt-Winters aditivo) | ≥ 2 ciclos |

Los tres suavizados están implementados en `forecaster/smoothing.py` con NumPy y SciPy
(minimizando el error cuadrático). Ciclo = 7 (diario), 52 (semanal) o 12 (mensual).

## 3. Validación
Validación de origen móvil: se entrena con el historial hasta un punto, se predicen los
siguientes periodos y se compara con lo vendido; se repite hasta 3 veces. La métrica es el
**WAPE** (suma de errores absolutos / suma de ventas), que tolera periodos con 0 ventas. Gana
el modelo con menor WAPE; si empatan, el más simple.

## 4. Rango probable
`previsión ± 1,28 · σ` (≈ 80 %), con σ la desviación típica del error a un paso del modelo
elegido. Es una aproximación: asume errores simétricos y de varianza constante.

## 5. Sugerencia de compra
`pedido = ventas previstas en (plazo + días extra) + z · σ · √periodos − stock actual`,
con `z` del nivel de servicio (95 % → 1,64). Redondeado hacia arriba y nunca negativo.

## Limitaciones
- Ventas ≠ demanda: si hubo roturas de stock, las ventas pasadas la subestiman.
- No incluye precios, promociones futuras, festivos ni variables externas.
- Con menos de dos ciclos completos no se puede aprender estacionalidad.

## 6. Resumen en lenguaje llano
`forecaster/summary.py` convierte los resultados en frases y acciones: tendencia reciente
(últimos N periodos frente a los N anteriores, ±10 % = estable), previsión frente al periodo
anterior, cantidades a pedir, fiabilidad por producto (Alta ≤ 25 % de error, Media ≤ 50 %) y
una lista de acciones. Un producto se marca como prioritario si su stock actual es menor que la
demanda prevista durante el plazo de entrega. Los nombres de producto se escapan para que no
puedan inyectar formato (enlaces, negritas) en el texto.
