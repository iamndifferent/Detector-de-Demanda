# Modelo de amenazas

Alcance: la app web (Streamlit) que recibe un CSV de ventas, lo procesa en memoria y muestra
resultados descargables. No tiene cuentas, base de datos ni llamadas a servicios externos.

## Qué hay que proteger

- **Los datos de ventas del usuario** (confidenciales para su negocio).
- **La disponibilidad de la app** (un archivo malicioso no debe tumbarla).
- **El equipo de quien descarga los resultados** (los CSV no deben ejecutar nada al abrirse).

## Amenazas y mitigaciones

| Amenaza | Mitigación en el código |
|---|---|
| Archivo gigante para agotar memoria o CPU | Límite de 5 MB (`maxUploadSize` y `MAX_FILE_BYTES`), máximo 200.000 filas y 2.000 productos, comprobados antes de procesar. |
| CSV malformado o con codificación rara | Lectura como texto, UTF-8 con respaldo Latin-1, errores convertidos en mensajes claros (`DataError`) en lugar de excepciones. |
| Inyección de fórmulas al abrir el CSV exportado en Excel | `export.sanitize_cell` antepone `'` a celdas de texto que empiezan por `=`, `+`, `-` o `@`. Con test. |
| Cantidades o fechas manipuladas para falsear resultados | Conversión estricta a número y fecha; filas inválidas descartadas y contabilizadas en el informe de limpieza. |
| Nombres de producto con markdown o enlaces (`[clic](http://…)`) que aparecerían en el resumen | `formatting.md_escape` neutraliza `[ ] * _ # | < > ~ $` antes de mostrar texto del usuario. Con test. |
| Previsiones usadas como certeza | Rango probable, error de validación, nivel de fiabilidad y avisos visibles en pantalla. |
| Fuga de datos por el uso de servidores de terceros | Aviso de privacidad en la interfaz y alternativa de ejecución local (Docker o Codespaces privado). La app no escribe los archivos en disco. |
| Dependencias vulnerables | Dependencias mínimas (5), versiones mínimas fijadas y CI en cada cambio. Recomendado activar Dependabot. |
| Contenedor con demasiados privilegios | La imagen Docker se ejecuta con un usuario sin privilegios. |

## Riesgos residuales y decisiones conscientes

- En Codespaces se desactivan CORS y la protección XSRF de Streamlit porque el proxy del
  puerto reenviado rompe la subida de archivos si están activas. Es aceptable porque el puerto
  es **privado por defecto**; no lo hagas público con datos reales. En despliegues normales
  (Docker, Streamlit Community Cloud) esas protecciones siguen activadas.
- La app aloja datos de usuarios solo en memoria durante la sesión, pero no controla lo que
  haga la plataforma de alojamiento con ella.
