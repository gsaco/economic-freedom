# DPI (Database of Political Institutions) — intento de integración

## Qué se hizo
- Se descargó **DPI 2012** desde el World Bank Data Catalog.
- Se inspeccionaron variables disponibles (orientación del ejecutivo, composición legislativa, etc.).

## Resultado
- DPI **no incluye votos ni márgenes electorales**.
- Sirve para codificar orientación ideológica del ejecutivo, pero **no permite construir un running variable RD**.

## Interpretación
DPI puede usarse como **fuente auxiliar de ideología** si se combina con otra base que tenga **vote shares**. Por sí sola no permite ejecutar el diseño de elecciones cerradas.
