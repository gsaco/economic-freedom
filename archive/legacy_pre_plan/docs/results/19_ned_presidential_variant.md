# NED presidencial (vote shares) con mapeo a ParlGov

## Qué se hizo
- Se descargó el **National Elections Database v2** y se usó la base **presidential** (vote shares).  
- Se mapearon partidos a ideología usando ParlGov por **match de nombre** (normalización simple).  
- Se construyó un running variable **market vs non‑market** con votos de ronda 1 y se aplicó el diseño RD‑IV.

## Resultado operativo
- La intersección **NED + ParlGov (match de partidos)** fue extremadamente pequeña.
- **Muestra efectiva**: 2 elecciones, 1 país, negativos = 0.
- **Primer estadio / IRFs**: NaN por falta de observaciones en ventana válida.

## Interpretación
Esta alternativa **no es usable** con un mapeo básico de nombres. Se necesitaría un **match de partidos más sofisticado** (fuzzy + diccionarios manuales) o una fuente de ideología que ya venga **codificada en NED** para ampliar cobertura.
