# Variante “switch‑only” (winner_market ≠ incumbent_market)

## Qué se hizo
Se restringió el análisis a elecciones donde el gabinete cambia de orientación económica (`market_switch = 1`) y se re‑estimaron validez RD e IRFs IV.

## Validez RD
- Densidad sin saltos fuertes, pero **tamaños extremadamente pequeños** (≈8–12 obs en ventanas válidas).
- Positivos: p‑values de balance **fallan** en casi todas las ventanas; negativos solo pasan a partir de ±0.03.

## Primer estadio
- **Positivos**: coef ≈ 2.54 con `se = inf` (problema numérico por N muy bajo).
- **Negativos**: coef ≈ 0.34, p ≈ 0.17 (débil).

## IRFs
- Varias estimaciones degeneradas (SE ~ 0 o infinitas) por colapso de la muestra.

## Interpretación
Esta restricción **no funciona** con los datos actuales: reduce la muestra a un nivel en el que los estimadores colapsan. No es una vía útil sin aumentar potencia (más elecciones / CLEA).
