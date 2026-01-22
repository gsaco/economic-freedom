# Variante “top‑2 margin” (bloc winner vs runner‑up)

## Qué se hizo
Se construyó el running variable como la diferencia entre el **mejor partido market** y el **mejor partido non‑market** en voto (top‑2 por bloque). Se re‑estimaron validez RD y LP‑IV usando `running_var_top2`.

## Validez RD
- **Densidad**: sin salto fuerte (pos: z≈0.00; neg: z≈0.95).
- **Balance**:
  - Positivos: pasa el umbral p>0.15 en **±0.05**.
  - Negativos: **no** hay ventana que pase p>0.15; se usa fallback **±0.03**.

## Primer estadio (Z → Shock EFW)
- **Positivos**: coef ≈ −0.196, p ≈ 0.279 (N≈27).
- **Negativos**: coef ≈ 0.102, p ≈ 0.579 (N≈23).

## IRFs IV (Shock EFW → log PIBpc)
- Coeficientes pequeños, imprecisos y sin significancia; varios horizontes cambian de signo.

## Comparación con baseline (vote‑margin)
- El primer estadio **no mejora**: la precisión es similar y el coeficiente en positivos cambia de signo.
- La muestra negativa sigue sin ventana “limpia” con p>0.15.

## Interpretación
La variante top‑2 margin **no mejora** la potencia ni el primer estadio. De hecho, en positivos el primer estadio invierte signo y la validez en negativos no alcanza el umbral de balance. En suma, **no hay evidencia de mejora** respecto del diseño con vote‑margin.
