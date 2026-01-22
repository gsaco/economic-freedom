# Ventanas post más largas (t+1..t+5, t+1..t+7)

## Qué se hizo
Se redefinió el shock de EFW como diferencia entre promedios pre y post usando ventanas más largas:
- **Post‑5**: promedio t+1..t+5
- **Post‑7**: promedio t+1..t+7

## Primer estadio (Z → Shock EFW)
- **Positivos**:
  - Post‑5: coef ≈ 0.300, p ≈ 0.063 (mejora vs baseline).
  - Post‑7: coef ≈ 0.199, p ≈ 0.154.
- **Negativos**:
  - Post‑5: coef ≈ 0.041, p ≈ 0.757 (sin mejora).
  - Post‑7: coef ≈ 0.050, p ≈ 0.681.

## IRFs IV
- Los coeficientes siguen siendo imprecisos; no hay significancia robusta.
- El único avance es un **primer estadio más fuerte** en positivos con Post‑5, pero aún marginal.

## Interpretación
Las ventanas post más largas **mejoran parcialmente** el primer estadio en el caso positivo (p≈0.06), pero no resuelven el problema en negativos ni generan IRFs precisas. Es una mejora limitada, no concluyente.
