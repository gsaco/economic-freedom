# Postmortem RD con seat shares

## Resumen
- El running variable basado en seat share muestra discontinuidad de densidad y desbalances pre‑tratamiento relevantes.
- Esto invalida la interpretación causal estándar del RD con seat shares.

## Evidencia clave
- **Discontinuidad de densidad**: log‑diff ≈ 0.598 (más masa a la derecha del cutoff).
- **Balance pre‑tratamiento**:
  - `lag1_trade_open_gdp`: coef ≈ 39.27, p ≈ 0.005 (desbalance fuerte).
  - `lag1_efw_summary`: coef ≈ −0.34, p ≈ 0.063 (desbalance marginal).

## Interpretación
El running variable con seat shares exhibe heaping mecánico y discontinuidades que se trasladan a covariables pre‑elección (en particular trade openness). Esto rompe el supuesto de continuidad y confirma la necesidad de migrar a vote shares para el diseño RD.
