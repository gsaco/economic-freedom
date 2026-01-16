# Evaluacion de hipotesis (reformas EFW y dinamica quinquenal)

## Hipotesis principales
- H1. Las reformas positivas de EFW aumentan el crecimiento quinquenal del GDPpc en el corto plazo.
- H2. Las reformas negativas tienen efectos contractivos mas fuertes que los positivos (asimetria).
- H3. Las reformas positivas reducen la probabilidad de colapso de crecimiento (downside risk).

## Evidencia clave (stacked event study)
- Positivas: coeficiente en t=0 = 0.064 (p=0.006). Ver `outputs/tables/stacked_event_irf_pos_gdppc_growth_5y.csv`.
- Negativas: coeficiente en t=0 = -0.099 (p=0.230). Ver `outputs/tables/stacked_event_irf_neg_gdppc_growth_5y.csv`.
- Robustez post (positivas): wild bootstrap p=0.007, two-way cluster p=0.028, placebo timing p=0.035.
- Pretrends: p-valores conjuntos pos=0.866, neg=0.793 ver `outputs/tables/stacked_event_pretrend_tests.csv`.
- Downside risk: colapso en t=0 = -0.078 (p=0.228). Ver `outputs/tables/stacked_event_irf_pos_growth_collapse.csv`.

## Evaluacion H1-H3
- H1: Apoyada (efecto positivo contemporaneo y robusto).
- H2: Inconclusa (signo negativo pero baja precision).
- H3: No apoyada (efectos de colapso no significativos).

## Proximos pasos
- Implementar SE de dos vias por horizonte y placebo outcomes (ver `docs/08_q1_must_do_plan.md`).
- Revisar balance pre-tratamiento y considerar ajustes de matching/pesos.
