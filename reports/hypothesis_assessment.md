# Evaluación preliminar de hipótesis (EFW shocks y crecimiento)

## Hipótesis iniciales
- H1. ¿Cuál es la respuesta dinámica del crecimiento del PBI real per cápita ante choques discretos positivos y negativos en el índice EFW?
- H2. ¿Los choques negativos generan efectos más rápidos, intensos y persistentes que choques positivos comparables?
- H3. ¿La respuesta dinámica es no lineal según (i) magnitud del choque y (ii) estado inicial del país (nivel EFW e ingreso)?

## Evidencia descriptiva y patrones (con cifras)
- Event-study (q90): crecimiento medio en t=0: positivo=0.120, negativo=0.006; en t=5: positivo=0.091, negativo=0.034 (ver `outputs/figures/event_study_growth.png` y `outputs/tables/event_study_growth.csv`).
- LP lineal h=0: 0.066 (p=0.000); h=5: 0.018 (p=0.213) (ver `outputs/figures/lp_irf_linear.png`).
- LP asimétrico h=10: ΔEFW+ -0.005 (p=0.808) vs ΔEFW- 0.047 (p=0.085) (ver `outputs/figures/lp_irf_asymmetric.png`).
- LP magnitud h=5: ΔEFW 0.016 (p=0.371) y ΔEFW*Large 0.004 (p=0.834) (ver `outputs/figures/lp_irf_magnitude.png`).
- La comparación WB vs PWT muestra alta correlación agregada y colas en el log ratio (ver `outputs/figures/gdppc_wb_pwt_log_ratio.png` y `outputs/tables/gdppc_wb_pwt_corr_overall.csv`).

## Evaluación de H1–H3 (provisional)
- H1: El event-study muestra brechas visibles entre shocks positivos y negativos; el LP lineal sugiere respuesta contemporánea positiva (h=0) y efectos más débiles en h=5.
- H2: La especificación asimétrica en h=10 muestra coeficientes distintos para ΔEFW- vs ΔEFW+, con señal adversa más marcada en los negativos (recordar que ΔEFW- es negativo, por lo que un coeficiente positivo implica efecto negativo).
- H3: En la especificación de magnitud, el término ΔEFW*Large es pequeño y no significativo en h=5; la interacción con estados iniciales sugiere heterogeneidad pero con baja precisión en este prototipo.

## Hipótesis revisadas / mejoradas
- H1R: Los shocks negativos discretos en EFW reducen el crecimiento quinquenal con mayor intensidad en países con baja EFW inicial, mientras que shocks positivos muestran efectos más graduales.
- H2R: La asimetría en IRFs se concentra en horizontes medios (h=10), con respuestas negativas más persistentes en regiones con menor apertura comercial.
- H3R: La magnitud del shock interactúa con el nivel inicial de ingreso: shocks grandes en países de ingreso medio muestran mayor volatilidad posterior que en extremos de ingreso.

## Implicancias para próximos modelos LP
- Incluir 1–2 rezagos adicionales del crecimiento y EFW para capturar persistencia.
- Probar FE de país + tiempo y variantes con tendencias lineales por país.
- Definir shocks negativos usando cuantiles por año y verificar robustez con umbral absoluto.
- Estimar efectos separados por regiones WB e ingreso inicial (interacciones o submuestras).
