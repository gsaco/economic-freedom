# CLEA (Constituency‑Level Elections Archive) — intento de descarga

## Qué se intentó
- Descargar CLEA (lower chamber) desde ElectionDataArchive.

## Resultado
- El sitio está protegido con **Cloudflare**, y el download automático fue bloqueado.
- Sin acceso programático, no se pudo integrar CLEA en el pipeline.

## Interpretación
CLEA sigue siendo la **mejor vía para aumentar potencia**, pero requiere:
- descarga manual (browser) o
- un mirror público (p. ej., Dataverse/GESIS) sin protección.

Si autorizás, puedo incorporar CLEA manualmente a `data/01_raw/clea/` y ejecutar el pipeline con márgenes distritales.
