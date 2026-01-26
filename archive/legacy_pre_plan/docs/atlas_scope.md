# Descriptive Atlas Scope

This project now produces a descriptive quinquennial atlas of Economic Freedom of the World (EFW) and macro indicators.
The pipeline excludes causal inference, regression models, event studies, and synthetic control methods.

Rules
- Descriptive only: report levels, changes, distributions, ranks, co-movements, and coverage.
- No inference imports in executed modules or notebooks (e.g., statsmodels, linearmodels).
- Every data artifact must include metadata with sources, transforms, and coverage.
- Maps and figures must export PNG (500 dpi) and PDF.
- Join keys are strictly validated on (iso3c, year).

Legacy inference code is archived under `archive/inference_deprecated_v1/`.
