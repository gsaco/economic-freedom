# Robustness and Placebos

Placebo timing (shifted treatment) and donor restrictions are executed via tagged estimator runs.

Outputs:
- `outputs/tables/estimates_eu_placebo_timing_*.parquet`
- `outputs/tables/estimates_wto_placebo_timing_*.parquet`

If additional placebo-in-space or leave-one-out tests are needed, see `src/analysis/robustness.py` for extension points.


References:
- Economic Freedom of the World (Fraser Institute) documentation
- Arkhangelsky et al. (2021) Synth-DID
- Callaway & Sant'Anna (2021) DID; Sun & Abraham (2021) event studies
- Ben-Michael et al. (2021) Augmented SCM
- Schimmelfennig & Sedelmeier (2004) EU conditionality
- EU Commission enlargement process documentation
- Tang & Wei (2009) WTO accessions
- Brotto (IMF WP 2024) WTO accession impacts
- Chemutai & Escaith (2017) WTO commitments measurement
