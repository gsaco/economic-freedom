# Replication package checklist

## One-command rebuild
- Command: `python tools/run_all.py` (runs jupytext sync, executes notebooks, mirrors outputs, builds report).
- Pass criteria: `output/report.md` lists all figures/tables/logs; `output/logs/run_all.json` populated.

## Data provenance + licensing
- Raw data cached in `data/raw/` with `.meta` files documenting URL, download date, and variable notes.
- Key sources (check licensing in `.meta`): EFW (Fraser), WDI (World Bank), PWT, SWIID, Polity/V-Dem, Laeven–Valencia crises.
- Pass criteria: each raw dataset has a corresponding `.meta` file.

## Environment pinning
- Python deps pinned in `requirements.txt`.
- Pass criteria: clean environment can install deps and run `python tools/run_all.py`.

## Seeds + configs logged
- Event definitions: `output/logs/stacked_event_config.json`.
- Inference settings: `output/logs/stacked_event_inference_config.json`.
- Spec ledger (model grid): `outputs/spec_ledger.csv`.

## Paper integration
- Figures/tables live in `outputs/` and are mirrored to `output/` for inspection.
- Pass criteria: `output/report.md` enumerates all outputs; paper drafts cite figure/table paths.

## Appendix benchmark (Rondón 2025)
- Replicate thesis-style specs as a labeled appendix section only (no new claims).
- Pass criteria: appendix outputs clearly labeled “benchmark replication” and separated from flagship results.
