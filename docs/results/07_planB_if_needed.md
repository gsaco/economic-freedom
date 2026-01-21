# QA and Reproducibility

Quality gates enforced by tests:
- No inference imports in executed notebooks/modules.
- Unique key check for (iso3c, year).
- Year grid strictly matches quinquennial years.
- Metadata JSON files exist for intermediate and final datasets.
- Figures exported as PNG (500 dpi) and PDF.

Run ledger:
- `output/logs/run_ledger.jsonl`

Environment snapshots:
- `output/logs/env_snapshot.json`
- `output/logs/run_config.json`
