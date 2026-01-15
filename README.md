# Economic Freedom (EFW) quinquennial pipeline

## How to run
1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the full pipeline (data + notebooks):
   - `python tools/run_pipeline.py`

Notes:
- The pipeline downloads and caches raw data in `data/raw/` with `.meta` files.
- Processed panels go to `data/processed/`.
- Figures/tables go to `outputs/figures/` and `outputs/tables/`.
- Reports live in `reports/`.
- If you only want to sync notebooks without executing, use `python tools/run_pipeline.py --skip-exec`.
