# Economic Freedom (EFW) quinquennial pipeline

## How to run
1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the notebook pipeline:
   - `jupytext --to ipynb notebooks/01_efw_macro_quinquennial_analysis.py`
   - `jupyter notebook notebooks/01_efw_macro_quinquennial_analysis.ipynb`

Outputs are written to `data/processed/`, `outputs/figures/`, `outputs/tables/`, and `reports/`.
