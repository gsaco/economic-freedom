# %% [markdown]
# Appendix: EFW components and diagnostics

# %%
import pandas as pd

from src.config import PROCESSED_DIR, DIAGNOSTICS_DIR

# %%
ef = pd.read_parquet(PROCESSED_DIR / "efw_quinquennial.parquet")
components = [c for c in ef.columns if c.startswith("efw_area")]
correlations = ef[["efw_overall"] + components].corr()
correlations

# %%
try:
    diag = pd.read_csv(DIAGNOSTICS_DIR / "sdid_eu_diag_baseline.csv")
    diag.head()
except FileNotFoundError:
    "Diagnostics not found yet."
