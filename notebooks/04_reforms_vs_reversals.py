# %% [markdown]
# Reforms vs reversals

# %%
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DIR
from src.viz.figures import fig_eu_sdid_reforms_vs_reversals_wave2004

# %%
_ = fig_eu_sdid_reforms_vs_reversals_wave2004()
plt.show()

# %%
panel = pd.read_parquet(PROCESSED_DIR / "panel_quinquennial.parquet")
summary = panel.groupby("year")[["reform_event_03", "reversal_event_03"]].mean().reset_index()
summary.head()
