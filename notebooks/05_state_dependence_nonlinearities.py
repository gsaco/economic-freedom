# %% [markdown]
# State dependence and nonlinearities

# %%
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TABLES_DIR
from src.viz.figures import fig_state_dependence_by_baseline_efw

# %%
_ = fig_state_dependence_by_baseline_efw()
plt.show()

# %%
path = TABLES_DIR / "heterogeneity_state_dependence.csv"
if path.exists() and path.stat().st_size > 1:
    heterogeneity = pd.read_csv(path)
    heterogeneity.head()
else:
    "Heterogeneity table not available."
