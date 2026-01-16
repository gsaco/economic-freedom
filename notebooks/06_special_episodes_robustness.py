# %% [markdown]
# Special episodes robustness

# %%
import pandas as pd

from src.config import TABLES_DIR

# %%
interactions = pd.read_csv(TABLES_DIR / "episode_interactions.csv")
interactions.head()
