# %% [markdown]
# Build quinquennial data
#
# This notebook runs ingestion and build steps and inspects the resulting panel.

# %%
import pandas as pd

from src.ingest.efw import ingest_efw
from src.ingest.wdi import ingest_wdi
from src.ingest.pwt import ingest_pwt
from src.ingest.eu_events import ingest_eu_events
from src.ingest.wto_events import ingest_wto_events
from src.ingest.wto_acdb import ingest_wto_acdb
from src.build.outcomes import build_macro_outcomes
from src.build.exposure import build_exposures
from src.build.reforms import build_reforms
from src.build.panel import build_panel
from src.config import PROCESSED_DIR

# %%
_ = ingest_efw()
_ = ingest_wdi()
_ = ingest_pwt()
_ = ingest_eu_events()
_ = ingest_wto_events()
_ = ingest_wto_acdb()

# %%
_ = build_macro_outcomes()
_ = build_exposures()
_ = build_reforms()
_ = build_panel()

# %%
panel = pd.read_parquet(PROCESSED_DIR / "panel_quinquennial.parquet")
panel.head()
