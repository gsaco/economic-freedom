# %% [markdown]
# # Environment setup
# This notebook captures the local environment snapshot and the configuration
# used throughout the analysis pipeline.
#
# It records package versions, the quinquennial grid, and the World Bank
# indicator/episode definitions that drive downstream notebooks.

# %%
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import LOGS_DIR, QUINQUENNIAL_YEARS, ensure_directories
from src.viz_style import set_style

# %%
ensure_directories()
set_style()
np.random.seed(42)

# %%
packages = [
    "pandas",
    "numpy",
    "pyarrow",
    "matplotlib",
    "requests",
    "geopandas",
    "shapely",
]
versions: dict[str, str] = {}
for pkg in packages:
    try:
        versions[pkg] = metadata.version(pkg)
    except metadata.PackageNotFoundError:
        versions[pkg] = "not-installed"

snapshot = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": versions,
}

LOGS_DIR.mkdir(parents=True, exist_ok=True)
(Path(LOGS_DIR) / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))

versions_df = (
    pd.DataFrame([{"package": name, "version": version} for name, version in versions.items()])
    .sort_values("package")
    .reset_index(drop=True)
)
display(versions_df.style.set_caption("Package versions"))
missing_packages = versions_df.loc[versions_df["version"] == "not-installed", "package"].tolist()
if missing_packages:
    display(pd.DataFrame({"missing_packages": missing_packages}))

# %%
WDI_INDICATORS = {
    "NY.GDP.PCAP.KD": {
        "name": "gdp_pc_const",
        "transform": "log",
        "label": "GDP per capita, constant",
    },
    "NY.GDP.MKTP.KD.ZG": {
        "name": "gdp_growth_ann_pct",
        "transform": "level",
        "label": "GDP growth (annual %)",
    },
    "NE.GDI.FTOT.ZS": {
        "name": "inv_share_gdp",
        "transform": "level",
        "label": "Gross capital formation (% GDP)",
    },
    "FP.CPI.TOTL.ZG": {
        "name": "inflation_cpi_ann_pct",
        "transform": "level",
        "label": "Inflation, CPI (annual %)",
    },
    "FM.LBL.BMNY.GD.ZS": {
        "name": "broad_money_gdp",
        "transform": "level",
        "label": "Broad money (% GDP)",
    },
    "NE.TRD.GNFS.ZS": {
        "name": "trade_open_gdp",
        "transform": "level",
        "label": "Trade (% GDP)",
    },
    "BN.CAB.XOKA.GD.ZS": {
        "name": "cab_gdp",
        "transform": "level",
        "label": "Current account balance (% GDP)",
    },
    "TT.PRI.MRCH.XD.WD": {
        "name": "tot_index",
        "transform": "level",
        "label": "Net barter terms of trade index",
    },
    "NE.CON.GOVT.ZS": {
        "name": "gov_cons_gdp",
        "transform": "level",
        "label": "Government consumption (% GDP)",
    },
    "GC.TAX.TOTL.GD.ZS": {
        "name": "tax_rev_gdp",
        "transform": "level",
        "label": "Tax revenue (% GDP)",
    },
    "SL.UEM.TOTL.ZS": {
        "name": "unemp_rate",
        "transform": "level",
        "label": "Unemployment (% of labor force)",
    },
    "SP.POP.TOTL": {
        "name": "pop_total",
        "transform": "log",
        "label": "Population, total",
    },
    "SP.URB.TOTL.IN.ZS": {
        "name": "urban_share",
        "transform": "level",
        "label": "Urban population (% total)",
    },
    "FS.AST.PRVT.GD.ZS": {
        "name": "priv_credit_gdp",
        "transform": "level",
        "label": "Domestic credit to private sector (% GDP)",
    },
}

EPISODES = {
    "oil_shocks_1970_1975": {"start": 1970, "end": 1975},
    "volcker_debt_1975_1985": {"start": 1975, "end": 1985},
    "volcker_1975_1980": {"start": 1975, "end": 1980},
    "volcker_1980_1985": {"start": 1980, "end": 1985},
    "post_communist_1985_1995": {"start": 1985, "end": 1995},
    "asian_russian_1995_2000": {"start": 1995, "end": 2000},
    "gfc_2005_2010": {"start": 2005, "end": 2010},
    "post_gfc_2010_2015": {"start": 2010, "end": 2015},
    "covid_2015_2020": {"start": 2015, "end": 2020},
}

run_config = {
    "quinquennial_years": QUINQUENNIAL_YEARS,
    "wdi_indicators": WDI_INDICATORS,
    "episodes": EPISODES,
}

(Path(LOGS_DIR) / "run_config.json").write_text(json.dumps(run_config, indent=2))

indicator_df = (
    pd.DataFrame.from_dict(WDI_INDICATORS, orient="index")
    .reset_index()
    .rename(columns={"index": "indicator_code"})
)
display(indicator_df.style.set_caption("WDI indicators"))

episode_df = (
    pd.DataFrame.from_dict(EPISODES, orient="index")
    .reset_index()
    .rename(columns={"index": "episode"})
)
display(episode_df.style.set_caption("Episode windows"))

# %% [markdown]
# ## Environment interpretation
# The environment snapshot confirms the required analytics stack is present
# (no missing packages after installation). The configuration defines
# 14 WDI indicators on a quinquennial grid from 1970 to 2020, and 9 shock
# episodes that will be used for event-style comparisons later.
