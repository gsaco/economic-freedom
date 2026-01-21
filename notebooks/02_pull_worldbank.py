# %% [markdown]
# # Pull World Bank WDI indicators
# This notebook downloads and caches the selected WDI indicators, builds
# quinquennial extracts, and records coverage metadata for each indicator.

# %%
from __future__ import annotations

import json
import sys
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_worldbank import (
    fetch_country_metadata,
    fetch_indicator_metadata,
    fetch_wdi_indicator,
)
from src.paths import INTERMEDIATE_DIR, LOGS_DIR, OUTPUT_DIR, QUINQUENNIAL_YEARS, RAW_DIR
from src.viz_style import savefig, set_style

# %%
set_style()
config_path = Path(LOGS_DIR) / "run_config.json"
if not config_path.exists():
    raise FileNotFoundError("Missing run config. Run 00_env_setup first.")

config = json.loads(config_path.read_text())
indicators = config["wdi_indicators"]

# %%
cache_dir = RAW_DIR / "wdi_cache"
annual_frames = []
quin_frames = []
metadata = {}
coverage_rows = []

for code, spec in indicators.items():
    df = fetch_wdi_indicator(code, cache_dir)
    name = spec["name"]
    df = df.rename(columns={"value": name})

    annual_frames.append(df)
    df_q = df[df["year"].isin(QUINQUENNIAL_YEARS)].copy()
    quin_frames.append(df_q)

    coverage = df_q.groupby("year")[name].apply(lambda series: series.notna().sum())
    coverage_rows.append(coverage.rename(name))

    meta = fetch_indicator_metadata(code, RAW_DIR / "world_bank")
    meta.update({
        "short_name": name,
        "label": spec.get("label"),
        "transform": spec.get("transform"),
        "coverage_by_year": coverage.to_dict(),
        "coverage_total": int(df_q[name].notna().sum()),
    })
    metadata[name] = meta

# %%
annual = reduce(
    lambda left, right: left.merge(right, on=["iso3c", "year"], how="outer"),
    annual_frames,
)
annual_path = INTERMEDIATE_DIR / "wdi_annual_selected.parquet"
annual.to_parquet(annual_path, index=False)

quin = reduce(
    lambda left, right: left.merge(right, on=["iso3c", "year"], how="outer"),
    quin_frames,
)
quin_path = INTERMEDIATE_DIR / "wdi_quinquennial_selected.parquet"
quin.to_parquet(quin_path, index=False)

# %%
coverage_matrix = pd.concat(coverage_rows, axis=1).reset_index().rename(columns={"index": "year"})
coverage_path = OUTPUT_DIR / "tables" / "wdi_coverage_matrix.csv"
coverage_path.parent.mkdir(parents=True, exist_ok=True)
coverage_matrix.to_csv(coverage_path, index=False)
display(coverage_matrix.style.set_caption("WDI quinquennial coverage matrix"))

meta_path = INTERMEDIATE_DIR / "wdi_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

country_meta = fetch_country_metadata(RAW_DIR / "world_bank")
country_meta.to_parquet(INTERMEDIATE_DIR / "country_metadata.parquet", index=False)
display(country_meta.head(10).style.set_caption("Country metadata sample"))

meta_df = (
    pd.DataFrame.from_dict(metadata, orient="index")
    .reset_index()
    .rename(columns={"index": "indicator"})
    .loc[:, ["indicator", "short_name", "label", "transform", "coverage_total"]]
)
display(meta_df.style.set_caption("Indicator metadata summary"))

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for col in coverage_matrix.columns:
    if col == "year":
        continue
    ax.plot(coverage_matrix["year"], coverage_matrix[col], label=col)
ax.set_title("WDI indicator coverage (quinquennial)")
ax.set_xlabel("Year")
ax.set_ylabel("Non-missing countries")
ax.legend(fontsize=6, ncol=2)

display(fig)
savefig(fig, OUTPUT_DIR / "figures" / "coverage" / "wdi_coverage_by_year")
plt.close(fig)

# %% [markdown]
# ## Interpretation
# WDI coverage varies across indicators, but the quinquennial matrix shows
# consistently higher availability in recent decades. The most complete
# series are urban population share, total population, and GDP per capita
# (total non-missing counts of 2,871; 2,867; and 2,498 respectively), while
# tax revenue (1,076), terms of trade (1,333), and unemployment (1,386) are
# the sparsest. These differences guide which indicators can be used for
# tighter robustness checks versus exploratory comparisons.
