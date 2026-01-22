# %% [markdown]
# # Ingest ParlGov data
# This notebook downloads the ParlGov data bundle, harmonizes ISO3 codes,
# and writes cleaned election, cabinet, and party tables for analysis.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.elections import (
    download_parlgov_codebook,
    download_parlgov_zip,
    load_parlgov_bundle,
    parlgov_metadata,
    write_parlgov_metadata,
    write_parlgov_outputs,
)
from src.paths import INTERMEDIATE_DIR, PAPER_LOGS_DIR, RAW_DIR
from src.qc import assert_unique_key
from src.viz_style import set_style

# %%
set_style()
raw_dir = RAW_DIR / "parlgov"

zip_path = download_parlgov_zip(raw_dir)
codebook_path = download_parlgov_codebook(raw_dir)

bundle = load_parlgov_bundle(zip_path)
paths = write_parlgov_outputs(bundle, INTERMEDIATE_DIR)

meta = parlgov_metadata(bundle, zip_path, codebook_path)
meta_path = PAPER_LOGS_DIR / "parlgov_metadata.json"
write_parlgov_metadata(meta, meta_path)

display(pd.DataFrame([meta["build"]["tables"]]).style.set_caption("ParlGov table sizes"))
display(bundle.elections.head(5).style.set_caption("ParlGov elections (sample)"))
display(bundle.parties.head(5).style.set_caption("ParlGov parties (sample)"))

# %%
assert_unique_key(bundle.elections, ["election_id"])
assert_unique_key(bundle.parties, ["party_id"])
assert_unique_key(bundle.cabinets, ["cabinet_id"])

for name, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing expected output: {name}")

# %% [markdown]
# ## Interpretation
# The ParlGov tables now align with ISO3 country codes and are available as
# parquet files for constructing the close-election running variable and
# cabinet ideology diagnostics.
