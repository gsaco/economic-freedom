# %% [markdown]
# # Ingest CLEA data (global legislative elections)
# This notebook reads a local CLEA download using a manifest that maps columns,
# aggregates constituency results to national totals, and writes intermediate tables.

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

from src.clea import build_clea_bundle, clea_metadata, write_clea_metadata, write_clea_outputs
from src.paths import INTERMEDIATE_DIR, PAPER_LOGS_DIR, RAW_DIR
from src.qc import assert_unique_key
from src.viz_style import set_style

# %%
set_style()
raw_dir = RAW_DIR / "clea"
manifest_path = raw_dir / "clea_manifest.json"

if not manifest_path.exists():
    print("CLEA manifest missing; skipping CLEA ingest.")
else:
    bundle = build_clea_bundle(raw_dir, manifest_path)
    paths = write_clea_outputs(bundle, INTERMEDIATE_DIR)

    meta = clea_metadata(bundle, manifest_path)
    meta_path = PAPER_LOGS_DIR / "clea_metadata.json"
    write_clea_metadata(meta, meta_path)

    display(pd.DataFrame([meta["build"]["tables"]]).style.set_caption("CLEA table sizes"))
    display(bundle.elections.head(5).style.set_caption("CLEA elections (sample)"))
    display(bundle.results.head(5).style.set_caption("CLEA results (sample)"))
    assert_unique_key(bundle.elections, ["election_id"])

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing expected output: {name}")

# %% [markdown]
# ## Interpretation
# CLEA elections and results are now available in `data/02_intermediate/` for constructing
# close-election margins at national level.
