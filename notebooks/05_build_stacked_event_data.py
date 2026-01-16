# %% [markdown]
# # 05. Build stacked event-study datasets

# %%
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(6):
        if (current / "src").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start.resolve()


ROOT = find_repo_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stacked_event import build_stacked_panel
from src.viz import set_plot_style

DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"

for path in [OUTPUTS, OUTPUT_TABLES]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %%
panel = pd.read_csv(DATA_PROC / "panel_master_quinquennial_1970_2020.csv")
events = pd.read_csv(DATA_PROC / "efw_reform_events.csv")

stacked_pos = build_stacked_panel(
    panel,
    events,
    window_pre=2,
    window_post=3,
    event_type="pos",
    exclude_any_event=True,
)
stacked_neg = build_stacked_panel(
    panel,
    events,
    window_pre=2,
    window_post=3,
    event_type="neg",
    exclude_any_event=True,
)

pos_path = DATA_PROC / "stacked_event_pos.csv"
neg_path = DATA_PROC / "stacked_event_neg.csv"
stacked_pos.to_csv(pos_path, index=False)
stacked_neg.to_csv(neg_path, index=False)
print("Stacked datasets saved:", pos_path, neg_path)

summary = pd.DataFrame(
    {
        "event_type": ["pos", "neg"],
        "n_rows": [int(stacked_pos.shape[0]), int(stacked_neg.shape[0])],
        "n_events": [stacked_pos["stack_id"].nunique() if not stacked_pos.empty else 0,
                     stacked_neg["stack_id"].nunique() if not stacked_neg.empty else 0],
        "n_countries": [stacked_pos["iso3"].nunique() if not stacked_pos.empty else 0,
                        stacked_neg["iso3"].nunique() if not stacked_neg.empty else 0],
    }
)
summary.to_csv(OUTPUT_TABLES / "stacked_event_panel_summary.csv", index=False)
