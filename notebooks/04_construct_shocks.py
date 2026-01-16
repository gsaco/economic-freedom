# %% [markdown]
# # 04. Construct stacked-event reform shocks
#
# Defines positive/negative reform events for stacked event studies.

# %%
from __future__ import annotations

import json
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

from src.stacked_event import EventConfig, identify_events
from src.viz import set_plot_style

DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"
OUTPUT_LOGS = ROOT / "output" / "logs"

for path in [OUTPUTS, OUTPUT_TABLES, OUTPUT_LOGS]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %%
panel = pd.read_csv(DATA_PROC / "panel_master_quinquennial_1970_2020.csv")

config = EventConfig(
    pos_quantile=0.90,
    neg_quantile=0.10,
    sustain_epsilon=0.2,
    sustain_horizons=[5],
    cooldown_years=10,
    window_pre=2,
    window_post=3,
)

min_event_count = 20
fallback_used = False

def count_events(events: pd.DataFrame) -> tuple[int, int]:
    if events.empty:
        return 0, 0
    counts = events.groupby("event_type").size().to_dict()
    return int(counts.get("pos", 0)), int(counts.get("neg", 0))

result = identify_events(
    panel,
    change_col="d_efw_summary",
    level_col="efw_summary",
    config=config,
)

pos_n, neg_n = count_events(result.events)
if pos_n < min_event_count or neg_n < min_event_count:
    fallback_used = True
    config = EventConfig(
        pos_quantile=0.85,
        neg_quantile=0.15,
        sustain_epsilon=0.2,
        sustain_horizons=[5],
        cooldown_years=10,
        window_pre=2,
        window_post=3,
    )
    result = identify_events(
        panel,
        change_col="d_efw_summary",
        level_col="efw_summary",
        config=config,
    )
    pos_n, neg_n = count_events(result.events)

EVENTS_PATH = DATA_PROC / "efw_reform_events.csv"
result.events.to_csv(EVENTS_PATH, index=False)
print("Events saved:", EVENTS_PATH)

# Thresholds + counts
thresholds = pd.DataFrame(
    {
        "pos_quantile": [config.pos_quantile],
        "neg_quantile": [config.neg_quantile],
        "pos_threshold": [result.pos_threshold],
        "neg_threshold": [result.neg_threshold],
        "sustain_epsilon": [config.sustain_epsilon],
        "sustain_horizons": [";".join(str(h) for h in config.sustain_horizons)],
        "cooldown_years": [config.cooldown_years],
        "window_pre": [config.window_pre],
        "window_post": [config.window_post],
        "min_event_count": [min_event_count],
        "fallback_used": [fallback_used],
    }
)
thresholds.to_csv(OUTPUT_TABLES / "stacked_event_thresholds.csv", index=False)

counts = result.events.groupby("event_type").size().reset_index(name="n_events")
counts.to_csv(OUTPUT_TABLES / "stacked_event_counts.csv", index=False)

# Log config
log = {
    "pos_quantile": config.pos_quantile,
    "neg_quantile": config.neg_quantile,
    "pos_threshold": result.pos_threshold,
    "neg_threshold": result.neg_threshold,
    "sustain_epsilon": config.sustain_epsilon,
    "sustain_horizons": config.sustain_horizons,
    "cooldown_years": config.cooldown_years,
    "window_pre": config.window_pre,
    "window_post": config.window_post,
    "min_event_count": min_event_count,
    "fallback_used": fallback_used,
    "event_count_pos": pos_n,
    "event_count_neg": neg_n,
    "event_count_total": int(result.events.shape[0]),
}
(OUTPUT_LOGS / "stacked_event_config.json").write_text(json.dumps(log, indent=2, sort_keys=True))
