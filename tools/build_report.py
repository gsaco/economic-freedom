from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import OUTPUT_DIR


def _rel(path: Path) -> str:
    return path.as_posix()


def _maybe_figure(path: Path) -> str:
    return f"![]({_rel(path)})" if path.exists() else "(figure not built)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.md"
    created_at = datetime.now(timezone.utc).isoformat()

    sections = []
    sections.append("# Quinquennial EFW Atlas Report")
    sections.append("")
    sections.append(f"Generated: {created_at}")
    sections.append("")

    sections.append("## Coverage and Missingness")
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "coverage" / "efw_coverage_by_year.png"))
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "coverage" / "wdi_coverage_by_year.png"))
    sections.append("")

    sections.append("## Global Trends and Distributions")
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "trends" / "efw_summary_trend.png"))
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "trends" / "efw_summary_boxplot.png"))
    sections.append("")

    sections.append("## Maps (Levels and Changes)")
    sections.append("See `output/figures/maps/` for level and change choropleths.")
    sections.append("")

    sections.append("## Components and Mobility")
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "components" / "efw_area_trends.png"))
    sections.append("See `output/figures/mobility/` and `output/tables/mobility_transition_matrices/`. ")
    sections.append("")

    sections.append("## Macro Co-movement")
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "comovement" / "scatter_levels.png"))
    sections.append(_maybe_figure(OUTPUT_DIR / "figures" / "comovement" / "correlation_heatmap_levels.png"))
    sections.append("")

    sections.append("## Shock Episodes")
    sections.append("See `output/figures/episodes/` and `output/tables/episode_leaderboards_*.csv`. ")
    sections.append("")

    report_path.write_text("\n".join(sections))


if __name__ == "__main__":
    main()
