from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DIR


def _latest_file(pattern: str) -> Path | None:
    files = sorted(FIGURES_DIR.parent.glob(pattern))
    return files[-1] if files else None


def _load_latest_estimates(prefix: str) -> pd.DataFrame:
    files = sorted((FIGURES_DIR.parent / "tables").glob(f"{prefix}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def _placeholder(ax: plt.Axes, title: str) -> None:
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()


def fig_eu_sdid_efw_level_wave2004() -> Path:
    estimates = _load_latest_estimates("estimates_eu")
    out_path = FIGURES_DIR / "fig_eu_sdid_efw_level_wave2004.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    if estimates.empty or "method" not in estimates.columns:
        _placeholder(ax, "EU SDID: EFW Level (2004 wave)")
    else:
        subset = estimates[
            (estimates["method"] == "sdid")
            & (estimates["cohort"] == "2004")
            & (estimates["outcome"] == "efw_overall")
        ]
        if subset.empty:
            _placeholder(ax, "EU SDID: EFW Level (2004 wave)")
        else:
            ax.plot(subset["event_time"], subset["att"], marker="o")
            ax.fill_between(subset["event_time"], subset["ci_low"], subset["ci_high"], alpha=0.2)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title("EU SDID: EFW Level (2004 wave)")
            ax.set_xlabel("Event time (quinquennial)")
            ax.set_ylabel("ATT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def fig_eu_sdid_reforms_vs_reversals_wave2004() -> Path:
    estimates = _load_latest_estimates("estimates_eu")
    out_path = FIGURES_DIR / "fig_eu_sdid_reforms_vs_reversals_wave2004.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    if estimates.empty or "method" not in estimates.columns:
        _placeholder(ax, "EU SDID: Reform vs Reversal (2004 wave)")
    else:
        reform = estimates[
            (estimates["method"] == "sdid")
            & (estimates["cohort"] == "2004")
            & (estimates["outcome"] == "reform_event_03")
        ]
        reversal = estimates[
            (estimates["method"] == "sdid")
            & (estimates["cohort"] == "2004")
            & (estimates["outcome"] == "reversal_event_03")
        ]

        if reform.empty and reversal.empty:
            _placeholder(ax, "EU SDID: Reform vs Reversal (2004 wave)")
        else:
            if not reform.empty:
                ax.plot(reform["event_time"], reform["att"], marker="o", label="Reform")
            if not reversal.empty:
                ax.plot(reversal["event_time"], reversal["att"], marker="s", label="Reversal")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title("EU SDID: Reform vs Reversal (2004 wave)")
            ax.set_xlabel("Event time (quinquennial)")
            ax.set_ylabel("ATT")
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def fig_wto_did_eventstudy_efw() -> Path:
    estimates = _load_latest_estimates("estimates_wto")
    out_path = FIGURES_DIR / "fig_wto_did_eventstudy_efw.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    if estimates.empty or "method" not in estimates.columns:
        _placeholder(ax, "WTO CS DiD: EFW Event Study")
    else:
        subset = estimates[(estimates["method"] == "cs_did") & (estimates["outcome"] == "efw_overall")]
        if subset.empty:
            _placeholder(ax, "WTO CS DiD: EFW Event Study")
        else:
            ax.plot(subset["event_time"], subset["att"], marker="o")
            ax.fill_between(subset["event_time"], subset["ci_low"], subset["ci_high"], alpha=0.2)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title("WTO CS DiD: EFW Event Study")
            ax.set_xlabel("Event time (quinquennial)")
            ax.set_ylabel("ATT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def fig_macro_tfp_eventstudy_eu() -> Path:
    estimates = _load_latest_estimates("estimates_eu")
    out_path = FIGURES_DIR / "fig_macro_tfp_eventstudy_eu.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    if estimates.empty or "method" not in estimates.columns:
        _placeholder(ax, "EU SDID: TFP Event Study")
    else:
        subset = estimates[
            (estimates["method"] == "sdid")
            & (estimates["cohort"] == "2004")
            & (estimates["outcome"] == "delta5_log_tfp")
        ]
        if subset.empty:
            _placeholder(ax, "EU SDID: TFP Event Study")
        else:
            ax.plot(subset["event_time"], subset["att"], marker="o")
            ax.fill_between(subset["event_time"], subset["ci_low"], subset["ci_high"], alpha=0.2)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title("EU SDID: TFP Event Study")
            ax.set_xlabel("Event time (quinquennial)")
            ax.set_ylabel("ATT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def fig_state_dependence_by_baseline_efw() -> Path:
    panel = pd.read_parquet(PROCESSED_DIR / "panel_quinquennial.parquet")
    out_path = FIGURES_DIR / "fig_state_dependence_by_baseline_efw.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    treated = panel[panel["EU_neg_share"] > 0]
    if treated.empty or "baseline_efw_tercile_eu" not in treated.columns:
        _placeholder(ax, "State Dependence: Baseline EFW")
    else:
        summary = treated.groupby("baseline_efw_tercile_eu")["efw_delta5"].mean().reset_index()
        ax.bar(summary["baseline_efw_tercile_eu"].astype(str), summary["efw_delta5"], color="#4C72B0")
        ax.set_title("State Dependence: Baseline EFW Terciles")
        ax.set_xlabel("Baseline EFW tercile")
        ax.set_ylabel("Mean Δ5 EFW")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def fig_dose_response_wto_commitment_depth() -> Path:
    panel = pd.read_parquet(PROCESSED_DIR / "panel_quinquennial.parquet")
    out_path = FIGURES_DIR / "fig_dose_response_wto_commitment_depth.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    if "I_wto" not in panel.columns or panel["I_wto"].isna().all():
        _placeholder(ax, "Dose Response: WTO Commitment Depth")
    else:
        df = panel[(panel["WTO_post_share"] > 0) & panel["I_wto"].notna()]
        if df.empty:
            _placeholder(ax, "Dose Response: WTO Commitment Depth")
        else:
            ax.scatter(df["I_wto"], df["efw_delta5"], alpha=0.5)
            ax.set_title("Dose Response: WTO Commitment Depth vs Δ5 EFW")
            ax.set_xlabel("WTO commitment index")
            ax.set_ylabel("Δ5 EFW")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def generate_all_figures() -> list[Path]:
    paths = [
        fig_eu_sdid_efw_level_wave2004(),
        fig_eu_sdid_reforms_vs_reversals_wave2004(),
        fig_wto_did_eventstudy_efw(),
        fig_macro_tfp_eventstudy_eu(),
        fig_state_dependence_by_baseline_efw(),
        fig_dose_response_wto_commitment_depth(),
    ]
    return paths
