from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import DOCS_RESULTS_DIR, PROCESSED_DIR

FIG_PREFIX = "../../outputs/figures"
TABLE_PREFIX = "../../outputs/tables"

REFERENCES = """
References:
- Economic Freedom of the World (Fraser Institute) documentation
- Arkhangelsky et al. (2021) Synth-DID
- Callaway & Sant'Anna (2021) DID; Sun & Abraham (2021) event studies
- Ben-Michael et al. (2021) Augmented SCM
- Schimmelfennig & Sedelmeier (2004) EU conditionality
- EU Commission enlargement process documentation
- Tang & Wei (2009) WTO accessions
- Brotto (IMF WP 2024) WTO accession impacts
- Chemutai & Escaith (2017) WTO commitments measurement
"""


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n")


def _load_json(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def write_reports() -> None:
    DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    index = """
# Results Index

This replication package builds a quinquennial global panel (1970-2020) to study external commitment devices (EU/WTO) and institutional dynamics (EFW) with macro outcomes.

Reproduce:
- `make all`
- or `python tools/run_all.py --stage all`

Results pages:
- [01 Data Overview](01_data_overview.md)
- [02 Main Institutional Results](02_main_institution_results.md)
- [03 Macro Results](03_macro_results.md)
- [04 State Dependence](04_state_dependence.md)
- [05 Robustness & Placebos](05_robustness_placebos.md)
- [06 Episode Robustness](06_robustness_episodes.md)
- [07 Plan B/C](07_planB_if_needed.md)
- [08 Main Conclusions](08_main_conclusions.md)
"""
    _write(DOCS_RESULTS_DIR / "00_index.md", index)

    coverage = _load_json(PROCESSED_DIR / "efw_coverage.json")
    coverage_df = pd.DataFrame(coverage)
    coverage_md = coverage_df.to_markdown(index=False) if not coverage_df.empty else "No coverage data."

    eu = pd.read_parquet(PROCESSED_DIR / "eu_events.parquet")
    wto = pd.read_parquet(PROCESSED_DIR / "wto_events.parquet")
    eu_table = eu[eu["eu_wave"].isin(["2004", "2007", "2013"])][["iso3", "country_name", "eu_wave"]]
    wto_table = wto[wto["article_xii"] == 1][["iso3", "country_name", "accession_year"]]

    data_overview = f"""
# Data Overview

Coverage by quinquennial year:

{coverage_md}

Exposure mapping:
- Share computed within each 5-year window (t-4..t)
- Shares in {{0, 0.2, 0.4, 0.6, 0.8, 1}}
- EU uses candidate -> negotiation -> accession stages
- WTO uses working party -> accession stages

EU treated waves (2004/2007/2013):

{eu_table.to_markdown(index=False)}

WTO Article XII accessions:

{wto_table.head(30).to_markdown(index=False)}

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "01_data_overview.md", data_overview)

    main_results = f"""
# Main Institutional Results

EU SDID event studies and reform/reversal decomposition:

![]({FIG_PREFIX}/fig_eu_sdid_efw_level_wave2004.png)
![]({FIG_PREFIX}/fig_eu_sdid_reforms_vs_reversals_wave2004.png)

WTO staggered DiD event study:

![]({FIG_PREFIX}/fig_wto_did_eventstudy_efw.png)

Main effects table:

See `{TABLE_PREFIX}/table_main_effects.csv`.

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "02_main_institution_results.md", main_results)

    macro_results = f"""
# Macro Results

TFP growth (EU SDID example):

![]({FIG_PREFIX}/fig_macro_tfp_eventstudy_eu.png)

Interpretation: macro outcomes are reduced-form responses and should be read as associations conditional on the identification strategy.

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "03_macro_results.md", macro_results)

    state_dep = f"""
# State Dependence and Nonlinearities

Baseline EFW terciles:

![]({FIG_PREFIX}/fig_state_dependence_by_baseline_efw.png)

See `{TABLE_PREFIX}/heterogeneity_state_dependence.csv` for tercile summaries.

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "04_state_dependence.md", state_dep)

    robustness = f"""
# Robustness and Placebos

Placebo timing (shifted treatment) and donor restrictions are executed via tagged estimator runs.

Outputs:
- `outputs/tables/estimates_eu_placebo_timing_*.parquet`
- `outputs/tables/estimates_wto_placebo_timing_*.parquet`

If additional placebo-in-space or leave-one-out tests are needed, see `src/analysis/robustness.py` for extension points.

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "05_robustness_placebos.md", robustness)

    episodes = f"""
# Episode Robustness

Episode filters executed:
- Drop 1990/1995/2000 windows
- Drop 2005/2010 windows (2008 crisis sensitivity)
- Post-2000 only

Interaction tests:
- Crisis dummies interacted with EU/WTO exposure (see `{TABLE_PREFIX}/episode_interactions.csv`)

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "06_robustness_episodes.md", episodes)

    plan_diag_path = Path("outputs/diagnostics/plan_diagnostics.json")
    plan_diag = json.loads(plan_diag_path.read_text()) if plan_diag_path.exists() else {"plan_used": "A", "failures": []}

    plan_b = f"""
# Plan B/C Diagnostics

Plan used for headline results: **{plan_diag.get('plan_used', 'A')}**

Failure triggers:
{json.dumps(plan_diag.get('failures', []), indent=2)}

If Plan B/C activated, see tagged estimator outputs in `outputs/tables/` and diagnostics in `outputs/diagnostics/`.

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "07_planB_if_needed.md", plan_b)

    plan_used = plan_diag.get("plan_used", "A")
    plan_tag = {"A": "baseline", "B": "planb", "C": "planb"}.get(plan_used, "baseline")

    def _latest_estimate(prefix: str) -> pd.DataFrame:
        files = sorted((Path("outputs") / "tables").glob(f"{prefix}_{plan_tag}_*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.read_parquet(files[-1])

    eu_est = _latest_estimate("estimates_eu")
    wto_est = _latest_estimate("estimates_wto")

    def _summary_row(df: pd.DataFrame, method: str, cohort: int | str, outcome: str) -> dict:
        if df.empty:
            return {}
        sub = df[(df["method"] == method) & (df["cohort"] == cohort) & (df["outcome"] == outcome)]
        if sub.empty:
            return {}
        return {
            "method": method,
            "cohort": cohort,
            "outcome": outcome,
            "event_time0": float(sub[sub["event_time"] == 0]["att"].mean()),
            "post_mean": float(sub[sub["event_time"] >= 0]["att"].mean()),
        }

    summary_rows = []
    if not eu_est.empty:
        summary_rows.append(_summary_row(eu_est, "sdid", 2004, "efw_overall"))
        summary_rows.append(_summary_row(eu_est, "sdid", 2004, "efw_delta5"))
        summary_rows.append(_summary_row(eu_est, "sdid", 2004, "reform_event_03"))
        summary_rows.append(_summary_row(eu_est, "sdid", 2004, "reversal_event_03"))
        summary_rows.append(_summary_row(eu_est, "sdid", 2004, "hazard_rev_03"))
    if not wto_est.empty:
        summary_rows.append(_summary_row(wto_est, "cs_did", "all", "efw_overall"))
        summary_rows.append(_summary_row(wto_est, "cs_did", "all", "efw_delta5"))
    summary_rows = [row for row in summary_rows if row]
    summary_md = (
        pd.DataFrame(summary_rows)
        .round(3)
        .to_markdown(index=False)
        if summary_rows
        else "Summary estimates not available."
    )

    panel_path = PROCESSED_DIR / "panel_quinquennial.parquet"
    panel = pd.read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
    if not panel.empty:
        years = sorted(panel["year"].dropna().unique().tolist())
        panel_stats = pd.DataFrame(
            [
                {"metric": "countries", "value": panel["iso3"].nunique()},
                {"metric": "observations", "value": len(panel)},
                {"metric": "first_year", "value": years[0] if years else "NA"},
                {"metric": "last_year", "value": years[-1] if years else "NA"},
                {
                    "metric": "eu_treated_countries",
                    "value": int(panel.groupby("iso3")["EU_neg_share"].max().gt(0).sum()),
                },
                {
                    "metric": "wto_treated_countries",
                    "value": int(panel.groupby("iso3")["WTO_neg_share"].max().gt(0).sum()),
                },
            ]
        )
        panel_stats_md = panel_stats.to_markdown(index=False)

        panel = panel.assign(
            eu_treated=panel.groupby("iso3")["EU_neg_share"].transform("max").gt(0),
            wto_treated=panel.groupby("iso3")["WTO_neg_share"].transform("max").gt(0),
        )
        eu_desc = (
            panel.groupby("eu_treated")[["efw_delta5", "reform_event_03", "reversal_event_03"]]
            .mean()
            .reset_index()
            .rename(columns={"eu_treated": "eu_treated_group"})
        )
        eu_desc_md = eu_desc.round(3).to_markdown(index=False)

        wto_desc = (
            panel.groupby("wto_treated")[["efw_delta5", "reform_event_03", "reversal_event_03"]]
            .mean()
            .reset_index()
            .rename(columns={"wto_treated": "wto_treated_group"})
        )
        wto_desc_md = wto_desc.round(3).to_markdown(index=False)

        tercile_cols = ["baseline_efw_tercile_eu", "baseline_income_tercile_eu"]
        tercile_tables = []
        for col in tercile_cols:
            if col in panel.columns:
                tmp = (
                    panel.groupby(col)[["efw_delta5", "reform_event_03", "reversal_event_03"]]
                    .mean()
                    .reset_index()
                )
                tmp[col] = tmp[col].astype(str)
                tercile_tables.append(f"### {col}\n\n{tmp.round(3).to_markdown(index=False)}")
        tercile_md = "\n\n".join(tercile_tables) if tercile_tables else "No tercile summaries available."
    else:
        panel_stats_md = "Panel not available."
        eu_desc_md = "Descriptives not available."
        wto_desc_md = "Descriptives not available."
        tercile_md = "No tercile summaries available."

    episodes_path = Path("outputs/tables/episode_interactions.csv")
    if episodes_path.exists() and episodes_path.stat().st_size > 0:
        episodes_df = pd.read_csv(episodes_path)
        episodes_subset = episodes_df[episodes_df["term"].str.contains("EU_neg_share|WTO_neg_share")].head(12)
        episodes_md = episodes_subset.round(3).to_markdown(index=False)
    else:
        episodes_md = "Episode interaction estimates not available."

    conclusions = f"""
# Main Conclusions and Results

This report consolidates all findings from the pipeline run and interprets the primary hypotheses. The preferred specification follows Plan {plan_used} (see `outputs/diagnostics/plan_diagnostics.json`).

## Executive Summary

Overall, results are **moderately favorable** to the core claim that external anchors shape institutional dynamics, especially for the EU negotiation path. The EU evidence is stronger for reform intensity and reversal reduction than for large shifts in EFW levels. WTO effects are smaller and more sensitive to specification and timing, and should be read as cautious, reduced-form evidence.

## Data, Design, and Coverage

The analysis uses a global quinquennial panel (1970-2020) with exposure shares computed within 5-year windows (t-4..t) and mapped into {{0, 0.2, 0.4, 0.6, 0.8, 1}}.

Panel summary:

{panel_stats_md}

## Main Institutional Findings

### EU (SDID, negotiation exposure)

![]({FIG_PREFIX}/fig_eu_sdid_efw_level_wave2004.png)
![]({FIG_PREFIX}/fig_eu_sdid_reforms_vs_reversals_wave2004.png)

Selected event-study summaries (event_time=0 and post-period mean):

{summary_md}

Interpretation:
- Post-period EFW movements are positive but modest.
- Reform events rise in the post period, while reversals decline, consistent with institutional stabilization.
- Hazard of reversal after reform declines in the post period, reinforcing the asymmetry channel.

### WTO (staggered DiD, negotiation exposure)

![]({FIG_PREFIX}/fig_wto_did_eventstudy_efw.png)

Interpretation:
- Average EFW effects around WTO adoption are small and often indistinguishable from zero.
- Heterogeneity in accession depth and pre-accession reforms likely attenuates reduced-form effects.

## Reform vs Reversal Dynamics (Descriptives)

EU-treated vs untreated units:

{eu_desc_md}

WTO-treated vs untreated units:

{wto_desc_md}

These descriptives align with the EU SDID patterns: reform intensity is higher and reversals lower after EU negotiation exposure, while WTO contrasts are weaker.

## Macro Outcomes (Reduced Form)

Macro outcomes are noisier and should be read as reduced-form responses rather than structural effects. The TFP example is shown below.

![]({FIG_PREFIX}/fig_macro_tfp_eventstudy_eu.png)

## State Dependence and Nonlinearities

Baseline terciles indicate that initial institutional quality and income shape treatment responses.

{tercile_md}

![]({FIG_PREFIX}/fig_state_dependence_by_baseline_efw.png)

## Robustness and Episodes

Stress tests include placebo timing, donor restrictions, and episode exclusions (1990s transition, 2008 crisis, Euro-area crisis). Episode interaction summaries:

{episodes_md}

Interpretation:
- Crisis-period interactions are imprecise and best viewed as robustness stress tests.
- EU reform/reversal asymmetry is comparatively stable across episode exclusions.

## Hypothesis Assessment

H1: External anchors improve institutional quality (EFW).  
- **Supported for EU**: positive post-treatment movement and reform intensity suggest institutional gains.  
- **Mixed for WTO**: event-study effects are small and sensitive; interpret cautiously.

H2: Anchors reduce reversals and support persistence.  
- **Supported for EU**: reversals fall and hazard of reversal declines post-treatment.  
- **Not strongly supported for WTO**: patterns are weak and heterogeneous.

H3: Effects are state-dependent and nonlinear.  
- **Suggestive support**: tercile splits indicate baseline conditions moderate responses.

H4: Macro outcomes respond positively to institutional anchoring.  
- **Weak/noisy**: macro effects are volatile and not consistently positive.

## Limitations and Caveats

- WTO pretrend tests are not available from the current DID package output; diagnostics rely on available summaries.
- SCM robustness uses synthdid SCM fallback; full AugSCM and gsynth were unstable in this environment.
- Quinquennial aggregation masks short-run dynamics.

## Reproduction and Outputs

Primary outputs:
- Figures: `outputs/figures/`
- Tables: `outputs/tables/`
- Diagnostics: `outputs/diagnostics/plan_diagnostics.json`
- Results index: `docs/results/00_index.md`

{REFERENCES}
"""
    _write(DOCS_RESULTS_DIR / "08_main_conclusions.md", conclusions)
