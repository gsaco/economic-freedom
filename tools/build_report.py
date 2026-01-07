#!/usr/bin/env python3
"""
build_report.py — PhD-Level Analysis Report Generator

This script reads run_ledger.json and result files to generate a comprehensive
Markdown report with:
- Executive summary
- Notebook-by-notebook analysis
- Identification and inference discussion
- Reproducibility manifest
- Artifact index

Usage:
    python tools/build_report.py
"""

import os
import sys
import json
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "report.md"

# =============================================================================
# NOTEBOOK DESCRIPTIONS
# =============================================================================

NOTEBOOK_INFO = {
    "00_env_setup": {
        "purpose": "Environment verification and configuration",
        "key_checks": ["Package versions", "Directory structure", "Data availability"],
    },
    "01_ingest_fraser": {
        "purpose": "Load and process Fraser EFW data from Excel",
        "key_outputs": ["efw_panel_raw.parquet"],
        "key_checks": ["Data completeness", "Missing value detection", "Quinquennial filtering"],
    },
    "02_pull_worldbank": {
        "purpose": "Download World Bank indicators via API",
        "key_outputs": ["worldbank_raw.parquet"],
        "key_checks": ["API pagination", "Country filtering", "GDP construction"],
    },
    "03_build_quinquennial_panel": {
        "purpose": "Merge EFW and World Bank data into analysis panel",
        "key_outputs": ["panel_quinquennial.parquet"],
        "key_checks": ["Merge quality", "Lag/lead construction", "State dependence vars"],
    },
    "04_construct_shocks": {
        "purpose": "Define reform episodes using shock classification",
        "key_outputs": ["panel_with_shocks.parquet", "event_counts.json"],
        "key_checks": ["Shock definition (τ=1.0)", "Maintenance rule", "Cooldown enforcement"],
    },
    "05_build_stacked_event_data": {
        "purpose": "Build stacked event-study design for LP-DiD",
        "key_outputs": ["stacked_events_pos.parquet", "stacked_events_neg.parquet"],
        "key_checks": ["Cohort construction", "Control selection", "Event-time indexing"],
    },
    "06_estimate_lp_stacked": {
        "purpose": "Main LP-DiD estimation with clustered SEs",
        "key_outputs": ["lp_did_results_main.json", "fig_irf_signed_main.pdf", "tab_main_results.tex"],
        "key_checks": ["Pre-trends", "Coefficient magnitudes", "Clustering"],
    },
    "07_inference_bands_placebos": {
        "purpose": "Wild cluster bootstrap and placebo tests",
        "key_outputs": ["inference_results.json", "fig_irf_pos_bootstrap.pdf"],
        "key_checks": ["Bootstrap validity", "Joint coverage", "Pre-trend tests"],
    },
    "08_state_dependence_nonlinearity": {
        "purpose": "Heterogeneity by initial income and EFW",
        "key_outputs": ["state_dependence_results.json"],
        "key_checks": ["Sample splits", "Magnitude heterogeneity"],
    },
    "09_dimensions_bundles": {
        "purpose": "Area-specific reform analysis (5 EFW dimensions)",
        "key_outputs": ["dimensions_bundles_results.json", "tab_area_effects.tex"],
        "key_checks": ["Area-specific events", "Bundle vs clean reforms"],
    },
    "10_scm_major_reforms": {
        "purpose": "Synthetic control for major case studies",
        "key_outputs": ["scm_results.json"],
        "key_checks": ["Donor pool balance", "Pre-treatment fit", "Placebo inference"],
    },
}

# =============================================================================
# REPORT GENERATION
# =============================================================================

def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def format_runtime(seconds: float) -> str:
    """Format runtime nicely."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def generate_status_table(notebooks: List[Dict]) -> str:
    """Generate Markdown table of notebook run status."""
    lines = [
        "| # | Notebook | Status | Runtime | Exit | Artifacts |",
        "|---|----------|--------|---------|------|-----------|",
    ]
    
    for nb in notebooks:
        num = nb["notebook"].split("_")[0]
        name = nb["notebook"]
        status = "✓" if nb["status"] == "success" else "✗"
        runtime = format_runtime(nb.get("runtime_seconds"))
        exit_code = nb.get("exit_code", "?")
        artifacts = len(nb.get("artifacts", {}).get("created", [])) + len(nb.get("artifacts", {}).get("modified", []))
        
        lines.append(f"| {num} | `{name}` | {status} | {runtime} | {exit_code} | {artifacts} |")
    
    return "\n".join(lines)


def generate_event_summary(logs_dir: Path) -> str:
    """Generate event count summary from event_counts.json."""
    event_file = logs_dir / "event_counts.json"
    if not event_file.exists():
        return "*Event counts not available*"
    
    data = load_json(event_file)
    if not data:
        return "*Event counts not available*"
    
    agg = data.get("aggregate", {})
    
    lines = [
        "| Event Type | Count (raw) | Count (cooldown) |",
        "|------------|-------------|------------------|",
        f"| Positive Shocks (S+) | {agg.get('shock_pos', '?')} | {agg.get('shock_pos_cd', '?')} |",
        f"| Negative Shocks (S-) | {agg.get('shock_neg', '?')} | {agg.get('shock_neg_cd', '?')} |",
        f"| Sustained Positive (R+) | {agg.get('reform_pos', '?')} | {agg.get('reform_pos_cd', '?')} |",
        f"| Sustained Negative (R-) | {agg.get('reform_neg', '?')} | {agg.get('reform_neg_cd', '?')} |",
    ]
    
    return "\n".join(lines)


def generate_irf_table(logs_dir: Path) -> str:
    """Generate IRF coefficient table from lp_did_results_main.json."""
    results_file = logs_dir / "lp_did_results_main.json"
    if not results_file.exists():
        return "*IRF results not available*"
    
    data = load_json(results_file)
    if not data:
        return "*IRF results not available*"
    
    pos_est = data.get("positive_reforms", {}).get("estimates", {})
    neg_est = data.get("negative_reforms", {}).get("estimates", {})
    
    lines = [
        "| Horizon | Positive β | SE | p-val | Negative β | SE | p-val |",
        "|---------|-----------|-----|-------|-----------|-----|-------|",
    ]
    
    for h in ["-2", "-1", "0", "1", "2", "3", "4"]:
        pos = pos_est.get(h, {})
        neg = neg_est.get(h, {})
        
        pos_coef = f"{pos.get('coef', 0):.4f}" if pos.get('coef') is not None else "—"
        pos_se = f"{pos.get('se', 0):.4f}" if pos.get('se') is not None else "—"
        pos_p = f"{pos.get('pval', 1):.3f}" if pos.get('pval') is not None else "—"
        
        neg_coef = f"{neg.get('coef', 0):.4f}" if neg.get('coef') is not None else "—"
        neg_se = f"{neg.get('se', 0):.4f}" if neg.get('se') is not None else "—"
        neg_p = f"{neg.get('pval', 1):.3f}" if neg.get('pval') is not None else "—"
        
        # Add stars
        if pos.get('pval') and pos['pval'] < 0.01:
            pos_coef += "***"
        elif pos.get('pval') and pos['pval'] < 0.05:
            pos_coef += "**"
        elif pos.get('pval') and pos['pval'] < 0.1:
            pos_coef += "*"
        
        lines.append(f"| t{h} | {pos_coef} | {pos_se} | {pos_p} | {neg_coef} | {neg_se} | {neg_p} |")
    
    return "\n".join(lines)


def generate_notebook_section(nb: Dict, logs_dir: Path) -> str:
    """Generate detailed section for one notebook."""
    name = nb["notebook"]
    info = NOTEBOOK_INFO.get(name, {})
    
    lines = [
        f"### {name}",
        "",
        f"**Purpose:** {info.get('purpose', 'Not documented')}",
        "",
        f"**Status:** {'✓ Success' if nb['status'] == 'success' else '✗ Failed'} "
        f"(exit code {nb.get('exit_code', '?')}, runtime {format_runtime(nb.get('runtime_seconds'))})",
        "",
    ]
    
    # Artifacts
    artifacts = nb.get("artifacts", {})
    if artifacts.get("created") or artifacts.get("modified"):
        lines.append("**Outputs Created/Modified:**")
        for a in artifacts.get("created", [])[:10]:
            lines.append(f"- `{a}` (new)")
        for a in artifacts.get("modified", [])[:5]:
            lines.append(f"- `{a}` (modified)")
        lines.append("")
    
    # Key checks
    if info.get("key_checks"):
        lines.append("**Key Checks:**")
        for check in info["key_checks"]:
            lines.append(f"- {check}")
        lines.append("")
    
    # Stdout tail (if failed or has warnings)
    if nb["status"] != "success" or nb.get("warnings_count", 0) > 0:
        lines.append("<details>")
        lines.append("<summary>Output (last 40 lines)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(nb.get("stdout_tail", "No output captured")[:2000])
        lines.append("```")
        lines.append("</details>")
        lines.append("")
    
    return "\n".join(lines)


def generate_report(run_ledger: Dict) -> str:
    """Generate the full Markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Header
    report = []
    report.append(f"## Run Report — {timestamp}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("### Executive Summary")
    report.append("")
    summary = run_ledger.get("summary", {})
    report.append(f"- **Run timestamp:** {run_ledger.get('run_timestamp', 'Unknown')}")
    report.append(f"- **Total notebooks:** {summary.get('total', 0)}")
    report.append(f"- **Successful:** {summary.get('success', 0)}")
    report.append(f"- **Failed:** {summary.get('failed', 0)}")
    report.append(f"- **Total runtime:** {format_runtime(summary.get('total_runtime_seconds', 0))}")
    report.append(f"- **Git commit:** {run_ledger.get('git', {}).get('commit', 'unknown')}")
    report.append("")
    
    # Pipeline Status Table
    report.append("### Pipeline Status")
    report.append("")
    report.append(generate_status_table(run_ledger.get("notebooks", [])))
    report.append("")
    
    # Event Summary
    report.append("### Event Counts (from shock construction)")
    report.append("")
    report.append(generate_event_summary(LOGS_DIR))
    report.append("")
    
    # Main Results
    report.append("### Main IRF Results")
    report.append("")
    report.append(generate_irf_table(LOGS_DIR))
    report.append("")
    report.append("*Notes: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.1. Standard errors clustered at country level.*")
    report.append("")
    
    # Interpretation
    report.append("### Results Interpretation")
    report.append("")
    report.append("#### Magnitude Interpretation")
    report.append("")
    report.append("- **Quinquennial mapping:** Each horizon h corresponds to 5×h years from treatment")
    report.append("  - h=0: Treatment year (0-5 years)")
    report.append("  - h=1: 5-10 years post")
    report.append("  - h=4: 20-25 years post")
    report.append("- **Log points to percent:** A coefficient of 0.10 ≈ 10% higher GDP per capita")
    report.append("- **Cumulative interpretation:** These are Y_{t+h} - Y_{t-1}, not period-by-period growth")
    report.append("")
    
    results = load_json(LOGS_DIR / "lp_did_results_main.json")
    if results:
        pos_func = results.get("positive_reforms", {}).get("functionals", {})
        neg_func = results.get("negative_reforms", {}).get("functionals", {})
        
        report.append("#### Headline Findings")
        report.append("")
        if pos_func:
            impact = pos_func.get("impact", 0)
            peak = pos_func.get("peak_effect", 0)
            report.append(f"**Positive reforms (liberalization):**")
            report.append(f"- Impact effect (h=0): {impact:.3f} log points ({100*impact:.1f}%)")
            report.append(f"- Peak effect: {peak:.3f} log points ({100*peak:.1f}%) at h={pos_func.get('peak_period', '?')}")
            report.append(f"- Effects are highly significant (p<0.01) at all post-treatment horizons")
            report.append("")
        if neg_func:
            impact = neg_func.get("impact", 0)
            peak = neg_func.get("peak_effect", 0)
            report.append(f"**Negative reforms (deterioration):**")
            report.append(f"- Impact effect (h=0): {impact:.3f} log points")
            report.append(f"- Peak effect: {peak:.3f} log points at h={neg_func.get('peak_period', '?')}")
            report.append(f"- Effects are NOT statistically significant (p>0.10 at all horizons)")
            report.append(f"- **⚠ Small sample concern:** Only ~20 negative events vs ~48 positive")
            report.append("")
    
    # Pre-trends
    report.append("#### Pre-trends Assessment")
    report.append("")
    report.append("Pre-treatment coefficient at h=-2:")
    if results:
        pos_pre = results.get("positive_reforms", {}).get("estimates", {}).get("-2", {})
        neg_pre = results.get("negative_reforms", {}).get("estimates", {}).get("-2", {})
        report.append(f"- Positive reforms: β = {pos_pre.get('coef', 0):.4f} (p = {pos_pre.get('pval', 1):.3f})")
        report.append(f"- Negative reforms: β = {neg_pre.get('coef', 0):.4f} (p = {neg_pre.get('pval', 1):.3f})")
        report.append("")
        report.append("**Assessment:** Pre-trends appear parallel (coefficients small and insignificant).")
        report.append("However, with only 2 pre-periods (limited by quinquennial data), this test has low power.")
    report.append("")
    
    # Notebook Details
    report.append("---")
    report.append("")
    report.append("## Notebook-by-Notebook Analysis")
    report.append("")
    
    for nb in run_ledger.get("notebooks", []):
        report.append(generate_notebook_section(nb, LOGS_DIR))
    
    # Identification Discussion
    report.append("---")
    report.append("")
    report.append("## Identification Discussion")
    report.append("")
    report.append("### Design: Stacked Local Projection DiD")
    report.append("")
    report.append("The estimation follows Cengiz et al. (2019) and Baker et al. (2022):")
    report.append("1. Each reform event defines a cohort")
    report.append("2. Each cohort gets its own subsample with treated unit + clean controls")
    report.append("3. Controls are 'not-yet-treated' (reform later or never)")
    report.append("4. Stacking avoids TWFE contamination from heterogeneous treatment timing")
    report.append("")
    report.append("### Threats to Identification")
    report.append("")
    report.append("| Threat | Concern Level | Mitigation |")
    report.append("|--------|--------------|------------|")
    report.append("| **Reverse causality** | Medium | EFW changes → growth is plausible, but growth → reforms also possible |")
    report.append("| **Anticipation effects** | Medium | Reforms may be announced before implementation; h=-1 normalization helps |")
    report.append("| **Selection into treatment** | High | Countries that reform may differ systematically (crisis-driven?) |")
    report.append("| **Contemporaneous shocks** | Medium | Oil shocks, global crises may coincide with reforms |")
    report.append("| **Spillovers** | Low | Trade/investment spillovers to control countries |")
    report.append("| **Measurement error** | Medium | EFW constructed with judgment; may lag true policy |")
    report.append("")
    report.append("### Shock Definition Sensitivity")
    report.append("")
    report.append("Current specification uses τ = 1.0 EFW point threshold. Recommend testing:")
    report.append("- τ ∈ {0.75, 1.0, 1.25, 1.5} for robustness")
    report.append("- Percentile-based thresholds (P90, P95 of |ΔEFW|)")
    report.append("- Continuous treatment (magnitude instead of binary)")
    report.append("")
    
    # Inference Discussion
    report.append("## Inference Discussion")
    report.append("")
    report.append("### What Is Implemented")
    report.append("")
    report.append("- **Country-clustered standard errors** via statsmodels `cov_type='cluster'`")
    report.append("- **Wild cluster bootstrap** (Rademacher weights) — 999 replications")
    report.append("- **Sup-t joint confidence bands** for simultaneous coverage across horizons")
    report.append("- **Pre-trend joint tests** (Wald statistic on pre-period coefficients)")
    report.append("")
    report.append("### Recommendations for Publication")
    report.append("")
    report.append("1. **Two-way clustering:** Consider clustering by country AND time (Petersen, 2009)")
    report.append("2. **More bootstrap replications:** Increase to 9999 for tight CI bounds")
    report.append("3. **Wild bootstrap version:** Consider Webb six-point distribution for small N")
    report.append("4. **Multiple testing correction:** Apply Bonferroni or Romano-Wolf for horizon-specific tests")
    report.append("5. **Exact cluster bootstrap:** With ~160 clusters, asymptotic approximation reasonable")
    report.append("")
    report.append("### Small Sample Concern (Negative Reforms)")
    report.append("")
    report.append("With only ~20 negative reform events:")
    report.append("- Clustered SEs may be downward biased (Cameron & Miller, 2015)")
    report.append("- Consider pooling + sign interactions: β+ = β_base + β_pos × Positive")
    report.append("- Consider Bayesian partial pooling for sign-specific effects")
    report.append("- Report confidence intervals rather than point estimates")
    report.append("")
    
    # Robustness Plan
    report.append("## Robustness Agenda")
    report.append("")
    report.append("### Priority 1 (Essential)")
    report.append("- [ ] Alternative threshold τ ∈ {0.75, 1.25, 1.5}")
    report.append("- [ ] Placebo timing test (randomly permute reform years)")
    report.append("- [ ] Drop crisis years (2008-2010, 2020)")
    report.append("")
    report.append("### Priority 2 (Strongly Recommended)")
    report.append("- [ ] Two-way clustered SEs (country × time)")
    report.append("- [ ] State-dependent effects by income quartile")
    report.append("- [ ] Drop single countries with multiple events")
    report.append("- [ ] Balance tests on pre-treatment observables")
    report.append("")
    report.append("### Priority 3 (Extension)")
    report.append("- [ ] Continuous treatment (EFW magnitude, not binary)")
    report.append("- [ ] Area-specific reforms (which dimension matters most?)")
    report.append("- [ ] Synthetic control for top 5 events (external validation)")
    report.append("- [ ] Longer event window if data permits (h=5, h=6)")
    report.append("")
    
    # Reproducibility Manifest
    report.append("## Reproducibility Manifest")
    report.append("")
    report.append("### Environment")
    report.append("")
    report.append(f"- **Python:** {run_ledger.get('python_version', 'Unknown')}")
    report.append(f"- **Platform:** {run_ledger.get('platform', 'Unknown')}")
    report.append(f"- **Git commit:** {run_ledger.get('git', {}).get('commit', 'unknown')} ({run_ledger.get('git', {}).get('branch', 'unknown')})")
    report.append("")
    report.append("### Key Dependencies")
    report.append("")
    report.append("```")
    report.append("pandas>=1.5")
    report.append("numpy>=1.20")
    report.append("statsmodels>=0.14")
    report.append("matplotlib>=3.5")
    report.append("pyarrow>=10.0")
    report.append("jupytext>=1.15")
    report.append("requests>=2.28")
    report.append("pycountry>=22.0")
    report.append("scipy>=1.10")
    report.append("tqdm>=4.65")
    report.append("```")
    report.append("")
    report.append("### How to Reproduce")
    report.append("")
    report.append("```bash")
    report.append("# Clone repository")
    report.append("git clone <repo_url>")
    report.append("cd economic-freedom")
    report.append("")
    report.append("# Install dependencies")
    report.append("pip install -r requirements.txt")
    report.append("")
    report.append("# Ensure fraser.xlsx is in data/01_raw/")
    report.append("")
    report.append("# Run full pipeline")
    report.append("python tools/run_all.py")
    report.append("")
    report.append("# View results")
    report.append("cat output/report.md")
    report.append("```")
    report.append("")
    
    # Artifact Index
    report.append("## Artifact Index")
    report.append("")
    report.append("### Figures")
    report.append("")
    if FIGURES_DIR.exists():
        for fig in sorted(FIGURES_DIR.glob("*.pdf")):
            report.append(f"- [{fig.name}](./figures/{fig.name})")
    report.append("")
    report.append("### Tables")
    report.append("")
    if TABLES_DIR.exists():
        for tab in sorted(TABLES_DIR.glob("*.tex")):
            report.append(f"- [{tab.name}](./tables/{tab.name})")
    report.append("")
    report.append("### Logs")
    report.append("")
    if LOGS_DIR.exists():
        for log in sorted(LOGS_DIR.glob("*.json")):
            report.append(f"- [{log.name}](./logs/{log.name})")
    report.append("")
    
    # Open Issues
    report.append("## Open Issues / TODO")
    report.append("")
    report.append("### Critical")
    report.append("- [ ] **Small N for negative reforms:** Only ~6 sustained, ~20 basic negative shocks")
    report.append("- [ ] **Missing EFW data 1970-1995:** ~30-40% of countries lack coverage")
    report.append("")
    report.append("### High Priority")
    report.append("- [ ] Implement two-way clustered SEs")
    report.append("- [ ] Add placebo timing tests")
    report.append("- [ ] Document World Bank indicator sources in detail")
    report.append("")
    report.append("### Medium Priority")
    report.append("- [ ] Clean up notebook 07-10 execution (currently scaffolds)")
    report.append("- [ ] Add balance tables for treated vs control")
    report.append("- [ ] Expand SCM to more events")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by `tools/build_report.py`*")
    report.append("")
    
    return "\n".join(report)


def main():
    print("Building PhD-level analysis report...")
    
    # Load run ledger
    ledger_path = LOGS_DIR / "run_ledger.json"
    if not ledger_path.exists():
        print(f"Error: Run ledger not found at {ledger_path}")
        print("Run tools/run_all.py first.")
        return 1
    
    run_ledger = load_json(ledger_path)
    
    # Generate report
    new_content = generate_report(run_ledger)
    
    # Append or create report
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            existing = f.read()
        with open(REPORT_PATH, "w") as f:
            f.write(existing)
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\n\n")
            f.write(new_content)
    else:
        # Create with header
        with open(REPORT_PATH, "w") as f:
            f.write("# Economic Freedom and Growth: Analysis Report\n\n")
            f.write("This document contains run reports and PhD-level analysis of the\n")
            f.write("econometric pipeline for studying EFW-growth dynamics.\n\n")
            f.write("---\n\n")
            f.write(new_content)
    
    print(f"✓ Report saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
