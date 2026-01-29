# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cleanroom Full Procedure Notebook (Ultra Long)
#
# This notebook reproduces the **entire cleanroom procedure**: opening all datasets, performing the merges step-by-step,
# and generating the final outputs and audits. It mirrors the Sections 0–12 report and provides code evidence for each step.
#
# **Run from repo root** (preferred) or from `notebooks/` (auto-detects repo root).

# %%
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Robust root detection
cwd = Path.cwd().resolve()
if (cwd / "data" / "processed").exists():
    ROOT = cwd
elif (cwd.parent / "data" / "processed").exists():
    ROOT = cwd.parent
else:
    # allow running before outputs exist
    if (cwd / "data").exists() or (cwd.parent / "data").exists():
        ROOT = cwd if (cwd / "data").exists() else cwd.parent
    else:
        raise FileNotFoundError("Could not find repo root. Run from repo root or notebooks/.")

DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_INT = ROOT / "data" / "interim"
DATA_OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
AUDIT = REPORTS / "audit"

DATA_INT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

import sys
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# %% [markdown]
# ## 0) Executive summary (10–20 bullets)
#
# - Final deliverable: audit-ready, reproducible, cross-country elections dataset at the election-event grain with top-two results, ideology assignment, and EFW alignment.
# - Output files written (trimmed to keep-list): `data/processed/country.parquet`, `data/processed/party.parquet`, `data/processed/party_crosswalk.parquet`, `data/processed/party_ideology_obs.parquet`, `data/processed/election_event_sources.parquet`, `data/processed/election_event.parquet`, `data/processed/election_event_horizon_efw.parquet`.
# - Source-record layer size: 8,613 rows (`election_event_sources.parquet`).
# - Canonical election-event layer size: 8,607 rows (`election_event.parquet`).
# - Horizon table size: 77,463 rows (`election_event_horizon_efw.parquet`).
# - 2000+ coverage: 2,552 canonical events; ISO3 missing rate 8.46 percent; share_1 missing rate 3.21 percent.
# - Raw source counts verified from files: NED pres 1,409; NED parl 4,900; NELDA 4,158; CLEA LC 1,372,916 rows (2,304 election ids); V-Party 11,898; EFW panel 4,950 (165 ISO3); ETAD 3,133.
# - Country table built from ParlGov ISO list: 239 ISO3 rows; unmatched/ambiguous country-name mappings logged (NED 37, CLEA 23, NELDA 21, DPI 34, EFW 24).
# - NELDA matching outcomes on source records: matched 3,348; matched_outside_tolerance 28; ambiguous 1; unmatched 5,236.
# - Party matching remains sparse: 2000+ party_1_id missing 74.84 percent; 10_party_match_unmatched.csv contains 10,355 unique party strings; 1,943 ambiguous.
# - Ideology coverage: ideo_source counts in canonical events: vparty 412, parlgov 89, ches 5, elff 2, mixed 1,043, missing 7,056.
# - EFW baseline alignment: matched 2,107 events; no_efw_country 5,921; no_iso3 579.
# - Determinism controls: fixed config in `config/config.yaml`, stable sorts, no random seeds; build logs and audits written under `reports/audit/`.
# - No silent dropping: unmatched/ambiguous country, NELDA, party, and DES duplicates are written to audit tables; analysis-ready subsets are flagged rather than dropped.
# - Manual override channel: `manual_patch.csv` with deterministic patch application and audit columns.
# - Known gaps for manual review: party matching for coalitions/new parties; ideology after 2019 outside CHES/ParlGov; remaining unmapped historical entities.

# %% [markdown]
# ## Imports: cleanroom pipeline modules

# %%
from elections.config import load_config
from elections.audit import audit_rowcount
from elections.source_registry import build_source_registry
from elections.build_sources import (
    SourcePaths,
    write_raw_manifest,
    ingest_ned_pres,
    ingest_ned_parl,
    union_ned,
    aggregate_clea_lc,
    attach_iso3_ned,
    attach_iso3_clea,
    merge_nelda,
)
from elections.io import (
    read_ned_pres,
    read_ned_parl,
    read_nelda,
    read_clea_lc,
    read_efw_panel,
    read_vparty,
    read_cow2iso,
    read_partyfacts_core,
    read_partyfacts_external,
    read_ches,
    read_elff,
    read_parlgov_zip,
    read_des_zip,
    read_idea,
    read_dpi,
)
from elections.country_map import build_iso_reference, map_names_to_iso, build_cowcode_map, build_country_table
from elections.party_match import build_party_match_results
from elections.ideology import build_party_crosswalk, build_ideology_obs, assign_ideology_to_elections
from elections.dedupe import dedupe_events, attach_primary_secondary_shares
from elections.efw_align import prepare_efw_panel, attach_efw_baseline, build_efw_horizons
from elections.keep_lists import (
    COUNTRY_KEEP,
    PARTY_KEEP,
    PARTY_CROSSWALK_KEEP,
    PARTY_IDEOLOGY_KEEP,
    ELECTION_SOURCES_KEEP,
    ELECTION_EVENT_KEEP,
    ELECTION_HORIZON_KEEP,
)

cfg = load_config(ROOT)

# %% [markdown]
# ## 1) ZIP Inventory + Source Registry
#
# This step writes raw file manifest + source registry and then loads them for inspection.

# %%
write_raw_manifest(DATA_RAW, DATA_EXT, REPORTS / "manifests" / "raw_files_manifest.csv", extra_paths=[ROOT / "cow2iso.csv"])
source_registry = build_source_registry(DATA_RAW, DATA_EXT, REPORTS / "source_registry.csv")
source_registry.head(5)

# %% [markdown]
# ## 2) Open all datasets (raw + external)
#
# This section **opens every dataset** listed in the registry. Large datasets are loaded fully
# because the pipeline depends on full content (e.g., CLEA aggregation).

# %%
# Raw: NED pres/parl
ned_pres_raw = read_ned_pres(DATA_RAW / "ned" / "presidential_elections_v2.dta")
ned_parl_raw = read_ned_parl(DATA_RAW / "ned" / "parliamentary_elections_v2.dta")

# Raw: NELDA
nelda_raw = read_nelda(DATA_RAW / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta")

# Raw: CLEA LC
clea_lc_raw = read_clea_lc(DATA_RAW / "clea" / "clea_lc_20251015.sav")

# Raw: CLEA UC (optional but opened)
# We open only to verify availability; not used in baseline pipeline
import pyreadstat
clea_uc_raw, _ = pyreadstat.read_sav(DATA_RAW / "clea" / "clea_uc_20220111_spss" / "clea_uc_20220111.sav")

# Raw: V-Party, EFW, ETAD
vparty_raw = read_vparty(DATA_RAW / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv")
cow2iso_raw = read_cow2iso(ROOT / "cow2iso.csv")
efwp_raw = read_efw_panel(DATA_RAW / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx")
etad_raw = pd.read_csv(DATA_RAW / "etad" / "ETAD_v_1_0_0.csv")

# External: PartyFacts, CHES, ELFF, ParlGov, DES, IDEA, DPI
# Backward-compatible path access for older kernels
pf_core_path = getattr(cfg.paths, "partyfacts_core", DATA_EXT / "partyfacts_core_parties.csv")
pf_ext_path = getattr(cfg.paths, "partyfacts_external", DATA_EXT / "partyfacts_external_parties.csv")
pf_core = read_partyfacts_core(pf_core_path)
pf_ext = read_partyfacts_external(pf_ext_path)
ches = read_ches(DATA_EXT / "ches_1999_2024.csv")
elff = read_elff(DATA_EXT / "elff_partypos_summaries.csv")
parlgov = read_parlgov_zip(DATA_EXT / "parlgov.zip")
des = read_des_zip(DATA_EXT / "des_es_data_v50.zip")
idea = read_idea(DATA_EXT / "idea_export_electoral_system_design_database.xlsx")
dpi = read_dpi(DATA_EXT / "dpi" / "DPI2020" / "dpi2020.csv")

{
    "ned_pres_rows": len(ned_pres_raw),
    "ned_parl_rows": len(ned_parl_raw),
    "nelda_rows": len(nelda_raw),
    "clea_lc_rows": len(clea_lc_raw),
    "clea_uc_rows": len(clea_uc_raw),
    "vparty_rows": len(vparty_raw),
    "cow2iso_rows": len(cow2iso_raw),
    "efw_rows": len(efwp_raw),
    "etad_rows": len(etad_raw),
    "partyfacts_core_rows": len(pf_core),
    "partyfacts_external_rows": len(pf_ext),
    "ches_rows": len(ches),
    "elff_rows": len(elff),
    "parlgov_tables": list(parlgov.keys()),
    "des_rows": len(des),
    "idea_rows": len(idea),
    "dpi_rows": len(dpi),
}

# %% [markdown]
# ## 3) NED ingestion (pres + parl)
#
# Build canonical NED records with date parsing, shares, and flags.

# %%
spaths = SourcePaths(DATA_RAW, DATA_INT, AUDIT)

ned_pres = ingest_ned_pres(spaths)
ned_parl = ingest_ned_parl(spaths)

ned = union_ned(ned_pres, ned_parl, spaths)

{
    "ned_pres_rows": len(ned_pres),
    "ned_parl_rows": len(ned_parl),
    "ned_union_rows": len(ned),
}

# %% [markdown]
# ## 4) CLEA LC aggregation to election-level
#
# Aggregate pv1 to national totals by party, compute vote shares, and select top 2 parties.

# %%
clea_party, clea_elections = aggregate_clea_lc(spaths)
{
    "clea_party_rows": len(clea_party),
    "clea_elections_rows": len(clea_elections),
}

# %% [markdown]
# ## 5) Country crosswalk and ISO3 mapping
#
# Use ParlGov ISO list + name mapping + V-Party COW mapping.

# %%
iso_ref = build_iso_reference(DATA_EXT / "parlgov.zip")

ned_name_map = map_names_to_iso(ned[["country"]]["country"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
clea_name_map = map_names_to_iso(clea_elections["ctr_n"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])

cow_map = build_cowcode_map(cow2iso_raw, iso_ref=iso_ref)

country = build_country_table(
    iso_ref=iso_ref,
    ned=ned_pres_raw[["country"]].rename(columns={"country": "country"}),
    clea=clea_lc_raw,
    nelda=nelda_raw,
    vparty=vparty_raw,
    cow2iso=cow2iso_raw,
    dpi=dpi,
    efw=efwp_raw,
    min_score=cfg.params.country_match["min_score"],
    tie_delta=cfg.params.country_match["tie_delta"],
    audit_dir=AUDIT,
)

country[COUNTRY_KEEP].head()

# %% [markdown]
# ## 6) Attach ISO3 to NED and CLEA records
#
# This step merges the ISO3 mappings to the NED and CLEA election records.

# %%
ned_iso = attach_iso3_ned(ned, cow_map=cow_map, name_map=ned_name_map)
clea_iso = attach_iso3_clea(clea_elections, iso_ref=iso_ref, name_map=clea_name_map)

{
    "ned_iso_rows": len(ned_iso),
    "clea_iso_rows": len(clea_iso),
    "ned_iso_missing_iso3_pct": ned_iso["iso3"].isna().mean(),
    "clea_iso_missing_iso3_pct": clea_iso["iso3"].isna().mean(),
}

# %% [markdown]
# ## 7) Union NED + CLEA into source record layer
#
# The full source record layer keeps all provenance.

# %%
# Align columns and union
for col in ned_iso.columns:
    if col not in clea_iso.columns:
        clea_iso[col] = pd.NA
for col in clea_iso.columns:
    if col not in ned_iso.columns:
        ned_iso[col] = pd.NA

sources = pd.concat([ned_iso, clea_iso], ignore_index=True)
sources["country_name_raw"] = sources.get("country").combine_first(sources.get("ctr_n"))
sources["record_id"] = sources.apply(lambda r: f"{r['source']}:{r['source_election_id']}", axis=1)

# attach cow_code from country table
sources = sources.merge(country[["iso3", "cow_code"]], on="iso3", how="left", suffixes=("", "_country"))

sources.to_parquet(DATA_INT / "election_event_sources_base.parquet", index=False)

{
    "source_rows": len(sources),
    "source_unique_record_id": sources["record_id"].nunique(),
}

# %% [markdown]
# ## 8) Merge NELDA competitiveness flags
#
# Matching uses cow_code + election_year + office_type with date tolerance.

# %%
sources_nelda, nelda_report = merge_nelda(sources, nelda_raw)
nelda_report.to_csv(AUDIT / "09_nelda_match_report.csv", index=False)

# apply clean and competitive flags (replicating pipeline logic)

def apply_clean_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["flag_coup", "flag_inconsequential", "flag_unopposed", "flag_indirect"]:
        if col in df.columns:
            df[col] = df[col].fillna(False)
    df["clean_flag"] = (~df.get("flag_coup", False)) & (~df.get("flag_inconsequential", False)) & (~df.get("flag_unopposed", False)) & (~df.get("flag_indirect", False))

    def is_yes(x):
        return str(x).strip().lower() == "yes"

    df["competitive_flag"] = pd.NA
    if {"nelda3", "nelda4", "nelda5"}.issubset(df.columns):
        df["competitive_flag"] = df["nelda3"].map(is_yes) & df["nelda4"].map(is_yes) & df["nelda5"].map(is_yes)

    df["competitive_sample_strict"] = pd.NA
    mask = df["clean_flag"].notna() & df["competitive_flag"].notna()
    df.loc[mask, "competitive_sample_strict"] = df.loc[mask, "clean_flag"] & df.loc[mask, "competitive_flag"]

    df["competitive_sample_conservative"] = df["clean_flag"] & (df["competitive_flag"] == True)
    return df


sources_nelda = apply_clean_flags(sources_nelda)

nelda_report["nelda_match_status"].value_counts(dropna=False)

# %% [markdown]
# ## 9) Party matching to PartyFacts
#
# Match party names to PartyFacts using fuzzy matching within iso3 and year bounds.

# %%
party_match = build_party_match_results(
    sources_nelda,
    pf_core,
    partyfacts_external=pf_ext,
    min_score=cfg.params.party_match["min_score"],
    tie_delta=cfg.params.party_match["tie_delta"],
    out_dir=str(AUDIT),
)

# Merge party IDs onto sources
sources_parties = sources_nelda.merge(
    party_match.rename(columns={
        "party_name_raw": "party_1_name_raw",
        "partyfacts_id": "party_1_id",
        "match_score": "party_1_match_score",
        "match_method": "party_1_match_method",
        "match_status": "party_1_match_status",
    }),
    on=["iso3", "election_year", "party_1_name_raw"],
    how="left",
)

sources_parties = sources_parties.merge(
    party_match.rename(columns={
        "party_name_raw": "party_2_name_raw",
        "partyfacts_id": "party_2_id",
        "match_score": "party_2_match_score",
        "match_method": "party_2_match_method",
        "match_status": "party_2_match_status",
    }),
    on=["iso3", "election_year", "party_2_name_raw"],
    how="left",
)

{
    "party_1_missing_pct": sources_parties["party_1_id"].isna().mean(),
    "party_2_missing_pct": sources_parties["party_2_id"].isna().mean(),
}

# %% [markdown]
# ## 10) Build ideology observation table
#
# Combine V-Party, CHES, ParlGov, ELFF into a unified ideology observation table.

# %%
party_crosswalk = build_party_crosswalk(pf_ext, ["vparty", "ches", "parlgov"])
party_crosswalk = party_crosswalk[PARTY_CROSSWALK_KEEP]
party_crosswalk.to_parquet(DATA_OUT / "party_crosswalk.parquet", index=False)

parlgov_pos = parlgov.get("viewcalc_party_position.csv")

ideology_obs = build_ideology_obs(
    vparty=vparty_raw,
    ches=ches,
    elff=elff,
    parlgov_pos=parlgov_pos,
    pf_core=pf_core,
    pf_external=pf_ext,
    country_map=country,
    audit_dir=AUDIT,
)
ideology_obs = ideology_obs[PARTY_IDEOLOGY_KEEP]
ideology_obs.to_parquet(DATA_OUT / "party_ideology_obs.parquet", index=False)

ideology_obs["source"].value_counts(dropna=False)

# %% [markdown]
# ## 11) Assign ideology to elections + compute margin_market

# %%
sources_ideo = assign_ideology_to_elections(
    sources_parties,
    ideology_obs,
    source_priority=cfg.params.ideology["source_priority"],
    year_windows=cfg.params.ideology["year_windows"],
)

sources_ideo[["ideo_source", "margin_market", "D_market_win"]].head()

# %% [markdown]
# ## 12) Deduplicate to canonical election events
#
# Deterministic dedup by iso3 + office + date precision, with source rank preference.

# %%
sources_dedup, canonical = dedupe_events(sources_ideo, cfg.params.dedupe["source_rank"])

# Attach primary/secondary shares (CLEA vote share preferred when aligned)
canonical = attach_primary_secondary_shares(canonical, sources_dedup)

{
    "sources_rows": len(sources_dedup),
    "canonical_rows": len(canonical),
    "canonical_unique_election_id": canonical["election_id"].nunique(),
}

# %% [markdown]
# ## 13) Controls merge (DPI, IDEA, DES)
#
# Merge DPI on baseline year, IDEA on election year, DES on iso3-year.

# %%
# Prepare DPI ISO3 mapping
dpi_map = map_names_to_iso(dpi["countryname"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
dpi = dpi.merge(dpi_map[["source_name", "iso3"]], left_on="countryname", right_on="source_name", how="left")
dpi = dpi.drop_duplicates(subset=["iso3", "year"])

# Prepare DES
if des is not None:
    des["date"] = pd.to_datetime(des["date"], errors="coerce")
    des["election_year"] = des["date"].dt.year.astype("Int64")
    des_map = map_names_to_iso(des["country"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
    des = des.merge(des_map[["source_name", "iso3", "match_status"]], left_on="country", right_on="source_name", how="left")
    des = des.rename(columns={"iso3": "des_iso3"})
    des = des.sort_values(["des_iso3", "election_year", "date"], ascending=[True, True, True], kind="mergesort")
    des = des.drop_duplicates(subset=["des_iso3", "election_year"], keep="first")

# Prepare IDEA
idea = idea.drop_duplicates(subset=["ISO3", "Year"])

# Merge controls (replicating scripts/build_all.py logic)
canonical_controls = canonical.copy()
canonical_controls["dpi_year"] = canonical_controls["election_year"].astype("Int64") - 1
canonical_controls = canonical_controls.merge(dpi, left_on=["iso3", "dpi_year"], right_on=["iso3", "year"], how="left", suffixes=("", "_dpi"))
canonical_controls = canonical_controls.rename(columns={
    "system": "dpi_system",
    "execme": "dpi_execme",
    "checks": "dpi_checks",
    "checks_lax": "dpi_checks_lax",
    "military": "dpi_military",
    "execrlc": "dpi_execrlc",
    "gov1rlc": "dpi_gov1rlc",
    "opp1rlc": "dpi_opp1rlc",
})
canonical_controls["dpi_match_status"] = "matched"
canonical_controls.loc[canonical_controls["dpi_system"].isna(), "dpi_match_status"] = "unmatched"

idea = idea.rename(columns={
    "ISO3": "iso3",
    "Year": "idea_year",
    "Electoral system family": "idea_es_family",
    "Electoral system for national legislature": "idea_legislative_system",
    "Electoral system for the president": "idea_presidential_system",
})
idea["iso3"] = idea["iso3"].astype(str).str.upper()
idea["idea_year"] = pd.to_numeric(idea["idea_year"], errors="coerce").astype("Int64")
canonical_controls = canonical_controls.merge(
    idea[["iso3", "idea_year", "idea_es_family", "idea_legislative_system", "idea_presidential_system"]],
    left_on=["iso3", "election_year"],
    right_on=["iso3", "idea_year"],
    how="left",
)
canonical_controls["idea_match_status"] = "matched"
canonical_controls.loc[canonical_controls["idea_es_family"].isna(), "idea_match_status"] = "unmatched"
for col in ["idea_es_family", "idea_legislative_system", "idea_presidential_system"]:
    if col in canonical_controls.columns:
        canonical_controls[col] = canonical_controls[col].astype("string")

if des is not None:
    canonical_controls = canonical_controls.merge(
        des,
        left_on=["iso3", "election_year"],
        right_on=["des_iso3", "election_year"],
        how="left",
        suffixes=("", "_des"),
    )
    canonical_controls["des_match_status"] = "matched"
    canonical_controls.loc[canonical_controls["des_iso3"].isna(), "des_match_status"] = "unmatched"

# %% [markdown]
# ## 14) EFW alignment (baseline + horizons)
#
# Merge baseline EFW and build horizon table.

# %%
efwp = prepare_efw_panel(efwp_raw)
canonical_efw = attach_efw_baseline(canonical_controls, efwp)

horizons = cfg.params.efw["horizons"]
horizon = build_efw_horizons(canonical_efw, efwp, horizons)

{
    "canonical_with_efw_rows": len(canonical_efw),
    "horizon_rows": len(horizon),
}

# %% [markdown]
# ## 15) Export final trimmed outputs (keep-lists)
#
# Apply keep-lists and write final `data/processed/*` outputs.

# %%
country_out = country[COUNTRY_KEEP]
party_out = pf_core.rename(columns={"country": "iso3"})[PARTY_KEEP]

sources_out = sources_dedup[ELECTION_SOURCES_KEEP]
events_out = canonical_efw[ELECTION_EVENT_KEEP]
horizon_out = horizon[ELECTION_HORIZON_KEEP]

country_out.to_parquet(DATA_OUT / "country.parquet", index=False)
party_out.to_parquet(DATA_OUT / "party.parquet", index=False)
party_crosswalk.to_parquet(DATA_OUT / "party_crosswalk.parquet", index=False)
ideology_obs.to_parquet(DATA_OUT / "party_ideology_obs.parquet", index=False)
sources_out.to_parquet(DATA_OUT / "election_event_sources.parquet", index=False)
events_out.to_parquet(DATA_OUT / "election_event.parquet", index=False)
horizon_out.to_parquet(DATA_OUT / "election_event_horizon_efw.parquet", index=False)

{
    "country_rows": len(country_out),
    "party_rows": len(party_out),
    "party_crosswalk_rows": len(party_crosswalk),
    "party_ideology_obs_rows": len(ideology_obs),
    "sources_rows": len(sources_out),
    "events_rows": len(events_out),
    "horizon_rows": len(horizon_out),
}

# %% [markdown]
# ## 16) Diagnostics: missingness and coverage

# %%
events_2000 = events_out[events_out["election_year"] >= 2000]
{
    "rows_2000_plus": len(events_2000),
    "iso3_missing_pct": events_2000["iso3"].isna().mean(),
    "share_1_missing_pct": events_2000["share_1"].isna().mean(),
    "party_1_id_missing_pct": events_2000["party_1_id"].isna().mean(),
    "margin_market_missing_pct": events_2000["margin_market"].isna().mean(),
}

# %% [markdown]
# ## 17) Manual fill lists

# %%
unmatched_files = list(AUDIT.glob("04_country_mapping_*_unmatched.csv"))
{f.name: len(pd.read_csv(f)) for f in unmatched_files}

# %%
party_unmatched = AUDIT / "10_party_match_unmatched.csv"
party_ambiguous = AUDIT / "10_party_match_ambiguous.csv"
{
    "party_unmatched_rows": len(pd.read_csv(party_unmatched)) if party_unmatched.exists() else None,
    "party_ambiguous_rows": len(pd.read_csv(party_ambiguous)) if party_ambiguous.exists() else None,
}

# %% [markdown]
# ## 18) Full report reference
#
# The full narrative report is in `reports/cleanroom_final_report.md`.

# %%
REPORTS.joinpath("cleanroom_final_report.md").exists()
