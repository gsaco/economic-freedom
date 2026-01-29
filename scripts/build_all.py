from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from elections.config import load_config
from elections.audit import audit_rowcount
from elections.build_sources import (
    SourcePaths,
    write_raw_manifest,
    build_sources,
    merge_nelda,
)
from elections.country_map import build_iso_reference, map_names_to_iso, build_cowcode_map, build_country_table
from elections.io import (
    read_ned_pres,
    read_ned_parl,
    read_nelda,
    read_clea_lc,
    read_vparty,
    read_dpi,
    read_efw_panel,
    read_cow2iso,
    read_partyfacts_core,
    read_partyfacts_external,
    read_ches,
    read_elff,
    read_parlgov_zip,
    read_des_zip,
    read_idea,
)
from elections.party_match import build_party_match_results
from elections.ideology import build_party_crosswalk, build_ideology_obs, assign_ideology_to_elections
from elections.dedupe import dedupe_events, attach_primary_secondary_shares
from elections.efw_align import prepare_efw_panel, attach_efw_baseline, build_efw_horizons
from elections.validate import validate_country, validate_election_event, validate_horizon, validate_party
from elections.source_registry import build_source_registry
from elections.patches import apply_patches
from elections.keep_lists import (
    COUNTRY_KEEP,
    PARTY_KEEP,
    PARTY_CROSSWALK_KEEP,
    PARTY_IDEOLOGY_KEEP,
    ELECTION_SOURCES_KEEP,
    ELECTION_EVENT_KEEP,
    ELECTION_HORIZON_KEEP,
)


logger = logging.getLogger("cleanroom_pipeline")


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def apply_clean_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["flag_coup", "flag_inconsequential", "flag_unopposed", "flag_indirect"]:
        if col in df.columns:
            df[col] = df[col].fillna(False)
    df["clean_flag"] = (~df.get("flag_coup", False)) & (~df.get("flag_inconsequential", False)) & (~df.get("flag_unopposed", False)) & (~df.get("flag_indirect", False))

    # Competitive flag based on NELDA
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


def merge_party_ids(sources: pd.DataFrame, party_matches: pd.DataFrame) -> pd.DataFrame:
    df = sources.copy()
    df = df.merge(
        party_matches.rename(columns={
            "party_name_raw": "party_1_name_raw",
            "partyfacts_id": "party_1_id",
            "match_score": "party_1_match_score",
            "match_method": "party_1_match_method",
            "match_status": "party_1_match_status",
        }),
        on=["iso3", "election_year", "party_1_name_raw"],
        how="left",
    )
    df = df.merge(
        party_matches.rename(columns={
            "party_name_raw": "party_2_name_raw",
            "partyfacts_id": "party_2_id",
            "match_score": "party_2_match_score",
            "match_method": "party_2_match_method",
            "match_status": "party_2_match_status",
        }),
        on=["iso3", "election_year", "party_2_name_raw"],
        how="left",
    )
    return df


def merge_controls(elections: pd.DataFrame, dpi: pd.DataFrame, idea: pd.DataFrame, des: pd.DataFrame | None) -> pd.DataFrame:
    df = elections.copy()

    # DPI
    dpi["year"] = pd.to_numeric(dpi["year"], errors="coerce").astype("Int64")
    df["dpi_year"] = df["election_year"].astype("Int64") - 1
    df = df.merge(dpi, left_on=["iso3", "dpi_year"], right_on=["iso3", "year"], how="left", suffixes=("", "_dpi"))
    df = df.rename(columns={
        "system": "dpi_system",
        "execme": "dpi_execme",
        "checks": "dpi_checks",
        "checks_lax": "dpi_checks_lax",
        "military": "dpi_military",
        "execrlc": "dpi_execrlc",
        "gov1rlc": "dpi_gov1rlc",
        "opp1rlc": "dpi_opp1rlc",
    })
    df["dpi_match_status"] = "matched"
    df.loc[df["dpi_system"].isna(), "dpi_match_status"] = "unmatched"

    # IDEA
    idea = idea.rename(columns={
        "ISO3": "iso3",
        "Year": "idea_year",
        "Electoral system family": "idea_es_family",
        "Electoral system for national legislature": "idea_legislative_system",
        "Electoral system for the president": "idea_presidential_system",
    })
    idea["iso3"] = idea["iso3"].astype(str).str.upper()
    idea["idea_year"] = pd.to_numeric(idea["idea_year"], errors="coerce").astype("Int64")
    df = df.merge(idea[["iso3", "idea_year", "idea_es_family", "idea_legislative_system", "idea_presidential_system"]], left_on=["iso3", "election_year"], right_on=["iso3", "idea_year"], how="left")
    df["idea_match_status"] = "matched"
    df.loc[df["idea_es_family"].isna(), "idea_match_status"] = "unmatched"
    for col in ["idea_es_family", "idea_legislative_system", "idea_presidential_system"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # DES (optional, prepared with des_iso3 + election_year)
    if des is not None:
        df = df.merge(des, left_on=["iso3", "election_year"], right_on=["des_iso3", "election_year"], how="left", suffixes=("", "_des"))
        df["des_match_status"] = "matched"
        df.loc[df["des_iso3"].isna(), "des_match_status"] = "unmatched"

    return df


def main(argv: list[str] | None = None) -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="Cleanroom elections pipeline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    cfg = load_config(args.root)

    # ensure dirs
    for p in [cfg.paths.raw, cfg.paths.external, cfg.paths.interim, cfg.paths.processed, cfg.paths.reports, cfg.paths.audit, cfg.paths.manifests]:
        p.mkdir(parents=True, exist_ok=True)

    # Step 0: raw manifest + source registry
    write_raw_manifest(
        cfg.paths.raw,
        cfg.paths.external,
        cfg.paths.manifests / "raw_files_manifest.csv",
        extra_paths=[cfg.paths.cow2iso, cfg.paths.partyfacts_core, cfg.paths.partyfacts_external],
    )
    build_source_registry(cfg.paths.raw, cfg.paths.external, cfg.paths.reports / "source_registry.csv")

    # Read datasets for mapping
    ned_pres = read_ned_pres(cfg.paths.raw / "ned" / "presidential_elections_v2.dta")
    ned_parl = read_ned_parl(cfg.paths.raw / "ned" / "parliamentary_elections_v2.dta")
    ned_names = pd.concat([ned_pres[["country"]], ned_parl[["country"]]], ignore_index=True)

    clea_raw = read_clea_lc(cfg.paths.raw / "clea" / "clea_lc_20251015.sav")
    nelda = read_nelda(cfg.paths.raw / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta")
    vparty = read_vparty(cfg.paths.raw / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv")
    cow2iso = read_cow2iso(cfg.paths.cow2iso)
    dpi = read_dpi(cfg.paths.external / "dpi" / "DPI2020" / "dpi2020.csv")
    efw = read_efw_panel(cfg.paths.raw / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx")

    iso_ref = build_iso_reference(cfg.paths.external / "parlgov.zip")

    # build name maps
    ned_name_map = map_names_to_iso(ned_names["country"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
    clea_name_map = map_names_to_iso(clea_raw["ctr_n"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])

    cow_map = build_cowcode_map(cow2iso, iso_ref=iso_ref)

    country = build_country_table(
        iso_ref=iso_ref,
        ned=ned_names.assign(country=ned_names["country"]),
        clea=clea_raw,
        nelda=nelda,
        vparty=vparty,
        cow2iso=cow2iso,
        dpi=dpi,
        efw=efw,
        min_score=cfg.params.country_match["min_score"],
        tie_delta=cfg.params.country_match["tie_delta"],
        audit_dir=cfg.paths.audit,
    )
    country = apply_patches(country, cfg.paths.manual_patch, "country")
    country = country[COUNTRY_KEEP]
    country.to_parquet(cfg.paths.processed / "country.parquet", index=False)
    audit_rowcount(country, "country", "04_country", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["iso3"])

    # Build source records
    spaths = SourcePaths(cfg.paths.raw, cfg.paths.interim, cfg.paths.audit)
    sources = build_sources(spaths, cow_map, ned_name_map, clea_name_map, iso_ref)
    # add cow_code from country table when missing
    sources = sources.merge(country[["iso3", "cow_code"]], on="iso3", how="left", suffixes=("", "_country"))
    audit_rowcount(sources, "election_event_sources_base", "08_sources", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["record_id"])

    # Merge NELDA
    sources_nelda, nelda_report = merge_nelda(sources, nelda)
    nelda_report.to_csv(cfg.paths.audit / "09_nelda_match_report.csv", index=False)

    sources_nelda = apply_clean_flags(sources_nelda)
    audit_rowcount(sources_nelda, "election_event_sources_base", "09_nelda", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["record_id"])

    # Party matching
    pf_core = read_partyfacts_core(cfg.paths.partyfacts_core)
    pf_external = read_partyfacts_external(cfg.paths.partyfacts_external)
    party_match = build_party_match_results(
        sources_nelda,
        pf_core,
        partyfacts_external=pf_external,
        min_score=cfg.params.party_match["min_score"],
        tie_delta=cfg.params.party_match["tie_delta"],
        out_dir=str(cfg.paths.audit),
    )
    party_match = apply_patches(party_match, cfg.paths.manual_patch, "party_match")

    sources_parties = merge_party_ids(sources_nelda, party_match)
    audit_rowcount(sources_parties, "election_event_sources_party", "10_party_match", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["record_id"])

    # Ideology obs
    ches = read_ches(cfg.paths.external / "ches_1999_2024.csv")
    elff = read_elff(cfg.paths.external / "elff_partypos_summaries.csv")
    parlgov = read_parlgov_zip(cfg.paths.external / "parlgov.zip")
    parlgov_pos = parlgov.get("viewcalc_party_position.csv")

    party_crosswalk = build_party_crosswalk(pf_external, ["vparty", "ches", "parlgov"])
    party_crosswalk = party_crosswalk[PARTY_CROSSWALK_KEEP]
    party_crosswalk.to_parquet(cfg.paths.processed / "party_crosswalk.parquet", index=False)

    ideology_obs = build_ideology_obs(
        vparty=vparty,
        ches=ches,
        elff=elff,
        parlgov_pos=parlgov_pos,
        pf_core=pf_core,
        pf_external=pf_external,
        country_map=country,
        audit_dir=cfg.paths.audit,
    )
    ideology_obs = ideology_obs[PARTY_IDEOLOGY_KEEP]
    ideology_obs.to_parquet(cfg.paths.processed / "party_ideology_obs.parquet", index=False)

    # Assign ideology + margin_market
    sources_ideo = assign_ideology_to_elections(
        sources_parties,
        ideology_obs,
        source_priority=cfg.params.ideology["source_priority"],
        year_windows=cfg.params.ideology["year_windows"],
    )
    audit_rowcount(sources_ideo, "election_event_sources_ideo", "12_ideology", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["record_id"])

    # Dedup
    sources_ideo = apply_patches(sources_ideo, cfg.paths.manual_patch, "election_event_sources")
    sources_ideo.to_parquet(cfg.paths.processed / "election_event_sources_raw.parquet", index=False)
    sources_dedup, canonical = dedupe_events(sources_ideo, cfg.params.dedupe["source_rank"])
    sources_dedup = sources_dedup[ELECTION_SOURCES_KEEP]
    sources_dedup.to_parquet(cfg.paths.processed / "election_event_sources.parquet", index=False)
    audit_rowcount(sources_dedup, "election_event_sources", "13_sources_dedup", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["record_id"])

    # canonical share primary/secondary
    canonical = attach_primary_secondary_shares(canonical, sources_dedup)

    # Controls merge
    # map DPI country names to iso3
    dpi_map = map_names_to_iso(dpi["countryname"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
    dpi = dpi.merge(dpi_map[["source_name", "iso3"]], left_on="countryname", right_on="source_name", how="left")
    dpi = dpi.drop_duplicates(subset=["iso3", "year"])
    des = read_des_zip(cfg.paths.external / "des_es_data_v50.zip")
    des["date"] = pd.to_datetime(des["date"], errors="coerce")
    des["election_year"] = des["date"].dt.year.astype("Int64")
    des_map = map_names_to_iso(des["country"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"])
    des = des.merge(des_map[["source_name", "iso3", "match_status"]], left_on="country", right_on="source_name", how="left")
    des = des.rename(columns={"iso3": "des_iso3"})
    dupes = des[des.duplicated(["des_iso3", "election_year"], keep=False)]
    if not dupes.empty:
        dupes.to_csv(cfg.paths.audit / "14_des_duplicates.csv", index=False)
    des = des.sort_values(["des_iso3", "election_year", "date"], ascending=[True, True, True], kind="mergesort")
    des = des.drop_duplicates(subset=["des_iso3", "election_year"], keep="first")

    idea = read_idea(cfg.paths.external / "idea_export_electoral_system_design_database.xlsx")
    idea = idea.drop_duplicates(subset=["ISO3", "Year"])

    canonical_controls = merge_controls(canonical, dpi, idea, des)

    # EFW alignment
    efw_panel = prepare_efw_panel(efw)
    canonical_efw = attach_efw_baseline(canonical_controls, efw_panel)
    canonical_efw = apply_patches(canonical_efw, cfg.paths.manual_patch, "election_event")

    # Save canonical
    canonical_efw = canonical_efw[ELECTION_EVENT_KEEP]
    canonical_efw.to_parquet(cfg.paths.processed / "election_event.parquet", index=False)
    audit_rowcount(canonical_efw, "election_event", "13_canonical", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["election_id"])

    # Horizon table
    horizons = cfg.params.efw["horizons"]
    horizon = build_efw_horizons(canonical_efw, efw_panel, horizons)
    horizon = horizon[ELECTION_HORIZON_KEEP]
    horizon.to_parquet(cfg.paths.processed / "election_event_horizon_efw.parquet", index=False)
    audit_rowcount(horizon, "election_event_horizon_efw", "15_efw", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["election_id", "horizon"])

    # party table
    party = pf_core.rename(columns={"country": "iso3"})
    party = party[PARTY_KEEP]
    party.to_parquet(cfg.paths.processed / "party.parquet", index=False)
    audit_rowcount(party, "party", "party", cfg.paths.audit / "rowcount_coverage_audit.csv", key_cols=["partyfacts_id"])

    # validations
    errors = []
    errors += validate_country(country)
    errors += validate_party(party)
    errors += validate_election_event(canonical_efw)
    errors += validate_horizon(horizon)
    if errors:
        err_path = cfg.paths.audit / "validation_errors.txt"
        err_path.write_text("\n".join(errors))
        logger.error("Validation errors found. See %s", err_path)
        return 1

    logger.info("Build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
