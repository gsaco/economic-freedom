from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from elections.config import load_config
from elections.audit import audit_rowcount, write_output_manifest
from elections.build_sources import (
    SourcePaths,
    write_raw_manifest,
    build_sources,
)
from elections.country_map import build_iso_reference, map_names_to_iso, build_cowcode_map, build_country_table
from elections.io import (
    read_ned_pres,
    read_ned_parl,
    read_clea_lc,
    read_vparty,
    read_cow2iso,
    read_efw_panel,
    read_partyfacts_core,
    read_partyfacts_external,
)
from elections.party_match import build_party_match_results
from elections.ideology import build_party_crosswalk, build_ideology_obs, assign_ideology_to_elections
from elections.dedupe import dedupe_events, attach_primary_secondary_shares
from elections.merge_ledger import MergeLedger, logged_merge, logged_match
from elections.validate import validate_country, validate_election_event, validate_party
from elections.source_registry import build_source_registry
from elections.patches import apply_patches
from elections.keep_lists import (
    COUNTRY_KEEP,
    PARTY_KEEP,
    PARTY_CROSSWALK_KEEP,
    PARTY_IDEOLOGY_KEEP,
    ELECTION_SOURCES_KEEP,
    ELECTION_EVENT_KEEP,
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
    return df


def make_unique_columns(df: pd.DataFrame, audit_path: Path | None = None, table_name: str | None = None) -> pd.DataFrame:
    counts: dict[str, int] = {}
    new_cols: list[str] = []
    records: list[dict] = []
    for col in df.columns:
        counts[col] = counts.get(col, 0) + 1
        if counts[col] == 1:
            new_cols.append(col)
        else:
            new_name = f"{col}__dup{counts[col]}"
            new_cols.append(new_name)
            records.append({
                "table": table_name or "",
                "original": col,
                "new": new_name,
                "occurrence": counts[col],
            })
    out = df.copy()
    out.columns = new_cols
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records if records else [{"table": table_name or "", "original": "", "new": "", "occurrence": 0}]).to_csv(audit_path, index=False)
    return out


def merge_party_ids(
    sources: pd.DataFrame,
    party_matches: pd.DataFrame,
    ledger: MergeLedger | None = None,
) -> pd.DataFrame:
    df = sources.copy()

    pm1 = party_matches.rename(columns={
        "party_name_raw": "party_1_name_raw",
        "partyfacts_id": "party_1_id",
        "partyfacts_id_best": "party_1_id_best",
        "partyfacts_id_candidates": "party_1_id_candidates",
        "match_score": "party_1_match_score",
        "match_method": "party_1_match_method",
        "match_status": "party_1_match_status",
        "match_status_best": "party_1_match_status_best",
        "match_variant": "party_1_match_variant",
        "match_variant_best": "party_1_match_variant_best",
        "alias_source_key": "party_1_alias_source",
    })
    if ledger is None:
        df = df.merge(pm1, on=["iso3", "election_year", "party_1_name_raw"], how="left")
    else:
        df = logged_merge(
            ledger,
            df,
            pm1,
            how="left",
            on=["iso3", "election_year", "party_1_name_raw"],
            step_id="10_party_match_p1",
            step_name="Merge party matches for party_1",
            left_table="election_event_sources",
            right_table="party_match",
            context_cols_left=["record_id", "party_1_name_raw"],
            context_cols_right=["party_1_id", "party_1_match_status", "party_1_match_score"],
        )

    pm2 = party_matches.rename(columns={
        "party_name_raw": "party_2_name_raw",
        "partyfacts_id": "party_2_id",
        "partyfacts_id_best": "party_2_id_best",
        "partyfacts_id_candidates": "party_2_id_candidates",
        "match_score": "party_2_match_score",
        "match_method": "party_2_match_method",
        "match_status": "party_2_match_status",
        "match_status_best": "party_2_match_status_best",
        "match_variant": "party_2_match_variant",
        "match_variant_best": "party_2_match_variant_best",
        "alias_source_key": "party_2_alias_source",
    })
    if ledger is None:
        df = df.merge(pm2, on=["iso3", "election_year", "party_2_name_raw"], how="left")
    else:
        df = logged_merge(
            ledger,
            df,
            pm2,
            how="left",
            on=["iso3", "election_year", "party_2_name_raw"],
            step_id="10_party_match_p2",
            step_name="Merge party matches for party_2",
            left_table="election_event_sources",
            right_table="party_match",
            context_cols_left=["record_id", "party_2_name_raw"],
            context_cols_right=["party_2_id", "party_2_match_status", "party_2_match_score"],
        )
    return df


def main(argv: list[str] | None = None) -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="Cleanroom elections pipeline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.audit / "runs" / run_id
    ledger = MergeLedger(run_id=run_id, out_dir=run_dir)

    # ensure dirs
    for p in [cfg.paths.raw, cfg.paths.external, cfg.paths.interim, cfg.paths.processed, cfg.paths.reports, cfg.paths.audit, cfg.paths.manifests]:
        p.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: raw manifest + source registry
    efw_path = None
    efw_candidates: list[Path] = []
    for root in [args.root, cfg.paths.external, cfg.paths.raw, args.root / "data"]:
        if root.exists():
            efw_candidates += sorted(root.rglob("*efw*.xls*"))
            efw_candidates += sorted(root.rglob("*EFW*.xls*"))
    # dedupe preserving order
    seen = set()
    efw_candidates = [p for p in efw_candidates if not (p in seen or seen.add(p))]
    if efw_candidates:
        efw_path = efw_candidates[0]

    write_raw_manifest(
        cfg.paths.raw,
        cfg.paths.external,
        run_dir / "raw_files_manifest.csv",
        extra_paths=[p for p in [cfg.paths.cow2iso, cfg.paths.partyfacts_core, cfg.paths.partyfacts_external, efw_path] if p],
        root_dir=args.root,
        run_id=run_id,
    )
    build_source_registry(cfg.paths.raw, cfg.paths.external, cfg.paths.reports / "source_registry.csv")
    config_src = args.root / "config" / "config.yaml"
    if config_src.exists():
        (run_dir / "config.yaml").write_text(config_src.read_text())

    # Read datasets for mapping
    ned_pres = read_ned_pres(cfg.paths.raw / "ned" / "presidential_elections_v2.dta")
    ned_parl = read_ned_parl(cfg.paths.raw / "ned" / "parliamentary_elections_v2.dta")
    ned_names = pd.concat([ned_pres[["country"]], ned_parl[["country"]]], ignore_index=True)

    clea_raw = read_clea_lc(cfg.paths.raw / "clea" / "clea_lc_20251015.sav")
    vparty = read_vparty(cfg.paths.raw / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv")
    cow2iso = read_cow2iso(cfg.paths.cow2iso)
    # Optional inputs removed from the streamlined pipeline
    nelda = pd.DataFrame(columns=["country", "ccode"])
    dpi = pd.DataFrame(columns=["countryname", "ifs"])
    if efw_path is not None and efw_path.exists():
        efw = read_efw_panel(efw_path)
    else:
        efw = pd.DataFrame(columns=["Countries"])

    overrides_path = args.root / "data" / "manual" / "country_name_overrides.csv"
    overrides = pd.read_csv(overrides_path) if overrides_path.exists() else None

    iso_ref = build_iso_reference(cfg.paths.external / "parlgov.zip", cow2iso=cow2iso, manual_path=args.root / "data" / "manual" / "iso_reference_manual.csv")

    # build name maps
    name_maps: dict[str, pd.DataFrame] = {}
    name_maps["ned"] = map_names_to_iso(ned_names["country"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"], overrides=overrides, source_system="NED")
    logged_match(
        ledger,
        name_maps["ned"],
        step_id="04_country_map_ned",
        step_name="Country map NED names",
        left_table="ned_names",
        right_table="iso_ref",
        key_cols=["source_name"],
        context_cols=["name_norm", "iso3", "match_score", "match_method", "match_status", "override_applied"],
    )
    name_maps["clea"] = map_names_to_iso(clea_raw["ctr_n"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"], overrides=overrides, source_system="CLEA")
    logged_match(
        ledger,
        name_maps["clea"],
        step_id="04_country_map_clea",
        step_name="Country map CLEA names",
        left_table="clea_names",
        right_table="iso_ref",
        key_cols=["source_name"],
        context_cols=["name_norm", "iso3", "match_score", "match_method", "match_status", "override_applied"],
    )
    if "country_name" in vparty.columns:
        name_maps["vparty"] = map_names_to_iso(vparty["country_name"], iso_ref, cfg.params.country_match["min_score"], cfg.params.country_match["tie_delta"], overrides=overrides, source_system="VPARTY")
        logged_match(
            ledger,
            name_maps["vparty"],
            step_id="04_country_map_vparty",
            step_name="Country map V-Party names",
            left_table="vparty",
            right_table="iso_ref",
            key_cols=["source_name"],
            context_cols=["name_norm", "iso3", "match_score", "match_method", "match_status", "override_applied"],
        )

    ned_name_map = name_maps["ned"]
    clea_name_map = name_maps["clea"]

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
        audit_dir=run_dir,
        overrides=overrides,
        name_maps=name_maps,
        ledger=ledger,
    )
    country = apply_patches(country, cfg.paths.manual_patch, "country")
    country_full = country.copy()
    country_full = make_unique_columns(country_full, run_dir / "column_dedup_country.csv", "country")
    country_full.to_parquet(cfg.paths.processed / "country_full.parquet", index=False)
    country_view = country_full[COUNTRY_KEEP]
    country_view.to_parquet(cfg.paths.processed / "country.parquet", index=False)
    audit_rowcount(country_view, "country", "04_country", run_dir / "rowcount_coverage_audit.csv", key_cols=["iso3"], run_id=run_id)

    # Build source records
    spaths = SourcePaths(cfg.paths.raw, cfg.paths.interim, run_dir)
    sources = build_sources(spaths, cow_map, ned_name_map, clea_name_map, iso_ref, ledger=ledger)
    # add cow_code from country table when missing
    sources = logged_merge(
        ledger,
        sources,
        country[["iso3", "cow_code"]],
        how="left",
        on=["iso3"],
        step_id="08_sources_country",
        step_name="Attach country codes to sources",
        left_table="election_event_sources_base",
        right_table="country",
        context_cols_left=["record_id", "country_name_raw"],
        context_cols_right=["iso3", "cow_code"],
    )
    audit_rowcount(sources, "election_event_sources_base", "08_sources", run_dir / "rowcount_coverage_audit.csv", key_cols=["record_id"], run_id=run_id)

    # Clean flags (no NELDA merge in streamlined pipeline)
    sources_clean = apply_clean_flags(sources)
    audit_rowcount(sources_clean, "election_event_sources_base", "09_clean_flags", run_dir / "rowcount_coverage_audit.csv", key_cols=["record_id"], run_id=run_id)

    # Party matching
    pf_core = read_partyfacts_core(cfg.paths.partyfacts_core)
    pf_external = read_partyfacts_external(cfg.paths.partyfacts_external)
    party_match = build_party_match_results(
        sources_clean,
        pf_core,
        partyfacts_external=pf_external,
        min_score=cfg.params.party_match["min_score"],
        tie_delta=cfg.params.party_match["tie_delta"],
        year_window=cfg.params.party_match.get("year_window"),
        vparty=vparty,
        country_map=country,
        overrides=pd.read_csv(args.root / "data" / "manual" / "party_match_overrides.csv") if (args.root / "data" / "manual" / "party_match_overrides.csv").exists() else None,
        out_dir=str(run_dir),
        ledger=ledger,
        step_id="10_party_match",
    )
    party_match = apply_patches(party_match, cfg.paths.manual_patch, "party_match")

    sources_parties = merge_party_ids(sources_clean, party_match, ledger=ledger)
    audit_rowcount(sources_parties, "election_event_sources_party", "10_party_match", run_dir / "rowcount_coverage_audit.csv", key_cols=["record_id"], run_id=run_id)

    # Ideology obs (V-Party only)
    sources_enabled = cfg.params.ideology.get("sources_enabled", cfg.params.ideology.get("source_priority", ["vparty"]))
    ches = pd.DataFrame(columns=["party_id", "year", "lrecon"])
    elff = pd.DataFrame(columns=["countryname", "partyname", "year", "econlr"])
    parlgov_pos = pd.DataFrame(columns=["party_id", "state_market"])

    party_crosswalk = build_party_crosswalk(pf_external, ["vparty"])
    party_crosswalk_full = party_crosswalk.copy()
    party_crosswalk_full = make_unique_columns(party_crosswalk_full, run_dir / "column_dedup_party_crosswalk.csv", "party_crosswalk")
    party_crosswalk_full.to_parquet(cfg.paths.processed / "party_crosswalk_full.parquet", index=False)
    party_crosswalk_view = party_crosswalk_full[PARTY_CROSSWALK_KEEP]
    party_crosswalk_view.to_parquet(cfg.paths.processed / "party_crosswalk.parquet", index=False)

    ideology_obs = build_ideology_obs(
        vparty=vparty,
        ches=ches,
        elff=elff,
        parlgov_pos=parlgov_pos,
        pf_core=pf_core,
        pf_external=pf_external,
        country_map=country,
        audit_dir=run_dir,
        sources_enabled=sources_enabled,
    )
    ideology_obs_full = ideology_obs.copy()
    ideology_obs_full = make_unique_columns(ideology_obs_full, run_dir / "column_dedup_party_ideology_obs.csv", "party_ideology_obs")
    ideology_obs_full.to_parquet(cfg.paths.processed / "party_ideology_obs_full.parquet", index=False)
    ideology_obs_view = ideology_obs_full[PARTY_IDEOLOGY_KEEP]
    ideology_obs_view.to_parquet(cfg.paths.processed / "party_ideology_obs.parquet", index=False)

    # Assign ideology + margin_market
    sources_ideo = assign_ideology_to_elections(
        sources_parties,
        ideology_obs,
        source_priority=cfg.params.ideology["source_priority"],
        year_windows=cfg.params.ideology["year_windows"],
        stale_after_years=cfg.params.ideology.get("stale_after_years"),
    )
    audit_rowcount(sources_ideo, "election_event_sources_ideo", "12_ideology", run_dir / "rowcount_coverage_audit.csv", key_cols=["record_id"], run_id=run_id)

    # Dedup
    sources_ideo = apply_patches(sources_ideo, cfg.paths.manual_patch, "election_event_sources")
    sources_ideo.to_parquet(cfg.paths.processed / "election_event_sources_raw.parquet", index=False)
    sources_dedup, canonical = dedupe_events(
        sources_ideo,
        cfg.params.dedupe["source_rank"],
        margin_diff_max=cfg.params.dedupe.get("margin_diff_max", 5.0),
        audit_dir=str(run_dir),
        run_id=run_id,
    )
    sources_dedup_full = sources_dedup.copy()
    sources_dedup_full = make_unique_columns(sources_dedup_full, run_dir / "column_dedup_election_event_sources.csv", "election_event_sources")
    sources_dedup_full.to_parquet(cfg.paths.processed / "election_event_sources_full.parquet", index=False)
    sources_dedup_view = sources_dedup_full[ELECTION_SOURCES_KEEP]
    sources_dedup_view.to_parquet(cfg.paths.processed / "election_event_sources.parquet", index=False)
    audit_rowcount(sources_dedup_view, "election_event_sources", "13_sources_dedup", run_dir / "rowcount_coverage_audit.csv", key_cols=["record_id"], run_id=run_id)

    # canonical share primary/secondary
    canonical = attach_primary_secondary_shares(canonical, sources_dedup_full)

    # Save canonical (no controls/EFW in streamlined pipeline)
    canonical_final = apply_patches(canonical, cfg.paths.manual_patch, "election_event")
    canonical_full = canonical_final.copy()
    canonical_full = make_unique_columns(canonical_full, run_dir / "column_dedup_election_event.csv", "election_event")
    canonical_full.to_parquet(cfg.paths.processed / "election_event_full.parquet", index=False)
    canonical_view = canonical_full[ELECTION_EVENT_KEEP]
    canonical_view.to_parquet(cfg.paths.processed / "election_event.parquet", index=False)
    audit_rowcount(canonical_view, "election_event", "13_canonical", run_dir / "rowcount_coverage_audit.csv", key_cols=["election_id"], run_id=run_id)

    # party table
    party = pf_core.rename(columns={"country": "iso3"})
    party_full = party.copy()
    party_full = make_unique_columns(party_full, run_dir / "column_dedup_party.csv", "party")
    party_full.to_parquet(cfg.paths.processed / "party_full.parquet", index=False)
    party_view = party_full[PARTY_KEEP]
    party_view.to_parquet(cfg.paths.processed / "party.parquet", index=False)
    audit_rowcount(party_view, "party", "party", run_dir / "rowcount_coverage_audit.csv", key_cols=["partyfacts_id"], run_id=run_id)

    write_output_manifest(cfg.paths.processed, run_dir / "outputs_manifest.csv", root_dir=args.root, run_id=run_id)

    # validations
    errors = []
    errors += validate_country(country)
    errors += validate_party(party)
    errors += validate_election_event(canonical_view)
    if errors:
        err_path = run_dir / "validation_errors.txt"
        err_path.write_text("\n".join(errors))
        logger.error("Validation errors found. See %s", err_path)
        return 1

    logger.info("Build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
