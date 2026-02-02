from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from .audit import audit_rowcount, write_simple_audit
from .io import read_ned_pres, read_ned_parl, read_nelda, read_clea_lc
from .standardize import parse_date_parts, date_precision_from_parts, standardize_share, reorder_top_two, normalize_name, parse_nelda_date
from .merge_ledger import MergeLedger, logged_merge


@dataclass
class SourcePaths:
    raw_dir: Path
    interim_dir: Path
    audit_dir: Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw_manifest(
    raw_root: Path,
    external_root: Path,
    out_path: Path,
    extra_paths: list[Path] | None = None,
    root_dir: Path | None = None,
    run_id: str | None = None,
) -> None:
    records = []
    def rel(p: Path) -> str:
        if root_dir is None:
            return p.as_posix()
        try:
            return p.resolve().relative_to(root_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()
    for root in [raw_root, external_root]:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                record = {
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                if run_id is not None:
                    record["run_id"] = run_id
                records.append(record)
    if extra_paths:
        for path in extra_paths:
            if path and Path(path).is_file():
                p = Path(path)
                record = {
                    "path": rel(p),
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
                if run_id is not None:
                    record["run_id"] = run_id
                records.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)


def ingest_ned_pres(paths: SourcePaths) -> pd.DataFrame:
    df = read_ned_pres(paths.raw_dir / "ned" / "presidential_elections_v2.dta")
    df = df.rename(columns={
        "party_1": "party_1_name_raw",
        "party_2": "party_2_name_raw",
        "candidate_1": "candidate_1_name_raw",
        "candidate_2": "candidate_2_name_raw",
        "type_election": "type_election_raw",
    })
    date, year, month, day = parse_date_parts(df.get("date"), df.get("year"), df.get("month"))
    df["election_date"] = date
    df["election_year"] = year
    df["election_month"] = month
    df["election_day"] = day
    df["date_precision"] = date_precision_from_parts(date, year, month, day)

    # decisive vote shares: use second round only if both top-two present
    has_second = df["vote_share2_1"].notna() & df["vote_share2_2"].notna()
    df["share_1_raw"] = df["vote_share1_1"]
    df["share_2_raw"] = df["vote_share1_2"]
    df.loc[has_second, "share_1_raw"] = df.loc[has_second, "vote_share2_1"]
    df.loc[has_second, "share_2_raw"] = df.loc[has_second, "vote_share2_2"]
    df["share_1"] = standardize_share(df["share_1_raw"])
    df["share_2"] = standardize_share(df["share_2_raw"])
    df["share_metric"] = "vote_share_final"

    # flags
    for col in ["flag_coup", "flag_inconsequential", "flag_unopposed", "flag_indirect", "flag_two_round"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).astype(bool)

    df["office_type"] = "presidential"
    df["source"] = "ned_pres"
    df["source_election_id"] = df.index.astype(int)

    # reorder top two
    df = reorder_top_two(
        df,
        ("share_1", "share_2"),
        [
            ("share_1", "share_2"),
            ("share_1_raw", "share_2_raw"),
            ("party_1_name_raw", "party_2_name_raw"),
            ("candidate_1_name_raw", "candidate_2_name_raw"),
        ],
    )
    df["margin"] = df["share_1"] - df["share_2"]

    keep_cols = [
        "source",
        "source_election_id",
        "country",
        "country_cow",
        "country_abb",
        "type_election_raw",
        "election_date",
        "date_precision",
        "election_year",
        "election_month",
        "election_day",
        "party_1_name_raw",
        "party_2_name_raw",
        "candidate_1_name_raw",
        "candidate_2_name_raw",
        "share_1",
        "share_2",
        "share_metric",
        "margin",
        "flag_coup",
        "flag_inconsequential",
        "flag_unopposed",
        "flag_indirect",
        "flag_two_round",
        "office_type",
    ]
    out = df[keep_cols].copy()
    out.to_parquet(paths.interim_dir / "elections_ned_pres.parquet", index=False)
    write_simple_audit(out, paths.audit_dir / "01_ned_pres_audit.csv")
    return out


def ingest_ned_parl(paths: SourcePaths) -> pd.DataFrame:
    df = read_ned_parl(paths.raw_dir / "ned" / "parliamentary_elections_v2.dta")
    df = df.rename(columns={
        "party_1": "party_1_name_raw",
        "party_2": "party_2_name_raw",
        "type_election": "type_election_raw",
    })
    date, year, month, day = parse_date_parts(df.get("date"), df.get("year"), df.get("month"))
    df["election_date"] = date
    df["election_year"] = year
    df["election_month"] = month
    df["election_day"] = day
    df["date_precision"] = date_precision_from_parts(date, year, month, day)

    df["share_1"] = standardize_share(df["seat_share_1"])
    df["share_2"] = standardize_share(df["seat_share_2"])
    df["share_metric"] = "seat_share"

    for col in ["flag_coup", "flag_inconsequential", "flag_constituent", "flag_vacant_seats"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).astype(bool)

    df["office_type"] = "parliamentary"
    df["source"] = "ned_parl"
    df["source_election_id"] = df.index.astype(int)

    df = reorder_top_two(
        df,
        ("share_1", "share_2"),
        [
            ("share_1", "share_2"),
            ("party_1_name_raw", "party_2_name_raw"),
        ],
    )
    df["margin"] = df["share_1"] - df["share_2"]

    keep_cols = [
        "source",
        "source_election_id",
        "country",
        "country_cow",
        "country_abb",
        "type_election_raw",
        "election_date",
        "date_precision",
        "election_year",
        "election_month",
        "election_day",
        "party_1_name_raw",
        "party_2_name_raw",
        "share_1",
        "share_2",
        "share_metric",
        "margin",
        "flag_coup",
        "flag_inconsequential",
        "flag_constituent",
        "flag_vacant_seats",
        "office_type",
    ]
    out = df[keep_cols].copy()
    out.to_parquet(paths.interim_dir / "elections_ned_parl.parquet", index=False)
    write_simple_audit(out, paths.audit_dir / "02_ned_parl_audit.csv")
    return out


def union_ned(ned_pres: pd.DataFrame, ned_parl: pd.DataFrame, paths: SourcePaths) -> pd.DataFrame:
    combined = pd.concat([ned_pres, ned_parl], ignore_index=True)
    combined.to_parquet(paths.interim_dir / "election_records_ned.parquet", index=False)
    write_simple_audit(combined, paths.audit_dir / "03_ned_union_audit.csv")
    return combined


def _clean_clea_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s >= 0, pd.NA)
    return s


def aggregate_clea_lc(paths: SourcePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_clea_lc(paths.raw_dir / "clea" / "clea_lc_20251015.sav")
    # clean sentinel values
    for col in ["pv1", "pvs1", "seat", "vv1", "ivv1", "cv1", "cvs1"]:
        if col in df.columns:
            df[col] = _clean_clea_numeric(df[col])
    if "pvs1" in df.columns and "vv1" in df.columns:
        df["pvs1_weighted"] = df["pvs1"] * df["vv1"]

    # aggregate votes and seats by election-party
    agg = df.groupby(["id", "ctr", "ctr_n", "yr", "mn", "pty", "pty_n"], dropna=False).agg(
        pv1_sum=("pv1", "sum"),
        seat_sum=("seat", "sum"),
        vv1_sum=("vv1", "sum"),
        pvs1_weighted_sum=("pvs1_weighted", "sum"),
    ).reset_index()

    # total votes/seats per election
    totals = agg.groupby(["id"], dropna=False).agg(
        total_votes=("pv1_sum", "sum"),
        total_seats=("seat_sum", "sum"),
        total_vv1=("vv1_sum", "sum"),
        total_pvs1_weighted=("pvs1_weighted_sum", "sum"),
    ).reset_index()

    agg = agg.merge(totals, on="id", how="left")
    agg["vote_share_counts"] = np.where(agg["total_votes"].notna() & (agg["total_votes"] > 0), (agg["pv1_sum"] / agg["total_votes"]) * 100.0, pd.NA)
    agg["vote_share_weighted"] = np.where(
        agg["total_pvs1_weighted"].notna() & (agg["total_pvs1_weighted"] > 0),
        (agg["pvs1_weighted_sum"] / agg["total_pvs1_weighted"]) * 100.0,
        pd.NA,
    )
    agg["seat_share"] = np.where(agg["total_seats"].notna() & (agg["total_seats"] > 0), (agg["seat_sum"] / agg["total_seats"]) * 100.0, pd.NA)

    agg["vote_share_method"] = "missing"
    mask_counts = agg["total_votes"].notna() & (agg["total_votes"] > 0)
    mask_weighted = (~mask_counts) & agg["total_pvs1_weighted"].notna() & (agg["total_pvs1_weighted"] > 0)
    mask_seat = (~mask_counts) & (~mask_weighted) & agg["total_seats"].notna() & (agg["total_seats"] > 0)
    agg.loc[mask_counts, "vote_share_method"] = "counts"
    agg.loc[mask_weighted, "vote_share_method"] = "pvs1_weighted"
    agg.loc[mask_seat, "vote_share_method"] = "seat_fallback"

    agg["share_for_rank"] = pd.NA
    agg.loc[mask_counts, "share_for_rank"] = agg.loc[mask_counts, "vote_share_counts"]
    agg.loc[mask_weighted, "share_for_rank"] = agg.loc[mask_weighted, "vote_share_weighted"]
    agg.loc[mask_seat, "share_for_rank"] = agg.loc[mask_seat, "seat_share"]

    agg.to_parquet(paths.interim_dir / "clea_lc_election_party.parquet", index=False)

    # select top two parties by vote share
    agg["pty_norm"] = agg["pty_n"].map(normalize_name)
    agg = agg.sort_values(["id", "share_for_rank", "pv1_sum", "pty_norm"], ascending=[True, False, False, True], kind="mergesort")
    top2 = agg.groupby("id", as_index=False).head(2).copy()

    # pivot to wide
    top2["rank"] = top2.groupby("id").cumcount() + 1
    wide = top2.pivot(
        index=["id", "ctr", "ctr_n", "yr", "mn"],
        columns="rank",
        values=["pty_n", "share_for_rank", "seat_share", "pv1_sum", "vv1_sum", "seat_sum"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    wide = wide.rename(columns={
        "pty_n_1": "party_1_name_raw",
        "pty_n_2": "party_2_name_raw",
        "share_for_rank_1": "share_1",
        "share_for_rank_2": "share_2",
        "seat_share_1": "seat_share_1",
        "seat_share_2": "seat_share_2",
    })
    method = agg[["id", "vote_share_method"]].drop_duplicates(subset=["id"])
    wide = wide.merge(method, on="id", how="left")

    # date
    date = pd.to_datetime({"year": wide["yr"].astype(int), "month": wide["mn"].astype(int), "day": 1}, errors="coerce")
    wide["election_date"] = date
    wide["election_year"] = wide["yr"].astype("Int64")
    wide["election_month"] = wide["mn"].astype("Int64")
    wide["election_day"] = pd.Series([pd.NA] * len(wide), dtype="Int64")
    wide["date_precision"] = "month"

    method_map = {
        "counts": "vote_share_agg",
        "pvs1_weighted": "vote_share_pvs1_weighted",
        "seat_fallback": "seat_share_fallback",
        "missing": "missing",
    }
    wide["share_metric"] = wide["vote_share_method"].map(method_map)
    wide["margin"] = wide["share_1"] - wide["share_2"]
    wide["office_type"] = "parliamentary"
    wide["source"] = "clea_lc"
    wide["source_election_id"] = wide["id"]

    keep_cols = [
        "source",
        "source_election_id",
        "ctr",
        "ctr_n",
        "election_date",
        "date_precision",
        "election_year",
        "election_month",
        "election_day",
        "party_1_name_raw",
        "party_2_name_raw",
        "share_1",
        "share_2",
        "share_metric",
        "vote_share_method",
        "margin",
        "office_type",
        "seat_share_1",
        "seat_share_2",
        "pv1_sum_1",
        "pv1_sum_2",
        "vv1_sum_1",
        "vv1_sum_2",
        "seat_sum_1",
        "seat_sum_2",
    ]
    out = wide[keep_cols].copy()
    out.to_parquet(paths.interim_dir / "elections_clea_lc.parquet", index=False)
    write_simple_audit(out, paths.audit_dir / "06_clea_aggregation_audit.csv")

    # validation audit
    issues = []
    invalid_share = out[(out["share_1"].notna() & ((out["share_1"] < 0) | (out["share_1"] > 100))) | (out["share_2"].notna() & ((out["share_2"] < 0) | (out["share_2"] > 100)))]
    if not invalid_share.empty:
        tmp = invalid_share.copy()
        tmp["issue"] = "share_out_of_bounds"
        issues.append(tmp)
    missing_top2 = out[out["share_1"].isna() & out["share_2"].isna()]
    if not missing_top2.empty:
        tmp = missing_top2.copy()
        tmp["issue"] = "top_two_missing"
        issues.append(tmp)
    party_counts = agg.groupby("id")["pty"].nunique()
    low_party = out[out["source_election_id"].map(party_counts) < 2]
    if not low_party.empty:
        tmp = low_party.copy()
        tmp["issue"] = "less_than_two_parties"
        issues.append(tmp)
    if issues:
        pd.concat(issues, ignore_index=True).to_csv(paths.audit_dir / "clea_agg_validation.csv", index=False)
    return agg, out


def attach_iso3_ned(
    ned: pd.DataFrame,
    cow_map: pd.DataFrame,
    name_map: pd.DataFrame,
    ledger: MergeLedger | None = None,
) -> pd.DataFrame:
    ned = ned.copy()
    cow_map = cow_map.copy()
    if "cow_id" in cow_map.columns and "cowcode" not in cow_map.columns:
        cow_map = cow_map.rename(columns={"cow_id": "cowcode"})
    cow_map["cowcode"] = pd.to_numeric(cow_map.get("cowcode"), errors="coerce").astype("Int64")
    if "iso3" in cow_map.columns:
        cow_map["iso3"] = cow_map["iso3"].astype(str).str.upper()
        cow_map.loc[cow_map["iso3"].isin(["", "NAN", "NONE"]), "iso3"] = pd.NA
    for col in ["valid_from", "valid_until"]:
        if col in cow_map.columns:
            cow_map[col] = pd.to_numeric(cow_map[col], errors="coerce").astype("Int64")

    # expand to candidates and pick best by year range
    ned["_row_id"] = range(len(ned))
    if ledger is None:
        candidates = ned.merge(cow_map, left_on="country_cow", right_on="cowcode", how="left")
    else:
        candidates = logged_merge(
            ledger,
            ned,
            cow_map,
            how="left",
            left_on=["country_cow"],
            right_on=["cowcode"],
            step_id="04_ned_cow_map",
            step_name="Attach COW map to NED",
            left_table="ned",
            right_table="cow2iso",
            context_cols_left=["country_cow", "election_year"],
            context_cols_right=["cowcode", "iso3"],
        )

    def pick_best(grp: pd.DataFrame) -> pd.Series:
        year = grp["election_year"].iloc[0]
        cand = grp.dropna(subset=["iso3"])
        if cand.empty:
            return pd.Series({"iso3_cow": pd.NA, "iso3_cow_match_status": "unmatched"})
        if pd.notna(year):
            vf = pd.to_numeric(cand.get("valid_from"), errors="coerce").astype(float).fillna(-np.inf)
            vu = pd.to_numeric(cand.get("valid_until"), errors="coerce").astype(float).fillna(np.inf)
            in_range = (year >= vf) & (year <= vu)
            cand = cand.loc[in_range]
            if cand.empty:
                return pd.Series({"iso3_cow": pd.NA, "iso3_cow_match_status": "out_of_range"})
            vf = pd.to_numeric(cand.get("valid_from"), errors="coerce").astype(float).fillna(-np.inf)
            max_vf = vf.max()
            top = cand.loc[vf == max_vf]
            if len(top) > 1:
                return pd.Series({"iso3_cow": pd.NA, "iso3_cow_match_status": "ambiguous"})
            return pd.Series({"iso3_cow": top["iso3"].iloc[0], "iso3_cow_match_status": "matched"})
        vf = pd.to_numeric(cand.get("valid_from"), errors="coerce").astype(float).fillna(-np.inf)
        max_vf = vf.max()
        top = cand.loc[vf == max_vf]
        if len(top) > 1:
            return pd.Series({"iso3_cow": pd.NA, "iso3_cow_match_status": "ambiguous_no_year"})
        return pd.Series({"iso3_cow": top["iso3"].iloc[0], "iso3_cow_match_status": "matched_no_year"})

    picked = candidates.groupby("_row_id", group_keys=False).apply(pick_best)
    ned = ned.merge(picked, left_on="_row_id", right_index=True, how="left")
    ned = ned.drop(columns=["_row_id"])

    if ledger is None:
        ned = ned.merge(name_map[["source_name", "iso3"]], left_on="country", right_on="source_name", how="left")
    else:
        ned = logged_merge(
            ledger,
            ned,
            name_map[["source_name", "iso3"]],
            how="left",
            left_on=["country"],
            right_on=["source_name"],
            step_id="04_ned_name_map_join",
            step_name="Attach ISO3 by name to NED",
            left_table="ned",
            right_table="ned_name_map",
            context_cols_left=["country"],
            context_cols_right=["source_name", "iso3"],
        )
    ned = ned.rename(columns={"iso3": "iso3_name"})
    ned["iso3"] = ned["iso3_cow"].combine_first(ned["iso3_name"])
    ned["iso3_match_conflict"] = (ned["iso3_cow"].notna()) & (ned["iso3_name"].notna()) & (ned["iso3_cow"] != ned["iso3_name"])
    return ned


def attach_iso3_clea(
    clea: pd.DataFrame,
    iso_ref: pd.DataFrame,
    name_map: pd.DataFrame,
    ledger: MergeLedger | None = None,
) -> pd.DataFrame:
    clea = clea.copy()
    # use ctr if it's 3-letter and in iso_ref
    ctr = clea["ctr"].astype("string")
    valid_iso = set(iso_ref["iso3"].astype(str))
    clea["iso3_ctr"] = ctr.where(ctr.str.len() == 3)
    clea.loc[~clea["iso3_ctr"].isin(valid_iso), "iso3_ctr"] = pd.NA
    if ledger is None:
        clea = clea.merge(name_map[["source_name", "iso3"]], left_on="ctr_n", right_on="source_name", how="left")
    else:
        clea = logged_merge(
            ledger,
            clea,
            name_map[["source_name", "iso3"]],
            how="left",
            left_on=["ctr_n"],
            right_on=["source_name"],
            step_id="04_clea_name_map_join",
            step_name="Attach ISO3 by name to CLEA",
            left_table="clea",
            right_table="clea_name_map",
            context_cols_left=["ctr_n", "ctr"],
            context_cols_right=["source_name", "iso3"],
        )
    clea = clea.rename(columns={"iso3": "iso3_name"})
    clea["iso3"] = clea["iso3_ctr"].combine_first(clea["iso3_name"])
    return clea


def build_sources(
    paths: SourcePaths,
    cow_map: pd.DataFrame,
    ned_name_map: pd.DataFrame,
    clea_name_map: pd.DataFrame,
    iso_ref: pd.DataFrame,
    ledger: MergeLedger | None = None,
) -> pd.DataFrame:
    ned_pres = ingest_ned_pres(paths)
    ned_parl = ingest_ned_parl(paths)
    ned = union_ned(ned_pres, ned_parl, paths)

    # attach iso3 for NED
    ned_iso = attach_iso3_ned(ned, cow_map=cow_map, name_map=ned_name_map, ledger=ledger)

    # CLEA aggregation
    _, clea = aggregate_clea_lc(paths)
    clea_iso = attach_iso3_clea(clea, iso_ref=iso_ref, name_map=clea_name_map, ledger=ledger)

    # align columns
    for col in ned_iso.columns:
        if col not in clea_iso.columns:
            clea_iso[col] = pd.NA
    for col in clea_iso.columns:
        if col not in ned_iso.columns:
            ned_iso[col] = pd.NA

    combined = pd.concat([ned_iso, clea_iso], ignore_index=True)
    combined["country_name_raw"] = combined.get("country").combine_first(combined.get("ctr_n"))
    combined["record_id"] = combined.apply(lambda r: f"{r['source']}:{r['source_election_id']}", axis=1)

    combined.to_parquet(paths.interim_dir / "election_event_sources_base.parquet", index=False)
    write_simple_audit(combined, paths.audit_dir / "08_union_sources_audit.csv")
    return combined


def merge_nelda(
    sources: pd.DataFrame,
    nelda: pd.DataFrame,
    audit_dir: Path | None = None,
    ledger: MergeLedger | None = None,
    step_id: str = "09_merge_nelda",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nelda = nelda.copy()
    nelda["nelda_date"] = parse_nelda_date(nelda["year"], nelda["mmdd"])
    nelda["office_type"] = nelda["types"].map({
        "Executive": "presidential",
        "Legislative/Parliamentary": "parliamentary",
        "Constituent Assembly": None,
    })

    # prepare candidates by cow code + year + office_type
    candidates = nelda.dropna(subset=["ccode", "year"]).copy()
    candidates["ccode"] = pd.to_numeric(candidates["ccode"], errors="coerce")

    src = sources.copy()
    if "country_cow" in src.columns:
        src["cow_code"] = pd.to_numeric(src.get("country_cow"), errors="coerce")
    elif "cow_code" in src.columns:
        src["cow_code"] = pd.to_numeric(src.get("cow_code"), errors="coerce")
    if "nelda_ccode_from_nelda" in src.columns:
        src["nelda_ccode_from_nelda"] = pd.to_numeric(src.get("nelda_ccode_from_nelda"), errors="coerce")

    # choose code source: prefer nelda_ccode_from_nelda if available
    src["nelda_ccode_used"] = src.get("nelda_ccode_from_nelda")
    src["nelda_ccode_used_source"] = pd.NA
    src.loc[src["nelda_ccode_used"].notna(), "nelda_ccode_used_source"] = "nelda_modal"
    src.loc[src["nelda_ccode_used"].isna(), "nelda_ccode_used"] = src.get("cow_code")
    src.loc[src["nelda_ccode_used_source"].isna() & src.get("cow_code").notna(), "nelda_ccode_used_source"] = "cow_code"

    if ledger is None:
        merged = src.merge(
            candidates,
            left_on=["nelda_ccode_used", "election_year", "office_type"],
            right_on=["ccode", "year", "office_type"],
            how="left",
            suffixes=("", "_nelda"),
            indicator=True,
        )
    else:
        merged = logged_merge(
            ledger,
            src,
            candidates,
            how="left",
            left_on=["nelda_ccode_used", "election_year", "office_type"],
            right_on=["ccode", "year", "office_type"],
            step_id=step_id,
            step_name="Merge NELDA attributes",
            left_table="election_event_sources",
            right_table="nelda",
            context_cols_left=["record_id", "iso3", "election_year", "office_type"],
            context_cols_right=["ccode", "year", "office_type", "electionid"],
            keep_merge_indicator=True,
        )

    # compute date diff if both dates known
    merged["date_diff"] = (merged["election_date"] - merged["nelda_date"]).abs().dt.days
    merged["date_match_rule"] = "none"
    month_match = (merged["date_precision"] == "month") & merged["nelda_date"].notna()
    month_match &= merged["election_year"].notna() & merged["election_month"].notna()
    month_match &= (merged["election_year"] == merged["nelda_date"].dt.year) & (merged["election_month"] == merged["nelda_date"].dt.month)
    merged.loc[month_match, "date_diff"] = 0
    merged.loc[month_match, "date_match_rule"] = "month_exact"

    # select best match per record_id
    def pick_best(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # prefer month_exact then date_diff
        df = df.copy()
        df["date_rank"] = 2
        df.loc[df["date_match_rule"] == "month_exact", "date_rank"] = 0
        df.loc[(df["date_match_rule"] != "month_exact") & df["date_diff"].notna(), "date_rank"] = 1
        df["date_ok"] = df["date_diff"].where(df["date_diff"].notna(), np.inf)
        df = df.sort_values(["date_rank", "date_ok", "date_diff"], ascending=[True, True, True], kind="mergesort")
        best = df.iloc[0:1]
        # check ambiguity: if multiple with same date_ok and date_diff
        if len(df) > 1:
            second = df.iloc[1]
            if best["date_rank"].iloc[0] == second["date_rank"] and best["date_ok"].iloc[0] == second["date_ok"] and best["date_diff"].iloc[0] == second["date_diff"]:
                best["nelda_match_status"] = "ambiguous"
                return best
        if best["_merge"].iloc[0] == "left_only":
            best["nelda_match_status"] = "unmatched"
            return best
        # tolerance flag
        if pd.notna(best["date_diff"].iloc[0]) and best["date_diff"].iloc[0] > 60:
            best["nelda_match_status"] = "matched_outside_tolerance"
        else:
            best["nelda_match_status"] = "matched"
        return best

    picked = merged.groupby("record_id", group_keys=False).apply(pick_best)

    # retain only selected columns
    keep_cols = list(src.columns) + [
        "electionid",
        "nelda_date",
        "nelda3",
        "nelda4",
        "nelda5",
        "nelda_match_status",
        "date_diff",
        "date_match_rule",
    ]
    out = picked[keep_cols].copy()
    out = out.rename(columns={"electionid": "nelda_electionid"})

    # fallback to cow_code if modal ccode yields no match
    fallback_mask = out["nelda_match_status"].eq("unmatched") & out.get("cow_code").notna()
    fallback_mask &= out["nelda_ccode_used"].isna() | (out["nelda_ccode_used"] != out.get("cow_code"))
    if fallback_mask.any():
        fallback_src = out.loc[fallback_mask, src.columns].copy()
        fallback_src["nelda_ccode_used"] = fallback_src.get("cow_code")
        fallback_src["nelda_ccode_used_source"] = "cow_fallback"
        if ledger is None:
            fallback_merged = fallback_src.merge(
                candidates,
                left_on=["nelda_ccode_used", "election_year", "office_type"],
                right_on=["ccode", "year", "office_type"],
                how="left",
                suffixes=("", "_nelda"),
                indicator=True,
            )
        else:
            fallback_merged = logged_merge(
                ledger,
                fallback_src,
                candidates,
                how="left",
                left_on=["nelda_ccode_used", "election_year", "office_type"],
                right_on=["ccode", "year", "office_type"],
                step_id=f"{step_id}_cow_fallback",
                step_name="NELDA fallback via cow_code",
                left_table="election_event_sources",
                right_table="nelda",
                context_cols_left=["record_id", "iso3", "election_year", "office_type"],
                context_cols_right=["ccode", "year", "office_type", "electionid"],
                keep_merge_indicator=True,
            )
        fallback_merged["date_diff"] = (fallback_merged["election_date"] - fallback_merged["nelda_date"]).abs().dt.days
        fallback_merged["date_match_rule"] = "none"
        month_match = (fallback_merged["date_precision"] == "month") & fallback_merged["nelda_date"].notna()
        month_match &= fallback_merged["election_year"].notna() & fallback_merged["election_month"].notna()
        month_match &= (fallback_merged["election_year"] == fallback_merged["nelda_date"].dt.year) & (fallback_merged["election_month"] == fallback_merged["nelda_date"].dt.month)
        fallback_merged.loc[month_match, "date_diff"] = 0
        fallback_merged.loc[month_match, "date_match_rule"] = "month_exact"
        fallback_picked = fallback_merged.groupby("record_id", group_keys=False).apply(pick_best)
        fallback_picked = fallback_picked.rename(columns={"electionid": "nelda_electionid"})
        nelda_cols = [
            "nelda_electionid",
            "nelda_date",
            "nelda3",
            "nelda4",
            "nelda5",
            "nelda_match_status",
            "date_diff",
            "date_match_rule",
        ]
        fallback_update = fallback_picked.set_index("record_id")
        fallback_update = fallback_update[fallback_update["nelda_match_status"] != "unmatched"]
        if not fallback_update.empty:
            out = out.set_index("record_id")
            out.update(fallback_update[nelda_cols])
            out.loc[fallback_update.index, "nelda_ccode_used"] = out.loc[fallback_update.index, "cow_code"]
            out.loc[fallback_update.index, "nelda_ccode_used_source"] = "cow_fallback"
            out = out.reset_index()

    match_report = out[["record_id", "nelda_electionid", "nelda_match_status", "date_diff", "date_match_rule"]].copy()
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        out[out["nelda_match_status"] == "unmatched"].to_csv(audit_dir / "09_nelda_match_unmatched.csv", index=False)
        out[out["nelda_match_status"] == "ambiguous"].to_csv(audit_dir / "09_nelda_match_ambiguous.csv", index=False)
    return out, match_report
