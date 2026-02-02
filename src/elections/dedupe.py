from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .standardize import normalize_name


def compute_merge_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    key_day = df["election_date"].dt.strftime("%Y-%m-%d")
    key_month = df["election_date"].dt.strftime("%Y-%m")
    key_year = df["election_year"].astype("Int64").astype(str)

    df["merge_strategy"] = "year"
    df.loc[df["date_precision"] == "month", "merge_strategy"] = "month"
    df.loc[df["date_precision"] == "day", "merge_strategy"] = "day"

    df["merge_key"] = (
        df["iso3"].astype(str) + "|" + df["office_type"].astype(str) + "|" +
        df.apply(lambda r: key_day[r.name] if r["merge_strategy"] == "day" else (key_month[r.name] if r["merge_strategy"] == "month" else key_year[r.name]), axis=1)
    )
    # avoid grouping when iso3 missing
    df.loc[df["iso3"].isna(), "merge_key"] = df.loc[df["iso3"].isna(), "record_id"]
    return df


def score_record(df: pd.DataFrame) -> pd.Series:
    fields = ["share_1", "share_2", "party_1_id", "party_2_id", "election_date", "nelda3", "nelda4", "nelda5"]
    score = pd.Series(0, index=df.index, dtype=int)
    for col in fields:
        if col in df.columns:
            score += df[col].notna().astype(int)
    return score


def _party_id_set(row: pd.Series) -> set:
    vals = {row.get("party_1_id"), row.get("party_2_id")}
    return {v for v in vals if pd.notna(v)}


def _party_name_set(row: pd.Series) -> set:
    vals = []
    for col in ["party_1_name_raw", "party_2_name_raw"]:
        val = row.get(col)
        norm = normalize_name(val)
        if norm:
            vals.append(norm)
    return set(vals)


def _safe_day_diff(a: pd.Timestamp | pd.NaT, b: pd.Timestamp | pd.NaT) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.inf
    return abs((a - b).days)


def dedupe_events(
    df: pd.DataFrame,
    source_rank: list[str],
    margin_diff_max: float = 5.0,
    audit_dir: str | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = compute_merge_key(df)
    df["merge_score"] = score_record(df)

    # rank sources
    rank_map = {s: i for i, s in enumerate(source_rank)}
    df["source_rank"] = df["source"].map(rank_map).fillna(len(rank_map)).astype(int)

    # legacy seq + election_id for audit
    df = df.sort_values(["merge_key", "source_rank", "merge_score", "record_id"], ascending=[True, True, False, True], kind="mergesort")
    df["old_seq"] = df.groupby("merge_key").cumcount() + 1
    def make_old_election_id(row):
        if pd.isna(row.get("iso3")):
            return f"UNMAPPED_{row.get('record_id')}"
        date_key = row.get("merge_key").split("|")[-1]
        office = row.get("office_type")
        return f"{row['iso3']}_{office}_{date_key}_s{int(row['old_seq']):02d}"
    df["election_id_old"] = df.apply(make_old_election_id, axis=1)

    # new group key: month-level
    def group_key(row: pd.Series) -> tuple[str, str]:
        if pd.isna(row.get("iso3")) or pd.isna(row.get("office_type")) or pd.isna(row.get("election_year")):
            return f"UNMAPPED|{row.get('record_id')}", "unmapped"
        year = int(row.get("election_year"))
        month = row.get("election_month")
        if pd.notna(month):
            return f"{row['iso3']}|{row['office_type']}|{year}-{int(month):02d}", "month"
        return f"{row['iso3']}|{row['office_type']}|{year}", "year"

    keys = df.apply(group_key, axis=1, result_type="expand")
    df["event_group_key"] = keys[0]
    df["event_group_precision"] = keys[1]

    df["event_seq"] = pd.NA
    align_pairs = []
    unmatched_clea = []
    multi_primary = []

    # group alignment
    for key, group in df.groupby("event_group_key", sort=False):
        if key.startswith("UNMAPPED|"):
            df.loc[group.index, "event_seq"] = 1
            continue

        primary = group[group["source"] != "clea_lc"].copy()
        clea = group[group["source"] == "clea_lc"].copy()

        if len(primary) > 1:
            sample = primary.iloc[0]
            multi_primary.append({
                "event_group_key": key,
                "iso3": sample.get("iso3"),
                "office_type": sample.get("office_type"),
                "election_year": sample.get("election_year"),
                "election_month": sample.get("election_month"),
                "n_primary": len(primary),
            })

        # if no primary, assign seq to all by date/order
        if primary.empty:
            group_sorted = group.sort_values(["election_date", "record_id"], kind="mergesort")
            df.loc[group_sorted.index, "event_seq"] = range(1, len(group_sorted) + 1)
            continue

        primary_sorted = primary.sort_values(["election_date", "record_id"], kind="mergesort")
        primary_sorted = primary_sorted.assign(event_seq=range(1, len(primary_sorted) + 1))
        df.loc[primary_sorted.index, "event_seq"] = primary_sorted["event_seq"].values

        used_seq: set[int] = set()
        for idx, clea_row in clea.sort_values(["merge_score", "record_id"], ascending=[False, True], kind="mergesort").iterrows():
            candidates = []
            for _, ned_row in primary_sorted.iterrows():
                party_clea = _party_id_set(clea_row)
                party_ned = _party_id_set(ned_row)
                party_overlap = len(party_clea & party_ned) if party_clea and party_ned else 0
                party_equal = int(bool(party_clea) and party_clea == party_ned)
                party_score = 2 if party_equal else (1 if party_overlap > 0 else 0)

                name_clea = _party_name_set(clea_row)
                name_ned = _party_name_set(ned_row)
                name_overlap = len(name_clea & name_ned) if name_clea and name_ned else 0
                name_equal = int(bool(name_clea) and name_clea == name_ned)
                name_score = 2 if name_equal else (1 if name_overlap > 0 else 0)

                margin_diff = np.inf
                if pd.notna(clea_row.get("margin")) and pd.notna(ned_row.get("margin")):
                    margin_diff = abs(clea_row.get("margin") - ned_row.get("margin"))

                date_diff = _safe_day_diff(clea_row.get("election_date"), ned_row.get("election_date"))

                allowed = False
                reason = "no_match_signals"
                method = None
                if party_clea and party_ned:
                    if party_overlap > 0:
                        allowed = True
                        method = "party_id"
                    else:
                        reason = "party_id_mismatch"
                elif name_clea and name_ned:
                    if name_overlap > 0:
                        allowed = True
                        method = "party_name"
                    else:
                        reason = "name_mismatch"
                elif np.isfinite(margin_diff):
                    if margin_diff <= margin_diff_max:
                        allowed = True
                        method = "margin"
                    else:
                        reason = "margin_diff"

                if not allowed:
                    continue

                candidates.append({
                    "event_seq": int(ned_row["event_seq"]),
                    "party_score": party_score,
                    "party_overlap": party_overlap,
                    "name_score": name_score,
                    "name_overlap": name_overlap,
                    "margin_diff": margin_diff,
                    "date_diff": date_diff,
                    "record_id_primary": ned_row.get("record_id"),
                    "match_method": method,
                })

            # pick best candidate not already used
            chosen = None
            if candidates:
                cand_df = pd.DataFrame(candidates)
                cand_df = cand_df.sort_values(
                    ["party_score", "name_score", "margin_diff", "date_diff", "record_id_primary"],
                    ascending=[False, False, True, True, True],
                    kind="mergesort",
                )
                for _, cand in cand_df.iterrows():
                    if cand["event_seq"] not in used_seq:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = cand_df.iloc[0]

            if chosen is not None:
                seq_val = int(chosen["event_seq"])
                df.at[idx, "event_seq"] = seq_val
                used_seq.add(seq_val)
                align_pairs.append({
                    "event_group_key": key,
                    "record_id_left": chosen.get("record_id_primary"),
                    "record_id_right": clea_row.get("record_id"),
                    "source_left": "primary",
                    "source_right": "clea_lc",
                    "party_overlap": int(chosen.get("party_overlap", 0)),
                    "name_overlap": int(chosen.get("name_overlap", 0)),
                    "margin_diff": float(chosen.get("margin_diff", np.nan)) if np.isfinite(chosen.get("margin_diff", np.inf)) else np.nan,
                    "date_diff": float(chosen.get("date_diff", np.nan)) if np.isfinite(chosen.get("date_diff", np.inf)) else np.nan,
                    "decision": "matched",
                    "match_method": chosen.get("match_method"),
                })
            else:
                max_seq = int(primary_sorted["event_seq"].max())
                if used_seq:
                    max_seq = max(max_seq, max(used_seq))
                next_seq = max_seq + 1
                df.at[idx, "event_seq"] = next_seq
                unmatched_clea.append({
                    "event_group_key": key,
                    "record_id": clea_row.get("record_id"),
                    "iso3": clea_row.get("iso3"),
                    "office_type": clea_row.get("office_type"),
                    "election_year": clea_row.get("election_year"),
                    "election_month": clea_row.get("election_month"),
                    "reason": "no_match",
                })

    # finalize election_id
    def make_election_id(row):
        if pd.isna(row.get("iso3")) or row.get("event_group_key", "").startswith("UNMAPPED|"):
            return f"UNMAPPED_{row.get('record_id')}"
        date_key = row.get("event_group_key").split("|")[-1]
        office = row.get("office_type")
        return f"{row['iso3']}_{office}_{date_key}_e{int(row['event_seq']):02d}"

    df["election_id"] = df.apply(make_election_id, axis=1)

    # assign source seq within election_id
    df = df.sort_values(["election_id", "source_rank", "merge_score", "record_id"], ascending=[True, True, False, True], kind="mergesort")
    df["source_seq"] = df.groupby("election_id").cumcount() + 1
    df["merge_preferred"] = False
    df["n_source_records_in_event"] = df.groupby("election_id")["record_id"].transform("count")
    preferred_idx = df.groupby("election_id").head(1).index
    df.loc[preferred_idx, "merge_preferred"] = True

    # audit outputs
    if audit_dir is not None:
        out_dir = Path(audit_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if align_pairs:
            pd.DataFrame(align_pairs).to_csv(out_dir / "08_event_alignment_pairs.csv", index=False)
        if unmatched_clea:
            pd.DataFrame(unmatched_clea).to_csv(out_dir / "08_event_alignment_unmatched_clea.csv", index=False)
        if multi_primary:
            pd.DataFrame(multi_primary).to_csv(out_dir / "08_event_alignment_multi_ned_groups.csv", index=False)

    # canonical table
    canonical = df[df["merge_preferred"]].copy()

    return df, canonical


def attach_primary_secondary_shares(canonical: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    # For each election_id, prefer CLEA vote shares if parties align; otherwise keep canonical
    out = canonical.copy()
    out["share_metric_primary"] = out["share_metric"]
    out["share_metric_secondary"] = pd.NA
    out["share_1_secondary"] = pd.NA
    out["share_2_secondary"] = pd.NA
    out["margin_secondary"] = pd.NA
    out["share_primary_source"] = out.get("source")
    out["share_secondary_source"] = pd.NA
    out["share_alignment_status"] = "no_clea"

    clea = sources[sources["source"] == "clea_lc"].copy()
    if clea.empty:
        return out

    clea_best = clea.sort_values(["merge_score"], ascending=[False], kind="mergesort").groupby("election_id").head(1)
    clea_lookup = clea_best.set_index("election_id")

    for idx, row in out.iterrows():
        eid = row.get("election_id")
        if pd.isna(eid) or eid not in clea_lookup.index:
            continue
        clea_row = clea_lookup.loc[eid]

        # check party alignment (set equality for top two)
        parties_canon = {row.get("party_1_id"), row.get("party_2_id")}
        parties_clea = {clea_row.get("party_1_id"), clea_row.get("party_2_id")}
        parties_canon = {p for p in parties_canon if pd.notna(p)}
        parties_clea = {p for p in parties_clea if pd.notna(p)}

        aligned = parties_canon and parties_clea and parties_canon == parties_clea

        if aligned:
            out.at[idx, "share_metric_primary"] = clea_row.get("share_metric")
            out.at[idx, "share_metric_secondary"] = row.get("share_metric")
            out.at[idx, "share_1_secondary"] = row.get("share_1")
            out.at[idx, "share_2_secondary"] = row.get("share_2")
            out.at[idx, "margin_secondary"] = row.get("margin")

            out.at[idx, "share_1"] = clea_row.get("share_1")
            out.at[idx, "share_2"] = clea_row.get("share_2")
            out.at[idx, "margin"] = clea_row.get("margin")
            out.at[idx, "share_primary_source"] = "clea_lc"
            out.at[idx, "share_secondary_source"] = row.get("source")
            out.at[idx, "share_alignment_status"] = "party_id_aligned"
        else:
            # keep canonical primary; attach CLEA as secondary if present
            out.at[idx, "share_metric_secondary"] = clea_row.get("share_metric")
            out.at[idx, "share_1_secondary"] = clea_row.get("share_1")
            out.at[idx, "share_2_secondary"] = clea_row.get("share_2")
            out.at[idx, "margin_secondary"] = clea_row.get("margin")
            out.at[idx, "share_secondary_source"] = "clea_lc"
            out.at[idx, "share_alignment_status"] = "not_aligned"

    return out
