from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd

from .standardize import normalize_name

try:  # optional
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


@dataclass
class MatchResult:
    partyfacts_id: int | None
    score: int
    method: str
    status: str


def _fuzzy_best(query: str, choices: list[str]) -> Tuple[str | None, int, str | None, int]:
    if not query:
        return None, 0, None, 0
    if not choices:
        return None, 0, None, 0
    if process and fuzz:
        results = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=2)
        if not results:
            return None, 0, None, 0
        best, score1, _ = results[0]
        if len(results) > 1:
            second, score2, _ = results[1]
        else:
            second, score2 = None, 0
        return best, int(score1), second, int(score2)
    import difflib
    scores = [(c, int(difflib.SequenceMatcher(None, query, c).ratio() * 100)) for c in choices]
    scores.sort(key=lambda x: (-x[1], x[0]))
    best, score1 = scores[0]
    if len(scores) > 1:
        second, score2 = scores[1]
    else:
        second, score2 = None, 0
    return best, int(score1), second, int(score2)


def build_party_aliases(partyfacts: pd.DataFrame, partyfacts_external: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for _, row in partyfacts.iterrows():
        pid = row.get("partyfacts_id")
        iso3 = row.get("country")
        year_first = row.get("year_first")
        year_last = row.get("year_last")
        technical = row.get("technical")
        for col in ["name", "name_english", "name_short", "name_other"]:
            val = row.get(col)
            if pd.isna(val) or val is None:
                continue
            norm = normalize_name(val)
            if not norm:
                continue
            rows.append({
                "partyfacts_id": pid,
                "iso3": iso3,
                "year_first": year_first,
                "year_last": year_last,
                "technical": technical,
                "alias": norm,
                "alias_source": f"core:{col}",
            })

    if partyfacts_external is not None:
        ext = partyfacts_external.copy()
        # keep only linked rows with partyfacts_id and country
        if "partyfacts_id" in ext.columns:
            ext = ext.dropna(subset=["partyfacts_id"])
        # bring in core technical + fallback years
        core_cols = partyfacts[["partyfacts_id", "technical", "year_first", "year_last", "country"]].copy()
        core_cols = core_cols.rename(columns={"country": "country_core"})
        ext = ext.merge(core_cols, on="partyfacts_id", how="left", suffixes=("", "_core"))
        if "country" in ext.columns and "country_core" in ext.columns:
            ext["country"] = ext["country"].combine_first(ext["country_core"])
        ext = ext.dropna(subset=["country"])
        for _, row in ext.iterrows():
            pid = row.get("partyfacts_id")
            iso3 = row.get("country")
            year_first = row.get("year_first")
            year_last = row.get("year_last")
            if pd.isna(year_first):
                year_first = row.get("year_first_core")
            if pd.isna(year_last):
                year_last = row.get("year_last_core")
            technical = row.get("technical")
            dataset_key = row.get("dataset_key")
            for col in ["name", "name_english", "name_short"]:
                val = row.get(col)
                if pd.isna(val) or val is None:
                    continue
                norm = normalize_name(val)
                if not norm:
                    continue
                rows.append({
                    "partyfacts_id": pid,
                    "iso3": iso3,
                    "year_first": year_first,
                    "year_last": year_last,
                    "technical": technical,
                    "alias": norm,
                    "alias_source": f"external:{dataset_key}:{col}",
                })

    aliases = pd.DataFrame(rows)
    if aliases.empty:
        return aliases
    aliases = aliases.dropna(subset=["partyfacts_id", "iso3", "alias"])
    aliases = aliases.sort_values(
        ["iso3", "alias", "partyfacts_id", "alias_source"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).drop_duplicates()
    return aliases


def match_party_name(
    name: str | None,
    iso3: str | None,
    election_year: int | None,
    aliases: pd.DataFrame,
    min_score: int,
    tie_delta: float,
) -> MatchResult:
    if name is None or iso3 is None or pd.isna(name) or pd.isna(iso3):
        return MatchResult(None, 0, "missing", "unmatched")
    norm = normalize_name(name)
    if not norm:
        return MatchResult(None, 0, "missing", "unmatched")

    pool = aliases[aliases["iso3"] == iso3]
    if election_year is not None and not pd.isna(election_year):
        pool = pool[(pool["year_first"].isna() | (pool["year_first"] <= election_year)) & (pool["year_last"].isna() | (pool["year_last"] >= election_year))]

    if pool.empty:
        return MatchResult(None, 0, "no_candidates", "unmatched")

    choices = pool["alias"].tolist()
    best, score1, second, score2 = _fuzzy_best(norm, choices)
    if best is None:
        return MatchResult(None, 0, "fuzzy", "unmatched")

    if score1 >= min_score and (score1 - score2) <= tie_delta:
        # if top two aliases map to the same partyfacts_id, accept match
        if second is not None:
            cand_best = pool.loc[pool["alias"] == best, "partyfacts_id"].dropna().unique()
            cand_second = pool.loc[pool["alias"] == second, "partyfacts_id"].dropna().unique()
            if len(cand_best) == 1 and len(cand_second) == 1 and cand_best[0] == cand_second[0]:
                return MatchResult(int(cand_best[0]), score1, "fuzzy_tie_same_party", "matched")
        return MatchResult(None, score1, "fuzzy", "ambiguous")
    if score1 >= min_score:
        cand = pool.loc[pool["alias"] == best].copy()
        candidates = cand["partyfacts_id"].dropna().unique()
        if len(candidates) == 1:
            pid = candidates[0]
            return MatchResult(int(pid), score1, "fuzzy", "matched")

        # tiebreak among multiple partyfacts_id
        if cand.empty:
            return MatchResult(None, score1, "fuzzy", "unmatched")

        cand["technical_rank"] = cand.get("technical").map({False: 0, True: 1}).fillna(2)
        cand["source_rank"] = cand.get("alias_source").map(lambda x: 0 if str(x).startswith("core:") else 1)
        span = pd.to_numeric(cand.get("year_last"), errors="coerce") - pd.to_numeric(cand.get("year_first"), errors="coerce")
        cand["year_span"] = span.abs().fillna(float("inf"))

        sort_cols = ["technical_rank", "source_rank", "year_span", "partyfacts_id"]
        cand = cand.sort_values(sort_cols, ascending=[True, True, True, True], kind="mergesort")
        if len(cand) > 1:
            top = cand.iloc[0]
            second = cand.iloc[1]
            # if top and second tie on all criteria except partyfacts_id, keep ambiguous
            if all(top[c] == second[c] for c in ["technical_rank", "source_rank", "year_span"]):
                return MatchResult(None, score1, "fuzzy", "ambiguous")
        pid = cand["partyfacts_id"].iloc[0]
        return MatchResult(int(pid), score1, "fuzzy_tiebreak", "matched_tiebreak")
    return MatchResult(None, score1, "fuzzy", "unmatched")


def build_party_match_results(
    elections: pd.DataFrame,
    partyfacts: pd.DataFrame,
    min_score: int,
    tie_delta: float,
    out_dir: str | None = None,
    partyfacts_external: pd.DataFrame | None = None,
) -> pd.DataFrame:
    aliases = build_party_aliases(partyfacts, partyfacts_external=partyfacts_external)
    if out_dir and not aliases.empty:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        alias_counts = aliases.groupby(["iso3", "alias"])["partyfacts_id"].nunique().reset_index(name="partyfacts_id_n")
        alias_counts.loc[alias_counts["partyfacts_id_n"] > 1].to_csv(out / "10_party_alias_ambiguous.csv", index=False)

    # unique party strings
    party_names = pd.concat([
        elections[["iso3", "election_year", "party_1_name_raw"]].rename(columns={"party_1_name_raw": "party_name"}),
        elections[["iso3", "election_year", "party_2_name_raw"]].rename(columns={"party_2_name_raw": "party_name"}),
    ], ignore_index=True).drop_duplicates()

    results = []
    for _, row in party_names.iterrows():
        res = match_party_name(row["party_name"], row["iso3"], row["election_year"], aliases, min_score, tie_delta)
        results.append({
            "iso3": row["iso3"],
            "election_year": row["election_year"],
            "party_name_raw": row["party_name"],
            "party_string_key": f"{row['iso3']}|{row['election_year']}|{row['party_name']}",
            "partyfacts_id": res.partyfacts_id,
            "match_score": res.score,
            "match_method": res.method,
            "match_status": res.status,
        })

    res_df = pd.DataFrame(results)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res_df.to_parquet(out / "party_match_results.parquet", index=False)
        res_df[res_df["match_status"] == "unmatched"].to_csv(out / "10_party_match_unmatched.csv", index=False)
        res_df[res_df["match_status"] == "ambiguous"].to_csv(out / "10_party_match_ambiguous.csv", index=False)
        res_df[res_df["match_status"] == "matched_tiebreak"].to_csv(out / "10_party_match_tiebreak.csv", index=False)
    return res_df
