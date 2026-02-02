from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import re
import pandas as pd

from .standardize import normalize_name
from .merge_ledger import MergeLedger, logged_match

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
    alias: str | None = None
    alias_source: str | None = None
    candidates: list[int] | None = None


_PAREN_RE = re.compile(r"\(([^)]+)\)")
_STOPWORDS = {
    "alliance",
    "alianza",
    "coalition",
    "coalicion",
    "front",
    "bloc",
    "union",
    "movement",
    "league",
    "federation",
    "group",
    "platform",
    "party",
    "parties",
}


def generate_variants(name: str | None) -> dict[str, str]:
    if name is None or pd.isna(name):
        return {}
    raw = str(name)
    variants: dict[str, str] = {}
    norm = normalize_name(raw)
    if norm:
        variants["norm"] = norm

    noparen = normalize_name(_PAREN_RE.sub(" ", raw))
    if noparen and noparen != norm:
        variants["noparen"] = noparen

    paren_parts = _PAREN_RE.findall(raw)
    if paren_parts:
        paren_norm = normalize_name(" ".join(paren_parts))
        if paren_norm:
            variants["paren_only"] = paren_norm

    if norm:
        tokens = [t for t in norm.split() if t not in _STOPWORDS]
        stop_norm = " ".join(tokens)
        if stop_norm and stop_norm != norm:
            variants["stopwords_removed"] = stop_norm

    if norm and " and " in f" {norm} ":
        left, _, right = norm.partition(" and ")
        if left:
            variants["split_left"] = left.strip()
        if right:
            variants["split_right"] = right.strip()

    return variants


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


def build_party_aliases(
    partyfacts: pd.DataFrame,
    partyfacts_external: pd.DataFrame | None = None,
    vparty: pd.DataFrame | None = None,
    country_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    if vparty is not None and country_map is not None:
        vparty_map = vparty.copy()
        if "pf_party_id" in vparty_map.columns:
            vparty_map = vparty_map.dropna(subset=["pf_party_id"])
        if "country_id" in vparty_map.columns and "vdem_country_id" in country_map.columns:
            vparty_map = vparty_map.merge(
                country_map[["iso3", "vdem_country_id"]],
                left_on="country_id",
                right_on="vdem_country_id",
                how="left",
            )
        vparty_map = vparty_map.dropna(subset=["iso3"])
        for _, row in vparty_map.iterrows():
            pid = row.get("pf_party_id")
            iso3 = row.get("iso3")
            year_first = row.get("year")
            year_last = row.get("year")
            for col in ["v2paenname", "v2pashname", "v2paorname"]:
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
                    "technical": pd.NA,
                    "alias": norm,
                    "alias_source": f"vparty:{col}",
                })

    if rows:
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
    year_window: int | None = None,
) -> MatchResult:
    if name is None or iso3 is None or pd.isna(name) or pd.isna(iso3):
        return MatchResult(None, 0, "missing", "unmatched")
    norm = normalize_name(name)
    if not norm:
        return MatchResult(None, 0, "missing", "unmatched")

    pool = aliases[aliases["iso3"] == iso3]
    if election_year is not None and not pd.isna(election_year):
        if year_window is None:
            pool = pool[(pool["year_first"].isna() | (pool["year_first"] <= election_year)) & (pool["year_last"].isna() | (pool["year_last"] >= election_year))]
        else:
            pool = pool[(pool["year_first"].isna() | (pool["year_first"] <= election_year + year_window)) & (pool["year_last"].isna() | (pool["year_last"] >= election_year - year_window))]

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
                alias_source = pool.loc[pool["alias"] == best, "alias_source"].iloc[0] if "alias_source" in pool.columns else None
                return MatchResult(int(cand_best[0]), score1, "fuzzy_tie_same_party", "matched", alias=best, alias_source=alias_source, candidates=[int(cand_best[0])])
            cand_ids = sorted({int(x) for x in list(cand_best) + list(cand_second) if pd.notna(x)})
            return MatchResult(None, score1, "fuzzy", "ambiguous", alias=best, alias_source=None, candidates=cand_ids)
        return MatchResult(None, score1, "fuzzy", "ambiguous", alias=best, alias_source=None, candidates=None)
    if score1 >= min_score:
        cand = pool.loc[pool["alias"] == best].copy()
        candidates = cand["partyfacts_id"].dropna().unique()
        if len(candidates) == 1:
            pid = candidates[0]
            alias_source = cand.loc[cand["partyfacts_id"] == pid, "alias_source"].iloc[0] if "alias_source" in cand.columns and not cand.empty else None
            return MatchResult(int(pid), score1, "fuzzy", "matched", alias=best, alias_source=alias_source, candidates=[int(pid)])

        # tiebreak among multiple partyfacts_id
        if cand.empty:
            return MatchResult(None, score1, "fuzzy", "unmatched", alias=best, alias_source=None, candidates=None)

        cand["technical_rank"] = cand.get("technical").map({False: 0, True: 1}).fillna(2)
        cand["source_rank"] = cand.get("alias_source").map(lambda x: 0 if str(x).startswith("core:") else 1)
        span = pd.to_numeric(cand.get("year_last"), errors="coerce") - pd.to_numeric(cand.get("year_first"), errors="coerce")
        cand["year_span"] = span.abs().fillna(float("inf"))
        if election_year is not None and not pd.isna(election_year):
            mid = (pd.to_numeric(cand.get("year_first"), errors="coerce") + pd.to_numeric(cand.get("year_last"), errors="coerce")) / 2.0
            cand["year_mid_gap"] = (mid - float(election_year)).abs()
        else:
            cand["year_mid_gap"] = float("inf")

        sort_cols = ["technical_rank", "source_rank", "year_mid_gap", "year_span", "partyfacts_id"]
        cand = cand.sort_values(sort_cols, ascending=[True, True, True, True, True], kind="mergesort")
        if len(cand) > 1:
            top = cand.iloc[0]
            second_row = cand.iloc[1]
            # if top and second tie on all criteria except partyfacts_id, keep ambiguous
            if all(top[c] == second_row[c] for c in ["technical_rank", "source_rank", "year_mid_gap", "year_span"]):
                cand_ids = sorted({int(x) for x in candidates if pd.notna(x)})
                return MatchResult(None, score1, "fuzzy", "ambiguous", alias=best, alias_source=None, candidates=cand_ids)
        pid = cand["partyfacts_id"].iloc[0]
        alias_source = cand.loc[cand["partyfacts_id"] == pid, "alias_source"].iloc[0] if "alias_source" in cand.columns and not cand.empty else None
        return MatchResult(int(pid), score1, "fuzzy_tiebreak", "matched_tiebreak", alias=best, alias_source=alias_source, candidates=[int(pid)])
    return MatchResult(None, score1, "fuzzy", "unmatched", alias=best, alias_source=None, candidates=None)


def _filter_pool(aliases: pd.DataFrame, iso3: str, election_year: int | None, year_window: int | None) -> pd.DataFrame:
    pool = aliases[aliases["iso3"] == iso3]
    if election_year is None or pd.isna(election_year):
        return pool
    if year_window is None:
        return pool[(pool["year_first"].isna() | (pool["year_first"] <= election_year)) & (pool["year_last"].isna() | (pool["year_last"] >= election_year))]
    return pool[(pool["year_first"].isna() | (pool["year_first"] <= election_year + year_window)) & (pool["year_last"].isna() | (pool["year_last"] >= election_year - year_window))]


def _suggest_candidates(name_norm: str, pool: pd.DataFrame, top_n: int = 5) -> list[dict]:
    if not name_norm or pool.empty:
        return []
    choices = pool["alias"].tolist()
    results = []
    if process and fuzz:
        results = process.extract(name_norm, choices, scorer=fuzz.token_sort_ratio, limit=max(10, top_n * 2))
    else:
        import difflib
        scores = [(c, int(difflib.SequenceMatcher(None, name_norm, c).ratio() * 100)) for c in choices]
        scores.sort(key=lambda x: (-x[1], x[0]))
        results = [(c, s, None) for c, s in scores[: max(10, top_n * 2)]]

    cand_map: dict[int, dict] = {}
    for alias, score, *_ in results:
        subset = pool[pool["alias"] == alias]
        for _, row in subset.iterrows():
            pid = row.get("partyfacts_id")
            if pd.isna(pid):
                continue
            pid_int = int(pid)
            entry = cand_map.get(pid_int)
            if entry is None or score > entry["score"]:
                cand_map[pid_int] = {
                    "partyfacts_id": pid_int,
                    "alias": alias,
                    "score": int(score),
                    "alias_source": row.get("alias_source"),
                }
    return sorted(cand_map.values(), key=lambda x: (-x["score"], x["partyfacts_id"]))[:top_n]


def _pick_best_candidate(candidates: list[int], aliases: pd.DataFrame, election_year: int | None) -> int | None:
    if not candidates:
        return None
    cand = aliases[aliases["partyfacts_id"].isin(candidates)].copy()
    if cand.empty:
        return None
    cand["technical_rank"] = cand.get("technical").map({False: 0, True: 1}).fillna(2)
    cand["source_rank"] = cand.get("alias_source").map(lambda x: 0 if str(x).startswith("core:") else 1)
    span = pd.to_numeric(cand.get("year_last"), errors="coerce") - pd.to_numeric(cand.get("year_first"), errors="coerce")
    cand["year_span"] = span.abs().fillna(float("inf"))
    if election_year is not None and not pd.isna(election_year):
        mid = (pd.to_numeric(cand.get("year_first"), errors="coerce") + pd.to_numeric(cand.get("year_last"), errors="coerce")) / 2.0
        cand["year_mid_gap"] = (mid - float(election_year)).abs()
    else:
        cand["year_mid_gap"] = float("inf")
    grouped = cand.groupby("partyfacts_id", as_index=False).agg({
        "technical_rank": "min",
        "source_rank": "min",
        "year_mid_gap": "min",
        "year_span": "min",
    })
    grouped = grouped.sort_values(
        ["technical_rank", "source_rank", "year_mid_gap", "year_span", "partyfacts_id"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    return int(grouped.iloc[0]["partyfacts_id"])


def build_party_match_results(
    elections: pd.DataFrame,
    partyfacts: pd.DataFrame,
    min_score: int,
    tie_delta: float,
    out_dir: str | None = None,
    partyfacts_external: pd.DataFrame | None = None,
    vparty: pd.DataFrame | None = None,
    country_map: pd.DataFrame | None = None,
    year_window: int | None = None,
    overrides: pd.DataFrame | None = None,
    ledger: MergeLedger | None = None,
    step_id: str = "10_party_match",
) -> pd.DataFrame:
    aliases = build_party_aliases(
        partyfacts,
        partyfacts_external=partyfacts_external,
        vparty=vparty,
        country_map=country_map,
    )
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

    override_map = {}
    if overrides is not None and not overrides.empty:
        for _, row in overrides.iterrows():
            key = (row.get("iso3"), row.get("election_year"), row.get("party_name_raw"))
            override_map[key] = row

    results = []
    suggestions = []
    for _, row in party_names.iterrows():
        iso3 = row["iso3"]
        election_year = row["election_year"]
        party_name_raw = row["party_name"]

        override_key = (iso3, election_year, party_name_raw)
        if override_key in override_map:
            override = override_map[override_key]
            pid = override.get("partyfacts_id")
            results.append({
                "iso3": iso3,
                "election_year": election_year,
                "party_name_raw": party_name_raw,
                "party_name_norm": normalize_name(party_name_raw),
                "party_string_key": f"{iso3}|{election_year}|{party_name_raw}",
                "partyfacts_id": pid,
                "partyfacts_id_best": pid,
                "partyfacts_id_candidates": str(pid) if pd.notna(pid) else "",
                "match_score": 100,
                "match_method": "manual_override",
                "match_status": "manual_override",
                "match_status_best": "manual_override",
                "match_variant": "manual_override",
                "match_variant_best": "manual_override",
                "alias_source_key": "manual_override",
            })
            continue

        variants = generate_variants(party_name_raw)
        if not variants:
            results.append({
                "iso3": iso3,
                "election_year": election_year,
                "party_name_raw": party_name_raw,
                "party_name_norm": "",
                "party_string_key": f"{iso3}|{election_year}|{party_name_raw}",
                "partyfacts_id": None,
                "partyfacts_id_best": None,
                "partyfacts_id_candidates": "",
                "match_score": 0,
                "match_method": "missing",
                "match_status": "unmatched",
                "match_status_best": "unmatched",
                "match_variant": None,
                "match_variant_best": None,
                "alias_source_key": None,
            })
            continue

        variant_results = []
        for variant_key, variant_value in variants.items():
            res = match_party_name(variant_value, iso3, election_year, aliases, min_score, tie_delta, year_window=year_window)
            variant_results.append((variant_key, variant_value, res))

        def rank_status(status: str) -> int:
            return {"matched": 3, "matched_tiebreak": 2, "ambiguous": 1, "unmatched": 0}.get(status, 0)

        strict_candidates = [v for v in variant_results if v[2].status in {"matched", "matched_tiebreak"}]
        if strict_candidates:
            strict_candidates.sort(key=lambda v: (rank_status(v[2].status), v[2].score, v[0]), reverse=True)
            strict_variant, _, strict_res = strict_candidates[0]
        else:
            strict_variant, strict_res = None, None

        best_variant, _, best_res = max(variant_results, key=lambda v: (rank_status(v[2].status), v[2].score, v[0]))

        partyfacts_id_best = strict_res.partyfacts_id if strict_res else None
        match_status_best = strict_res.status if strict_res else best_res.status

        candidate_ids = []
        if strict_res and strict_res.candidates:
            candidate_ids = strict_res.candidates
        elif best_res and best_res.candidates:
            candidate_ids = best_res.candidates

        if partyfacts_id_best is None and best_res.status == "ambiguous" and candidate_ids:
            pool = _filter_pool(aliases, iso3, election_year, year_window)
            picked = _pick_best_candidate(candidate_ids, pool, election_year)
            partyfacts_id_best = picked
            match_status_best = "tied_flagged" if picked is not None else "ambiguous"

        if partyfacts_id_best is None and best_res.status == "unmatched":
            pool = _filter_pool(aliases, iso3, election_year, year_window)
            suggestions_list = _suggest_candidates(variants.get("norm", ""), pool, top_n=5)
            for rank, cand in enumerate(suggestions_list, start=1):
                suggestions.append({
                    "iso3": iso3,
                    "election_year": election_year,
                    "party_name_raw": party_name_raw,
                    "candidate_rank": rank,
                    "partyfacts_id": cand.get("partyfacts_id"),
                    "match_score": cand.get("score"),
                    "alias": cand.get("alias"),
                    "alias_source": cand.get("alias_source"),
                })
            if suggestions_list:
                candidate_ids = [c["partyfacts_id"] for c in suggestions_list]

        results.append({
            "iso3": iso3,
            "election_year": election_year,
            "party_name_raw": party_name_raw,
            "party_name_norm": variants.get("norm", ""),
            "party_string_key": f"{iso3}|{election_year}|{party_name_raw}",
            "partyfacts_id": strict_res.partyfacts_id if strict_res else None,
            "partyfacts_id_best": partyfacts_id_best,
            "partyfacts_id_candidates": "|".join(str(x) for x in candidate_ids) if candidate_ids else "",
            "match_score": strict_res.score if strict_res else best_res.score,
            "match_method": strict_res.method if strict_res else best_res.method,
            "match_status": strict_res.status if strict_res else best_res.status,
            "match_status_best": match_status_best,
            "match_variant": strict_variant,
            "match_variant_best": best_variant,
            "alias_source_key": strict_res.alias_source if strict_res else best_res.alias_source,
        })

    res_df = pd.DataFrame(results)
    if ledger is not None:
        logged_match(
            ledger,
            res_df,
            step_id=step_id,
            step_name="PartyFacts fuzzy match",
            left_table="party_strings",
            right_table="partyfacts_aliases",
            key_cols=["iso3", "election_year", "party_name_raw"],
            match_status_col="match_status",
            context_cols=[
                "party_name_norm",
                "partyfacts_id",
                "partyfacts_id_best",
                "partyfacts_id_candidates",
                "match_score",
                "match_method",
                "match_status",
                "match_status_best",
                "match_variant",
                "match_variant_best",
                "alias_source_key",
            ],
        )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res_df.to_parquet(out / "party_match_results.parquet", index=False)
        res_df[res_df["match_status"] == "unmatched"].to_csv(out / "10_party_match_unmatched.csv", index=False)
        res_df[res_df["match_status"] == "ambiguous"].to_csv(out / "10_party_match_ambiguous.csv", index=False)
        res_df[res_df["match_status"] == "matched_tiebreak"].to_csv(out / "10_party_match_tiebreak.csv", index=False)
        if suggestions:
            pd.DataFrame(suggestions).to_csv(out / "10_party_match_low_confidence_suggestions.csv", index=False)
    return res_df
