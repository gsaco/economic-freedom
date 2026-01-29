from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .standardize import normalize_name
from .io import read_parlgov_zip

try:  # optional
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


def build_iso_reference(parlgov_zip: Path) -> pd.DataFrame:
    tables = read_parlgov_zip(parlgov_zip)
    iso = tables["external_country_iso.csv"].copy()
    iso = iso.rename(columns={
        "country": "country_name",
        "iso2": "iso2",
        "iso3": "iso3",
        "isonumeric": "isonumeric",
    })
    iso["country_name_norm"] = iso["country_name"].map(normalize_name)
    iso["iso3"] = iso["iso3"].astype(str).str.upper()
    return iso


def _fuzzy_best(query: str, choices: list[str]) -> Tuple[str | None, int, int]:
    if not query:
        return None, 0, 0
    if not choices:
        return None, 0, 0
    if process and fuzz:
        # return top two scores
        results = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=2)
        if not results:
            return None, 0, 0
        best, score1, _ = results[0]
        score2 = results[1][1] if len(results) > 1 else 0
        return best, int(score1), int(score2)
    # fallback to difflib
    import difflib
    scores = [(c, int(difflib.SequenceMatcher(None, query, c).ratio() * 100)) for c in choices]
    scores.sort(key=lambda x: (-x[1], x[0]))
    best, score1 = scores[0]
    score2 = scores[1][1] if len(scores) > 1 else 0
    return best, int(score1), int(score2)


def map_names_to_iso(names: pd.Series, iso_ref: pd.DataFrame, min_score: int, tie_delta: float) -> pd.DataFrame:
    iso_names = iso_ref[["country_name_norm", "iso3"]].dropna()
    choices = iso_names["country_name_norm"].tolist()
    out = []
    for name in sorted(set(n for n in names.dropna().astype(str))):
        norm = normalize_name(name)
        row = {
            "source_name": name,
            "name_norm": norm,
            "iso3": None,
            "match_score": 0,
            "match_method": "unmatched",
            "match_status": "unmatched",
        }
        if not norm:
            out.append(row)
            continue
        exact = iso_ref.loc[iso_ref["country_name_norm"] == norm, "iso3"]
        if len(exact) == 1:
            row.update({"iso3": exact.iloc[0], "match_score": 100, "match_method": "exact", "match_status": "matched"})
            out.append(row)
            continue
        if len(exact) > 1:
            row.update({"match_score": 100, "match_method": "exact", "match_status": "ambiguous"})
            out.append(row)
            continue
        best, score1, score2 = _fuzzy_best(norm, choices)
        if best is None:
            out.append(row)
            continue
        # tie check
        if score1 >= min_score and (score1 - score2) <= tie_delta:
            row.update({"match_score": score1, "match_method": "fuzzy", "match_status": "ambiguous"})
            out.append(row)
            continue
        if score1 >= min_score:
            iso3 = iso_ref.loc[iso_ref["country_name_norm"] == best, "iso3"].iloc[0]
            row.update({"iso3": iso3, "match_score": score1, "match_method": "fuzzy", "match_status": "matched"})
            out.append(row)
            continue
        row.update({"match_score": score1, "match_method": "fuzzy", "match_status": "unmatched"})
        out.append(row)
    return pd.DataFrame(out)


def build_cowcode_map(cow2iso: pd.DataFrame, iso_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    df = cow2iso.copy()
    # normalize expected columns
    if "cow_id" in df.columns and "cowcode" not in df.columns:
        df = df.rename(columns={"cow_id": "cowcode"})
    df["cowcode"] = pd.to_numeric(df.get("cowcode"), errors="coerce").astype("Int64")
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype(str).str.upper()
        df.loc[df["iso3"].isin(["", "NAN", "NONE"]), "iso3"] = pd.NA
    for col in ["valid_from", "valid_until"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["match_status"] = "matched"
    df.loc[df["cowcode"].isna() | df["iso3"].isna(), "match_status"] = "unmatched"
    if iso_ref is not None and "iso3" in df.columns:
        iso_set = set(iso_ref["iso3"].astype(str))
        df["iso3_in_ref"] = df["iso3"].isin(iso_set)
    return df


def build_country_table(
    iso_ref: pd.DataFrame,
    ned: pd.DataFrame,
    clea: pd.DataFrame,
    nelda: pd.DataFrame,
    vparty: pd.DataFrame,
    cow2iso: pd.DataFrame,
    dpi: pd.DataFrame,
    efw: pd.DataFrame,
    min_score: int,
    tie_delta: float,
    audit_dir: Path,
) -> pd.DataFrame:
    audit_dir.mkdir(parents=True, exist_ok=True)

    # name mappings
    ned_map = map_names_to_iso(ned["country"], iso_ref, min_score, tie_delta)
    clea_map = map_names_to_iso(clea["ctr_n"], iso_ref, min_score, tie_delta)
    nelda_map = map_names_to_iso(nelda["country"], iso_ref, min_score, tie_delta)
    dpi_map = map_names_to_iso(dpi["countryname"], iso_ref, min_score, tie_delta)
    efw_map = map_names_to_iso(efw["Countries"], iso_ref, min_score, tie_delta)

    ned_map.to_csv(audit_dir / "04_country_mapping_ned.csv", index=False)
    clea_map.to_csv(audit_dir / "04_country_mapping_clea.csv", index=False)
    nelda_map.to_csv(audit_dir / "04_country_mapping_nelda.csv", index=False)
    dpi_map.to_csv(audit_dir / "04_country_mapping_dpi.csv", index=False)
    efw_map.to_csv(audit_dir / "04_country_mapping_efw.csv", index=False)

    # COW code mapping from cow2iso
    cow_map = build_cowcode_map(cow2iso, iso_ref=iso_ref)
    cow_map.to_csv(audit_dir / "04_country_mapping_cow.csv", index=False)

    # Build base country table from iso_ref
    country = iso_ref[[
        "iso3",
        "iso2",
        "isonumeric",
        "country_name",
        "continent",
        "region",
    ]].drop_duplicates().copy()
    country = country.rename(columns={"country_name": "country_name_std"})

    # Attach cow_code (mode per iso3)
    cow_rev = cow_map.dropna(subset=["iso3", "cowcode"]).copy()
    if "valid_from" in cow_rev.columns:
        cow_rev["valid_from_rank"] = pd.to_numeric(cow_rev["valid_from"], errors="coerce").astype(float).fillna(-np.inf)
    else:
        cow_rev["valid_from_rank"] = -np.inf
    if "valid_until" in cow_rev.columns:
        cow_rev["valid_until_rank"] = pd.to_numeric(cow_rev["valid_until"], errors="coerce").astype(float).fillna(np.inf)
    else:
        cow_rev["valid_until_rank"] = np.inf
    cow_rev = cow_rev.sort_values(
        ["iso3", "valid_from_rank", "valid_until_rank", "cowcode"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    cow_rev = cow_rev.groupby("iso3", as_index=False).head(1)
    country = country.merge(cow_rev[["iso3", "cowcode"]].rename(columns={"cowcode": "cow_code"}), on="iso3", how="left")

    # Attach NELDA ccode using cow map
    nelda_ccode = cow_rev[["iso3", "cowcode"]].rename(columns={"cowcode": "nelda_ccode"})
    country = country.merge(nelda_ccode, on="iso3", how="left")

    # CLEA ctr
    clea_named = clea.merge(clea_map[["source_name", "iso3"]], left_on="ctr_n", right_on="source_name", how="left")
    clea_ctr = clea_named.dropna(subset=["iso3"]).groupby("iso3")["ctr"].agg(lambda x: x.value_counts().index[0])
    country = country.merge(clea_ctr.rename("clea_ctr"), on="iso3", how="left")

    # V-Party country_id
    vparty_map = vparty.merge(map_names_to_iso(vparty["country_name"], iso_ref, min_score, tie_delta), left_on="country_name", right_on="source_name", how="left")
    vparty_id = vparty_map.dropna(subset=["iso3"]).groupby("iso3")["country_id"].agg(lambda x: x.value_counts().index[0])
    country = country.merge(vparty_id.rename("vdem_country_id"), on="iso3", how="left")

    # DPI IFS
    dpi_named = dpi.merge(dpi_map[["source_name", "iso3"]], left_on="countryname", right_on="source_name", how="left")
    dpi_ifs = dpi_named.dropna(subset=["iso3"]).groupby("iso3")["ifs"].agg(lambda x: x.value_counts().index[0])
    country = country.merge(dpi_ifs.rename("dpi_ifs"), on="iso3", how="left")

    # Name representatives per source
    def most_common(series: pd.Series):
        vc = series.value_counts(dropna=True)
        return vc.index[0] if not vc.empty else None

    # NED names (use name map to iso3)
    ned_named = ned.merge(ned_map[["source_name", "iso3"]], left_on="country", right_on="source_name", how="left")
    ned_name = ned_named.dropna(subset=["iso3"]).groupby("iso3")["country"].agg(most_common)
    country = country.merge(ned_name.rename("name_ned"), on="iso3", how="left")

    clea_name = clea_named.dropna(subset=["iso3"]).groupby("iso3")["ctr_n"].agg(most_common)
    country = country.merge(clea_name.rename("name_clea"), on="iso3", how="left")

    nelda_named = nelda.merge(nelda_map[["source_name", "iso3"]], left_on="country", right_on="source_name", how="left")
    nelda_name = nelda_named.dropna(subset=["iso3"]).groupby("iso3")["country"].agg(most_common)
    country = country.merge(nelda_name.rename("name_nelda"), on="iso3", how="left")

    vparty_named = vparty_map.dropna(subset=["iso3"]).groupby("iso3")["country_name"].agg(most_common)
    country = country.merge(vparty_named.rename("name_vparty"), on="iso3", how="left")

    dpi_name = dpi_named.dropna(subset=["iso3"]).groupby("iso3")["countryname"].agg(most_common)
    country = country.merge(dpi_name.rename("name_dpi"), on="iso3", how="left")

    efw_named = efw.merge(efw_map[["source_name", "iso3"]], left_on="Countries", right_on="source_name", how="left")
    efw_name = efw_named.dropna(subset=["iso3"]).groupby("iso3")["Countries"].agg(most_common)
    country = country.merge(efw_name.rename("name_efw"), on="iso3", how="left")

    # unmatched/ambiguous lists
    def write_unmatched(df: pd.DataFrame, name: str):
        df.loc[df["match_status"].isin(["unmatched", "ambiguous"])].to_csv(audit_dir / f"04_country_mapping_{name}_unmatched.csv", index=False)

    write_unmatched(ned_map, "ned")
    write_unmatched(clea_map, "clea")
    write_unmatched(nelda_map, "nelda")
    write_unmatched(dpi_map, "dpi")
    write_unmatched(efw_map, "efw")

    return country
