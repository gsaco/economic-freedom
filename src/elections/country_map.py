from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .standardize import normalize_name
from .io import read_parlgov_zip
from .merge_ledger import MergeLedger, logged_merge

try:  # optional
    import pycountry
except Exception:  # pragma: no cover
    pycountry = None

try:  # optional
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


def _coalesce_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NAN": pd.NA})


def _normalize_iso3(series: pd.Series) -> pd.Series:
    return _coalesce_str(series).str.upper()


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _ensure_map_cols(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_cols(
        df,
        [
            "source_name",
            "name_norm",
            "iso3",
            "match_score",
            "match_method",
            "match_status",
            "override_applied",
        ],
    )


def build_iso_reference(parlgov_zip: Path, cow2iso: pd.DataFrame | None = None, manual_path: Path | None = None) -> pd.DataFrame:
    tables = read_parlgov_zip(parlgov_zip)
    parlgov = tables["external_country_iso.csv"].copy()
    parlgov = parlgov.rename(columns={
        "country": "country_name",
        "iso2": "iso2",
        "iso3": "iso3",
        "isonumeric": "isonumeric",
    })
    parlgov["iso3"] = _normalize_iso3(parlgov["iso3"])
    parlgov["iso2"] = _coalesce_str(parlgov.get("iso2", pd.NA)).str.upper()
    parlgov["source"] = "parlgov"

    rows: list[dict] = []
    if pycountry is not None:
        for item in pycountry.countries:
            iso3 = getattr(item, "alpha_3", None)
            iso2 = getattr(item, "alpha_2", None)
            isonumeric = getattr(item, "numeric", None)
            for field in ["name", "official_name", "common_name"]:
                value = getattr(item, field, None)
                if value:
                    rows.append({
                        "country_name": value,
                        "iso3": iso3,
                        "iso2": iso2,
                        "isonumeric": isonumeric,
                        "source": "pycountry",
                        "alias_type": field,
                    })
    py_df = pd.DataFrame(rows)
    if not py_df.empty:
        py_df["iso3"] = _normalize_iso3(py_df["iso3"])
        py_df["iso2"] = _coalesce_str(py_df.get("iso2", pd.NA)).str.upper()

    cow_rows: list[dict] = []
    if cow2iso is not None:
        cow = cow2iso.copy()
        cow["iso3"] = _normalize_iso3(cow.get("iso3", pd.NA))
        cow["iso2"] = _coalesce_str(cow.get("iso2", pd.NA)).str.upper()
        cow["isonumeric"] = cow.get("iso_id", pd.NA)
        for col in ["cname", "cname_full", "statenme"]:
            if col in cow.columns:
                subset = cow[["iso3", "iso2", "isonumeric", col]].dropna(subset=[col])
                for _, row in subset.iterrows():
                    cow_rows.append({
                        "country_name": row[col],
                        "iso3": row["iso3"],
                        "iso2": row["iso2"],
                        "isonumeric": row["isonumeric"],
                        "source": "cow2iso",
                        "alias_type": col,
                    })
    cow_df = pd.DataFrame(cow_rows)

    manual_df = pd.DataFrame()
    if manual_path is not None and manual_path.exists():
        manual_df = pd.read_csv(manual_path)
        manual_df = manual_df.rename(columns={"country": "country_name"})
        manual_df["source"] = "manual"
        manual_df["iso3"] = _normalize_iso3(manual_df["iso3"])

    frames = [parlgov, py_df, cow_df, manual_df]
    combined = pd.concat([_ensure_cols(f, ["country_name", "iso3", "iso2", "isonumeric", "continent", "region", "source", "alias_type"]) for f in frames if f is not None and not f.empty], ignore_index=True)
    combined = combined.dropna(subset=["iso3", "country_name"]).copy()
    combined["iso3"] = _normalize_iso3(combined["iso3"])
    combined["isonumeric"] = pd.to_numeric(combined["isonumeric"], errors="coerce").astype("Int64")

    # fill iso2/isonumeric where missing using pycountry
    if not py_df.empty:
        iso_fill = py_df[["iso3", "iso2", "isonumeric"]].drop_duplicates()
        combined = combined.merge(iso_fill, on="iso3", how="left", suffixes=("", "_py"))
        combined["iso2"] = combined["iso2"].combine_first(combined["iso2_py"])
        combined["isonumeric"] = combined["isonumeric"].combine_first(combined["isonumeric_py"])
        combined = combined.drop(columns=["iso2_py", "isonumeric_py"])

    # fill region/continent from parlgov
    pg_meta = parlgov[["iso3", "continent", "region"]].drop_duplicates()
    combined = combined.merge(pg_meta, on="iso3", how="left", suffixes=("", "_pg"))
    combined["continent"] = combined["continent"].combine_first(combined["continent_pg"])
    combined["region"] = combined["region"].combine_first(combined["region_pg"])
    combined = combined.drop(columns=["continent_pg", "region_pg"])

    combined["country_name_norm"] = combined["country_name"].map(normalize_name)
    combined = combined.drop_duplicates(subset=["iso3", "country_name", "source", "alias_type"]).reset_index(drop=True)
    return combined


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


def map_names_to_iso(
    names: pd.Series,
    iso_ref: pd.DataFrame,
    min_score: int,
    tie_delta: float,
    overrides: pd.DataFrame | None = None,
    source_system: str | None = None,
) -> pd.DataFrame:
    iso_names = iso_ref[["country_name_norm", "iso3"]].dropna()
    choices = iso_names["country_name_norm"].tolist()
    override_map: dict[str, str] = {}
    overrides_applied: list[dict] = []
    if overrides is not None and source_system is not None and not overrides.empty:
        subset = overrides.copy()
        subset["source_system"] = subset["source_system"].astype(str)
        subset = subset[subset["source_system"].str.lower() == source_system.lower()]
        for _, row in subset.iterrows():
            key = str(row["source_name_raw"])
            iso3 = str(row["iso3"]).upper()
            override_map[key] = iso3
    out = []
    for name in sorted(set(n for n in names.dropna().astype(str))):
        norm = normalize_name(name)
        row = {
            "source_system": source_system,
            "source_name": name,
            "name_norm": norm,
            "iso3": None,
            "match_score": 0,
            "match_method": "unmatched",
            "match_status": "unmatched",
            "override_applied": False,
        }
        if name in override_map:
            iso3 = override_map[name]
            row.update({"iso3": iso3, "match_score": 100, "match_method": "override", "match_status": "matched", "override_applied": True})
            overrides_applied.append({"source_system": source_system, "source_name_raw": name, "iso3": iso3})
            out.append(row)
            continue
        if not norm:
            out.append(row)
            continue
        exact = iso_ref.loc[iso_ref["country_name_norm"] == norm, "iso3"].dropna().astype(str).unique()
        if len(exact) == 1:
            row.update({"iso3": exact[0], "match_score": 100, "match_method": "exact", "match_status": "matched"})
            out.append(row)
            continue
        if len(exact) > 1:
            row.update({"match_score": 100, "match_method": "exact", "match_status": "ambiguous"})
            out.append(row)
            continue
        # full scan to dedupe by iso3
        if not choices:
            out.append(row)
            continue
        scores = []
        if fuzz:
            for choice, iso3 in iso_names.itertuples(index=False):
                score = fuzz.token_sort_ratio(norm, choice)
                scores.append({"iso3": iso3, "score": int(score)})
        else:
            import difflib
            for choice, iso3 in iso_names.itertuples(index=False):
                score = int(difflib.SequenceMatcher(None, norm, choice).ratio() * 100)
                scores.append({"iso3": iso3, "score": score})
        score_df = pd.DataFrame(scores)
        score_df = score_df.groupby("iso3", as_index=False)["score"].max()
        score_df = score_df.sort_values(["score", "iso3"], ascending=[False, True], kind="mergesort")
        score1 = int(score_df.iloc[0]["score"]) if not score_df.empty else 0
        score2 = int(score_df.iloc[1]["score"]) if len(score_df) > 1 else 0
        best_iso3 = score_df.iloc[0]["iso3"] if not score_df.empty else None
        if best_iso3 is None:
            out.append(row)
            continue
        # tie check
        if score1 >= min_score and (score1 - score2) <= tie_delta:
            row.update({"match_score": score1, "match_method": "fuzzy", "match_status": "ambiguous"})
            out.append(row)
            continue
        if score1 >= min_score:
            row.update({"iso3": best_iso3, "match_score": score1, "match_method": "fuzzy", "match_status": "matched"})
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
    overrides: pd.DataFrame | None = None,
    name_maps: dict[str, pd.DataFrame] | None = None,
    ledger: MergeLedger | None = None,
) -> pd.DataFrame:
    audit_dir.mkdir(parents=True, exist_ok=True)
    has_nelda = not nelda.empty
    has_dpi = not dpi.empty
    has_efw = not efw.empty

    def _merge_country(
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        step_id: str,
        step_name: str,
        right_table: str,
        context_cols_right: list[str] | None = None,
    ) -> pd.DataFrame:
        if ledger is None:
            return left.merge(right, on="iso3", how="left")
        return logged_merge(
            ledger,
            left,
            right,
            how="left",
            on=["iso3"],
            step_id=step_id,
            step_name=step_name,
            left_table="country",
            right_table=right_table,
            context_cols_left=["iso3"],
            context_cols_right=context_cols_right,
        )

    def _merge_map(
        left: pd.DataFrame,
        right: pd.DataFrame,
        *,
        step_id: str,
        step_name: str,
        left_on: list[str],
        right_on: list[str],
        left_table: str,
        right_table: str,
        context_cols_left: list[str] | None = None,
        context_cols_right: list[str] | None = None,
    ) -> pd.DataFrame:
        if ledger is None:
            return left.merge(right, left_on=left_on, right_on=right_on, how="left")
        return logged_merge(
            ledger,
            left,
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
            step_id=step_id,
            step_name=step_name,
            left_table=left_table,
            right_table=right_table,
            context_cols_left=context_cols_left,
            context_cols_right=context_cols_right,
        )

    # name mappings
    ned_map = name_maps.get("ned") if name_maps and "ned" in name_maps else map_names_to_iso(ned["country"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="NED")
    clea_map = name_maps.get("clea") if name_maps and "clea" in name_maps else map_names_to_iso(clea["ctr_n"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="CLEA")
    if has_nelda:
        nelda_map = name_maps.get("nelda") if name_maps and "nelda" in name_maps else map_names_to_iso(nelda["country"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="NELDA")
    else:
        nelda_map = pd.DataFrame()
    if has_dpi:
        dpi_map = name_maps.get("dpi") if name_maps and "dpi" in name_maps else map_names_to_iso(dpi["countryname"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="DPI")
    else:
        dpi_map = pd.DataFrame()
    if has_efw:
        efw_map = name_maps.get("efw") if name_maps and "efw" in name_maps else map_names_to_iso(efw["Countries"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="EFW")
    else:
        efw_map = pd.DataFrame()

    ned_map = _ensure_map_cols(ned_map)
    clea_map = _ensure_map_cols(clea_map)
    nelda_map = _ensure_map_cols(nelda_map)
    dpi_map = _ensure_map_cols(dpi_map)
    efw_map = _ensure_map_cols(efw_map)

    ned_map.to_csv(audit_dir / "04_country_mapping_ned.csv", index=False)
    clea_map.to_csv(audit_dir / "04_country_mapping_clea.csv", index=False)
    if has_nelda:
        nelda_map.to_csv(audit_dir / "04_country_mapping_nelda.csv", index=False)
    if has_dpi:
        dpi_map.to_csv(audit_dir / "04_country_mapping_dpi.csv", index=False)
    if has_efw:
        efw_map.to_csv(audit_dir / "04_country_mapping_efw.csv", index=False)
    maps_for_overrides = [ned_map, clea_map]
    if has_nelda:
        maps_for_overrides.append(nelda_map)
    if has_dpi:
        maps_for_overrides.append(dpi_map)
    if has_efw:
        maps_for_overrides.append(efw_map)

    for df in maps_for_overrides:
        if "override_applied" not in df.columns:
            df["override_applied"] = False
    overrides_applied = pd.concat(
        [df.loc[df["override_applied"]] for df in maps_for_overrides],
        ignore_index=True,
    )
    if not overrides_applied.empty:
        overrides_applied.to_csv(audit_dir / "04_country_mapping_overrides_applied.csv", index=False)

    # COW code mapping from cow2iso
    cow_map = build_cowcode_map(cow2iso, iso_ref=iso_ref)
    cow_map.to_csv(audit_dir / "04_country_mapping_cow.csv", index=False)

    # Build base country table from iso_ref (one row per iso3)
    iso_base = iso_ref.copy()
    source_rank = {"parlgov": 0, "pycountry": 1, "cow2iso": 2, "manual": 3}
    iso_base["source_rank"] = iso_base["source"].map(source_rank).fillna(9).astype(int)
    iso_base = iso_base.sort_values(["iso3", "source_rank", "country_name"], kind="mergesort")
    iso_base = iso_base.drop_duplicates(subset=["iso3"], keep="first")
    country = iso_base[[
        "iso3",
        "iso2",
        "isonumeric",
        "country_name",
        "continent",
        "region",
    ]].copy()
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
    country = _merge_country(
        country,
        cow_rev[["iso3", "cowcode"]].rename(columns={"cowcode": "cow_code"}),
        step_id="04_country_cow_code",
        step_name="Attach COW codes",
        right_table="cow2iso",
        context_cols_right=["cow_code"],
    )

    # Attach NELDA-derived ccode (modal by iso3)
    if has_nelda:
        nelda_named = _merge_map(
            nelda,
            nelda_map[["source_name", "iso3"]],
            step_id="04_country_nelda_map_join",
            step_name="Attach ISO3 to NELDA",
            left_on=["country"],
            right_on=["source_name"],
            left_table="nelda",
            right_table="nelda_map",
            context_cols_left=["country", "ccode"],
            context_cols_right=["source_name", "iso3"],
        )
        nelda_ccode_from_nelda = nelda_named.dropna(subset=["iso3", "ccode"]).groupby("iso3")["ccode"].agg(lambda x: x.value_counts().index[0])
        country = _merge_country(
            country,
            nelda_ccode_from_nelda.rename("nelda_ccode_from_nelda").reset_index(),
            step_id="04_country_nelda_ccode_from_nelda",
            step_name="Attach NELDA modal ccode",
            right_table="nelda",
            context_cols_right=["nelda_ccode_from_nelda"],
        )

    # CLEA ctr
    clea_named = _merge_map(
        clea,
        clea_map[["source_name", "iso3"]],
        step_id="04_country_clea_map_join",
        step_name="Attach ISO3 to CLEA",
        left_on=["ctr_n"],
        right_on=["source_name"],
        left_table="clea",
        right_table="clea_map",
        context_cols_left=["ctr_n", "ctr"],
        context_cols_right=["source_name", "iso3"],
    )
    clea_ctr = clea_named.dropna(subset=["iso3"]).groupby("iso3")["ctr"].agg(lambda x: x.value_counts().index[0])
    country = _merge_country(
        country,
        clea_ctr.rename("clea_ctr").reset_index(),
        step_id="04_country_clea_ctr",
        step_name="Attach CLEA ctr",
        right_table="clea",
        context_cols_right=["clea_ctr"],
    )

    # V-Party country_id
    vparty_name_map = name_maps.get("vparty") if name_maps and "vparty" in name_maps else map_names_to_iso(vparty["country_name"], iso_ref, min_score, tie_delta, overrides=overrides, source_system="VPARTY")
    vparty_map = _merge_map(
        vparty,
        vparty_name_map,
        step_id="04_country_vparty_map_join",
        step_name="Attach ISO3 to V-Party",
        left_on=["country_name"],
        right_on=["source_name"],
        left_table="vparty",
        right_table="vparty_map",
        context_cols_left=["country_name", "country_id"],
        context_cols_right=["source_name", "iso3"],
    )
    vparty_id = vparty_map.dropna(subset=["iso3"]).groupby("iso3")["country_id"].agg(lambda x: x.value_counts().index[0])
    country = _merge_country(
        country,
        vparty_id.rename("vdem_country_id").reset_index(),
        step_id="04_country_vdem_id",
        step_name="Attach V-Dem country id",
        right_table="vparty",
        context_cols_right=["vdem_country_id"],
    )

    # DPI IFS
    if has_dpi:
        dpi_named = _merge_map(
            dpi,
            dpi_map[["source_name", "iso3"]],
            step_id="04_country_dpi_map_join",
            step_name="Attach ISO3 to DPI",
            left_on=["countryname"],
            right_on=["source_name"],
            left_table="dpi",
            right_table="dpi_map",
            context_cols_left=["countryname"],
            context_cols_right=["source_name", "iso3"],
        )
        dpi_ifs = dpi_named.dropna(subset=["iso3"]).groupby("iso3")["ifs"].agg(lambda x: x.value_counts().index[0])
        country = _merge_country(
            country,
            dpi_ifs.rename("dpi_ifs").reset_index(),
            step_id="04_country_dpi_ifs",
            step_name="Attach DPI ifs",
            right_table="dpi",
            context_cols_right=["dpi_ifs"],
        )

    # Name representatives per source
    def most_common(series: pd.Series):
        vc = series.value_counts(dropna=True)
        return vc.index[0] if not vc.empty else None

    # NED names (use name map to iso3)
    ned_named = _merge_map(
        ned,
        ned_map[["source_name", "iso3"]],
        step_id="04_country_ned_map_join",
        step_name="Attach ISO3 to NED",
        left_on=["country"],
        right_on=["source_name"],
        left_table="ned",
        right_table="ned_map",
        context_cols_left=["country"],
        context_cols_right=["source_name", "iso3"],
    )
    ned_name = ned_named.dropna(subset=["iso3"]).groupby("iso3")["country"].agg(most_common)
    country = _merge_country(
        country,
        ned_name.rename("name_ned").reset_index(),
        step_id="04_country_name_ned",
        step_name="Attach NED country names",
        right_table="ned",
        context_cols_right=["name_ned"],
    )

    clea_name = clea_named.dropna(subset=["iso3"]).groupby("iso3")["ctr_n"].agg(most_common)
    country = _merge_country(
        country,
        clea_name.rename("name_clea").reset_index(),
        step_id="04_country_name_clea",
        step_name="Attach CLEA country names",
        right_table="clea",
        context_cols_right=["name_clea"],
    )

    if has_nelda:
        nelda_named = _merge_map(
            nelda,
            nelda_map[["source_name", "iso3"]],
            step_id="04_country_nelda_map_join_2",
            step_name="Attach ISO3 to NELDA (names)",
            left_on=["country"],
            right_on=["source_name"],
            left_table="nelda",
            right_table="nelda_map",
            context_cols_left=["country"],
            context_cols_right=["source_name", "iso3"],
        )
        nelda_name = nelda_named.dropna(subset=["iso3"]).groupby("iso3")["country"].agg(most_common)
        country = _merge_country(
            country,
            nelda_name.rename("name_nelda").reset_index(),
            step_id="04_country_name_nelda",
            step_name="Attach NELDA country names",
            right_table="nelda",
            context_cols_right=["name_nelda"],
        )

    vparty_named = vparty_map.dropna(subset=["iso3"]).groupby("iso3")["country_name"].agg(most_common)
    country = _merge_country(
        country,
        vparty_named.rename("name_vparty").reset_index(),
        step_id="04_country_name_vparty",
        step_name="Attach V-Party country names",
        right_table="vparty",
        context_cols_right=["name_vparty"],
    )

    if has_dpi:
        dpi_name = dpi_named.dropna(subset=["iso3"]).groupby("iso3")["countryname"].agg(most_common)
        country = _merge_country(
            country,
            dpi_name.rename("name_dpi").reset_index(),
            step_id="04_country_name_dpi",
            step_name="Attach DPI country names",
            right_table="dpi",
            context_cols_right=["name_dpi"],
        )

    if has_efw:
        efw_named = _merge_map(
            efw,
            efw_map[["source_name", "iso3"]],
            step_id="04_country_efw_map_join",
            step_name="Attach ISO3 to EFW",
            left_on=["Countries"],
            right_on=["source_name"],
            left_table="efw",
            right_table="efw_map",
            context_cols_left=["Countries"],
            context_cols_right=["source_name", "iso3"],
        )
        efw_name = efw_named.dropna(subset=["iso3"]).groupby("iso3")["Countries"].agg(most_common)
        country = _merge_country(
            country,
            efw_name.rename("name_efw").reset_index(),
            step_id="04_country_name_efw",
            step_name="Attach EFW country names",
            right_table="efw",
            context_cols_right=["name_efw"],
        )

    # unmatched/ambiguous lists
    def write_unmatched(df: pd.DataFrame, name: str):
        df.loc[df["match_status"].isin(["unmatched", "ambiguous"])].to_csv(audit_dir / f"04_country_mapping_{name}_unmatched.csv", index=False)

    write_unmatched(ned_map, "ned")
    write_unmatched(clea_map, "clea")
    if has_nelda:
        write_unmatched(nelda_map, "nelda")
    if has_dpi:
        write_unmatched(dpi_map, "dpi")
    if has_efw:
        write_unmatched(efw_map, "efw")

    return country
