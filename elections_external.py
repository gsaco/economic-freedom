from __future__ import annotations

from pathlib import Path
import hashlib
import logging
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from elections_core import (
    PipelinePaths,
    best_fuzzy_match,
    dedupe_by_keys,
    harmonize_keys,
    merge_with_diagnostics,
    normalize_name,
    validate_required_columns,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download_external(paths: PipelinePaths, logger: logging.Logger) -> None:
    downloads = {
        "partyfacts_external_parties.csv": "https://partyfacts.herokuapp.com/download/external-parties-csv/",
        "partyfacts_core_parties.csv": "https://partyfacts.herokuapp.com/download/core-parties-csv/",
        "ches_1999_2024.csv": "https://www.chesdata.eu/s/1999-2024_CHES_dataset_means.csv",
        "elff_partypos_summaries.csv": "https://www.elff.eu/data/party-positions/partypos-summaries.csv",
        "des_es_data_v50.zip": "https://mattgolder.com/files/research/es_data-v50.zip",
        "parlgov.zip": "https://www.parlgov.org/data/parlgov-development_csv-utf-8.zip",
        "dpi2020_stata13.zip": "https://data.iadb.org/file/download/4ee05a1d-f142-4066-a3d9-33a20fd64c78",
    }
    manifest = []
    for filename, url in downloads.items():
        dest = paths.external / filename
        if dest.exists():
            manifest.append({"file": filename, "url": url, "sha256": sha256(dest), "status": "exists"})
            continue
        try:
            logger.info("Downloading %s -> %s", url, dest)
            download(url, dest)
            manifest.append({"file": filename, "url": url, "sha256": sha256(dest), "status": "downloaded"})
        except Exception as exc:  # pragma: no cover - network variance
            manifest.append({"file": filename, "url": url, "sha256": None, "status": f"failed: {exc}"})
            logger.warning("Download failed for %s: %s", url, exc)

    pd.DataFrame(manifest).to_csv(paths.external / "download_manifest.csv", index=False)


def load_master(paths: PipelinePaths) -> pd.DataFrame:
    return pd.read_parquet(paths.processed / "elections_master.parquet")


def load_efw_country_map(paths: PipelinePaths) -> pd.DataFrame:
    efw_path = paths.raw / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx"
    efw = pd.read_excel(efw_path, sheet_name="EFW Panel Dataset")
    efw = efw.rename(columns={"ISO_Code": "iso3", "Countries": "country_name"})
    efw["country_name_norm"] = efw["country_name"].map(normalize_name)
    return efw[["iso3", "country_name", "country_name_norm"]].drop_duplicates()


def load_partyfacts(paths: PipelinePaths) -> pd.DataFrame:
    preferred = paths.root / "partyfacts-external-parties.csv"
    if preferred.exists():
        return pd.read_csv(preferred)
    return pd.read_csv(paths.external / "partyfacts_external_parties.csv")


def load_ches(paths: PipelinePaths) -> pd.DataFrame:
    ches = pd.read_csv(paths.external / "ches_1999_2024.csv")
    ches["country_norm"] = ches["country"].map(normalize_name)
    ches["party_norm"] = ches["party"].map(normalize_name)
    return ches


def load_elff(paths: PipelinePaths) -> pd.DataFrame:
    elff = pd.read_csv(paths.external / "elff_partypos_summaries.csv")
    elff["country_norm"] = elff["countryname"].map(normalize_name)
    elff["party_norm"] = elff["partyname"].map(normalize_name)
    return elff


def load_parlgov_party(paths: PipelinePaths) -> pd.DataFrame:
    with zipfile.ZipFile(paths.external / "parlgov.zip") as z:
        with z.open("view_party.csv") as f:
            df = pd.read_csv(f)
    df["country_norm"] = df["country_name"].map(normalize_name)
    df["party_norm"] = df["party_name_english"].fillna(df["party_name"]).map(normalize_name)
    return df


def load_des(paths: PipelinePaths) -> pd.DataFrame:
    with zipfile.ZipFile(paths.external / "des_es_data_v50.zip") as z:
        with z.open("es_data-v5_0.csv") as f:
            df = pd.read_csv(f)
    df["country_norm"] = df["country"].map(normalize_name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_idea(paths: PipelinePaths, logger: logging.Logger) -> pd.DataFrame:
    idea_path = paths.external / "idea_export_electoral_system_design_database.xlsx"
    idea = pd.read_excel(idea_path)
    idea = idea.rename(columns={
        "ISO3": "iso3",
        "Year": "year",
        "Electoral system family": "idea_system_family",
        "Electoral system for national legislature": "idea_leg_system",
        "Electoral system for the president": "idea_pres_system",
        "Number of tiers": "idea_tiers",
        "Legislative size (directly elected)": "idea_leg_size_direct",
        "Legislative size (voting members)": "idea_leg_size_voting",
    })
    idea["iso3"] = idea["iso3"].astype(str).str.upper()
    idea["year"] = pd.to_numeric(idea["year"], errors="coerce").astype("Int64")
    for col in [
        "idea_system_family",
        "idea_leg_system",
        "idea_pres_system",
        "idea_tiers",
        "idea_leg_size_direct",
        "idea_leg_size_voting",
    ]:
        if col in idea.columns:
            idea[col] = idea[col].astype("string")
    idea = dedupe_by_keys(idea, ["iso3", "year"], logger=logger, name="idea")
    return idea


def load_dpi(paths: PipelinePaths, logger: logging.Logger) -> pd.DataFrame:
    dpi_path = paths.external / "dpi" / "DPI2020" / "dpi2020.csv"
    dpi = pd.read_csv(dpi_path, low_memory=False)
    dpi = dpi.rename(columns={"countryname": "country_name", "year": "year"})
    dpi["country_norm"] = dpi["country_name"].map(normalize_name)
    dpi["year"] = pd.to_numeric(dpi["year"], errors="coerce").astype("Int64")
    dpi = dedupe_by_keys(dpi, ["country_norm", "year"], logger=logger, name="dpi")
    return dpi


def map_iso_to_country(iso3: str, efw: pd.DataFrame, target_countries: pd.Series, min_score: int = 70) -> tuple:
    if iso3 is None or pd.isna(iso3):
        return (None, 0)
    row = efw.loc[efw["iso3"] == iso3]
    if row.empty:
        return (None, 0)
    name = row["country_name"].iloc[0]
    match, score = best_fuzzy_match(name, target_countries.tolist(), min_score=min_score)
    return (match, score) if score >= min_score else (None, score)


def rescale(series: pd.Series, target_min: float = 0.0, target_max: float = 6.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    minv = s.min()
    maxv = s.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (s - minv) / (maxv - minv) * (target_max - target_min) + target_min


def build_country_maps(
    efw: pd.DataFrame,
    ches: pd.DataFrame,
    elff: pd.DataFrame,
    parlgov: pd.DataFrame,
    des: pd.DataFrame,
    dpi: pd.DataFrame,
) -> pd.DataFrame:
    unique_iso = efw["iso3"].dropna().unique().tolist()
    rows = []
    for iso in unique_iso:
        ches_match, ches_score = map_iso_to_country(iso, efw, ches["country_norm"].dropna().unique(), min_score=75)
        elff_match, elff_score = map_iso_to_country(iso, efw, elff["country_norm"].dropna().unique(), min_score=75)
        parlgov_match, parlgov_score = map_iso_to_country(iso, efw, parlgov["country_norm"].dropna().unique(), min_score=75)
        des_match, des_score = map_iso_to_country(iso, efw, des["country_norm"].dropna().unique(), min_score=90)
        dpi_match, dpi_score = map_iso_to_country(iso, efw, dpi["country_norm"].dropna().unique(), min_score=85)
        rows.append({
            "iso3": iso,
            "ches_country_norm": ches_match,
            "ches_score": ches_score,
            "elff_country_norm": elff_match,
            "elff_score": elff_score,
            "parlgov_country_norm": parlgov_match,
            "parlgov_score": parlgov_score,
            "des_country_norm": des_match,
            "des_score": des_score,
            "dpi_country_norm": dpi_match,
            "dpi_score": dpi_score,
        })
    return pd.DataFrame(rows)


def match_party_name(party_name: str, party_list: pd.Series) -> tuple:
    if party_name is None or pd.isna(party_name):
        return (None, 0)
    match, score = best_fuzzy_match(party_name, party_list.tolist(), min_score=70)
    return (match, score)


def get_ches_ideology(ches: pd.DataFrame, country_norm: str, party_norm: str, year: int) -> tuple:
    subset = ches[ches["country_norm"] == country_norm]
    if subset.empty:
        return (np.nan, np.nan)
    if party_norm:
        party_match, _ = match_party_name(party_norm, subset["party_norm"].dropna().unique())
        if party_match is None:
            return (np.nan, np.nan)
        subset = subset[subset["party_norm"] == party_match]
    else:
        return (np.nan, np.nan)
    subset = subset.dropna(subset=["lrecon"]).copy()
    if subset.empty:
        return (np.nan, np.nan)
    subset["year_diff"] = (subset["electionyear"] - year).abs()
    closest = subset.sort_values("year_diff").iloc[0]
    return (closest["lrecon"], closest["year_diff"])


def get_elff_ideology(elff: pd.DataFrame, country_norm: str, party_norm: str, year: int) -> tuple:
    subset = elff[elff["country_norm"] == country_norm]
    if subset.empty:
        return (np.nan, np.nan)
    if party_norm:
        party_match, _ = match_party_name(party_norm, subset["party_norm"].dropna().unique())
        if party_match is None:
            return (np.nan, np.nan)
        subset = subset[subset["party_norm"] == party_match]
    else:
        return (np.nan, np.nan)
    subset = subset.dropna(subset=["econlr"]).copy()
    if subset.empty:
        return (np.nan, np.nan)
    subset["year_diff"] = (subset["year"] - year).abs()
    closest = subset.sort_values("year_diff").iloc[0]
    return (closest["econlr"], closest["year_diff"])


def get_parlgov_ideology(parlgov: pd.DataFrame, country_norm: str, party_norm: str) -> tuple:
    subset = parlgov[parlgov["country_norm"] == country_norm]
    if subset.empty:
        return (np.nan, np.nan)
    party_match, _ = match_party_name(party_norm, subset["party_norm"].dropna().unique())
    if party_match is None:
        return (np.nan, np.nan)
    row = subset[subset["party_norm"] == party_match].iloc[0]
    return (row["state_market"], row["left_right"])


def map_partyfacts(pf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pf_vparty = pf[pf["dataset_key"] == "vparty"].copy()
    pf_ches = pf[pf["dataset_key"] == "ches"].copy()
    pf_vparty["dataset_party_id"] = pd.to_numeric(pf_vparty["dataset_party_id"], errors="coerce")
    pf_ches["dataset_party_id"] = pd.to_numeric(pf_ches["dataset_party_id"], errors="coerce")
    return pf_vparty, pf_ches


def attach_des(df: pd.DataFrame, des: pd.DataFrame, country_map: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(country_map[["iso3", "des_country_norm"]], on="iso3", how="left")

    des = des.copy()

    def match_des(row):
        des_country = row.get("des_country_norm")
        if pd.isna(des_country):
            des_country = row.get("des_country_norm_y")
        if pd.isna(des_country):
            des_country = row.get("des_country_norm_x")
        if pd.isna(des_country) or pd.isna(row.get("date")):
            return pd.Series({})
        subset = des[des["country_norm"] == des_country].copy()
        if subset.empty:
            return pd.Series({})
        subset["date_diff"] = (subset["date"] - row.get("date")).abs()
        closest = subset.sort_values("date_diff").iloc[0]
        return pd.Series({
            "des_elecrule": closest.get("elecrule"),
            "des_tier1_formula": closest.get("tier1_formula"),
            "des_tier1_avemag": closest.get("tier1_avemag"),
            "des_mixed_type": closest.get("mixed_type"),
            "des_date_diff": closest.get("date_diff").days if pd.notna(closest.get("date_diff")) else np.nan,
        })

    des_cols = df.apply(match_des, axis=1)
    if not des_cols.empty:
        df = pd.concat([df, des_cols], axis=1)
    for col in ["des_elecrule", "des_tier1_formula", "des_tier1_avemag", "des_mixed_type", "des_date_diff"]:
        if col not in df.columns:
            df[col] = np.nan
    df["des_match"] = df["des_elecrule"].notna()
    df["des_match_2y"] = df["des_match"] & (df["des_date_diff"] <= 730)
    df["des_match_5y"] = df["des_match"] & (df["des_date_diff"] <= 1825)
    return df


def build_external_master(paths: PipelinePaths, logger: logging.Logger, strict: bool = False) -> pd.DataFrame:
    df = harmonize_keys(load_master(paths))
    efw = load_efw_country_map(paths)
    pf = load_partyfacts(paths)
    ches = load_ches(paths)
    elff = load_elff(paths)
    parlgov = load_parlgov_party(paths)
    des = load_des(paths)
    idea = load_idea(paths, logger)
    dpi = load_dpi(paths, logger)

    country_map = build_country_maps(efw, ches, elff, parlgov, des, dpi)
    country_map.to_csv(paths.reports / "external_country_map.csv", index=False)

    pf_vparty, pf_ches = map_partyfacts(pf)
    vparty_to_pf = (
        pf_vparty.dropna(subset=["dataset_party_id", "partyfacts_id"])
        .groupby("dataset_party_id")["partyfacts_id"]
        .agg(lambda x: x.mode().iloc[0])
    )
    pf_to_ches = (
        pf_ches.dropna(subset=["dataset_party_id", "partyfacts_id"])
        .groupby("partyfacts_id")["dataset_party_id"]
        .agg(lambda x: x.mode().iloc[0])
    )

    df["partyfacts_id1"] = df["party1_v2paid"].map(vparty_to_pf)
    df["partyfacts_id2"] = df["party2_v2paid"].map(vparty_to_pf)
    df["ches_party_id1"] = df["partyfacts_id1"].map(pf_to_ches)
    df["ches_party_id2"] = df["partyfacts_id2"].map(pf_to_ches)

    ches_party_map = ches.dropna(subset=["party_id", "lrecon"]).groupby("party_id")["lrecon"].mean()
    df["ideo_ches1_pf"] = df["ches_party_id1"].map(ches_party_map)
    df["ideo_ches2_pf"] = df["ches_party_id2"].map(ches_party_map)

    df = merge_with_diagnostics(
        df,
        country_map,
        keys=["iso3"],
        how="left",
        name="external_country_map",
        logger=logger,
    )

    df["party1_norm"] = df["party_1_name"].map(normalize_name)
    df["party2_norm"] = df["party_2_name"].map(normalize_name)

    ches_res = df.apply(
        lambda r: get_ches_ideology(
            ches,
            r.get("ches_country_norm"),
            r.get("party1_norm"),
            int(r.get("year")) if pd.notna(r.get("year")) else 0,
        ),
        axis=1,
    )
    df["ideo_ches1"] = [x[0] for x in ches_res]
    df["ches_year_diff1"] = [x[1] for x in ches_res]

    ches_res2 = df.apply(
        lambda r: get_ches_ideology(
            ches,
            r.get("ches_country_norm"),
            r.get("party2_norm"),
            int(r.get("year")) if pd.notna(r.get("year")) else 0,
        ),
        axis=1,
    )
    df["ideo_ches2"] = [x[0] for x in ches_res2]
    df["ches_year_diff2"] = [x[1] for x in ches_res2]

    df["ideo_ches1_combined"] = df["ideo_ches1"].combine_first(df["ideo_ches1_pf"])
    df["ideo_ches2_combined"] = df["ideo_ches2"].combine_first(df["ideo_ches2_pf"])

    elff_res = df.apply(
        lambda r: get_elff_ideology(
            elff,
            r.get("elff_country_norm"),
            r.get("party1_norm"),
            int(r.get("year")) if pd.notna(r.get("year")) else 0,
        ),
        axis=1,
    )
    df["ideo_elff1"] = [x[0] for x in elff_res]
    df["elff_year_diff1"] = [x[1] for x in elff_res]

    elff_res2 = df.apply(
        lambda r: get_elff_ideology(
            elff,
            r.get("elff_country_norm"),
            r.get("party2_norm"),
            int(r.get("year")) if pd.notna(r.get("year")) else 0,
        ),
        axis=1,
    )
    df["ideo_elff2"] = [x[0] for x in elff_res2]
    df["elff_year_diff2"] = [x[1] for x in elff_res2]

    parlgov_res = df.apply(
        lambda r: get_parlgov_ideology(parlgov, r.get("parlgov_country_norm"), r.get("party1_norm")), axis=1
    )
    df["ideo_parlgov_state1"] = [x[0] for x in parlgov_res]
    df["ideo_parlgov_lr1"] = [x[1] for x in parlgov_res]

    parlgov_res2 = df.apply(
        lambda r: get_parlgov_ideology(parlgov, r.get("parlgov_country_norm"), r.get("party2_norm")), axis=1
    )
    df["ideo_parlgov_state2"] = [x[0] for x in parlgov_res2]
    df["ideo_parlgov_lr2"] = [x[1] for x in parlgov_res2]

    df["ideo_ches1_r"] = rescale(df["ideo_ches1_combined"], 0, 6)
    df["ideo_ches2_r"] = rescale(df["ideo_ches2_combined"], 0, 6)
    df["ideo_elff1_r"] = rescale(df["ideo_elff1"], 0, 6)
    df["ideo_elff2_r"] = rescale(df["ideo_elff2"], 0, 6)
    df["ideo_parlgov1_r"] = rescale(df["ideo_parlgov_state1"], 0, 6)
    df["ideo_parlgov2_r"] = rescale(df["ideo_parlgov_state2"], 0, 6)

    def choose_ideo(vparty_val, parlgov_val, ches_val, elff_val):
        if pd.notna(vparty_val):
            return vparty_val, "vparty"
        if pd.notna(parlgov_val):
            return parlgov_val, "parlgov"
        if pd.notna(ches_val):
            return ches_val, "ches"
        if pd.notna(elff_val):
            return elff_val, "elff"
        return np.nan, "missing"

    final1 = df.apply(
        lambda r: choose_ideo(r.get("ideo_party1"), r.get("ideo_parlgov1_r"), r.get("ideo_ches1_r"), r.get("ideo_elff1_r")),
        axis=1,
    )
    df["ideo_final1"] = [x[0] for x in final1]
    df["ideo_source1"] = [x[1] for x in final1]

    final2 = df.apply(
        lambda r: choose_ideo(r.get("ideo_party2"), r.get("ideo_parlgov2_r"), r.get("ideo_ches2_r"), r.get("ideo_elff2_r")),
        axis=1,
    )
    df["ideo_final2"] = [x[0] for x in final2]
    df["ideo_source2"] = [x[1] for x in final2]

    df["margin_market_ext"] = np.where(
        df["ideo_final1"].notna() & df["ideo_final2"].notna(),
        np.where(df["ideo_final1"] >= df["ideo_final2"], df["margin"], -df["margin"]),
        np.nan,
    )
    df["D_market_win_ext"] = np.where(df["margin_market_ext"].notna(), (df["margin_market_ext"] >= 0).astype(int), np.nan)

    df = attach_des(df, des, country_map)

    idea_cols = [
        "idea_system_family",
        "idea_leg_system",
        "idea_pres_system",
        "idea_tiers",
        "idea_leg_size_direct",
        "idea_leg_size_voting",
    ]
    df = merge_with_diagnostics(
        df,
        idea[["iso3", "year"] + idea_cols],
        keys=["iso3", "year"],
        how="left",
        name="idea_merge",
        logger=logger,
    )
    df["idea_match"] = df["idea_system_family"].notna()

    if "dpi_country_norm" not in df.columns:
        df = merge_with_diagnostics(
            df,
            country_map[["iso3", "dpi_country_norm"]],
            keys=["iso3"],
            how="left",
            name="dpi_country_map",
            logger=logger,
        )

    dpi_subset = dpi.copy()
    dpi_subset = dpi_subset.rename(columns={
        "system": "dpi_system",
        "execme": "dpi_execme",
        "execrlc": "dpi_execrlc",
        "gov1rlc": "dpi_gov1rlc",
        "checks": "dpi_checks",
        "checks_lax": "dpi_checks_lax",
        "military": "dpi_military",
    })
    dpi_keep = [
        "country_norm",
        "year",
        "dpi_system",
        "dpi_execme",
        "dpi_execrlc",
        "dpi_gov1rlc",
        "dpi_checks",
        "dpi_checks_lax",
        "dpi_military",
    ]
    dpi_keep = [c for c in dpi_keep if c in dpi_subset.columns]
    dpi_subset = dpi_subset[dpi_keep]

    df = merge_with_diagnostics(
        df,
        dpi_subset,
        keys=None,
        left_on=["dpi_country_norm", "year"],
        right_on=["country_norm", "year"],
        how="left",
        name="dpi_merge",
        logger=logger,
    )
    df["dpi_match"] = df["dpi_system"].notna()

    out = paths.processed / "elections_master_external.parquet"
    df.to_parquet(out, index=False)

    validate_required_columns(
        df,
        ["iso3", "year", "margin_market_ext", "ideo_final1", "ideo_final2"],
        "elections_master_external",
        strict,
        logger,
    )

    return df


def build_combo_metrics(df: pd.DataFrame) -> pd.DataFrame:
    combos = []
    combos.append({
        "combo": "vparty_only",
        "margin_share": df["margin_market"].notna().mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "margin_market"].notna().mean(),
    })
    combos.append({
        "combo": "external_priority",
        "margin_share": df["margin_market_ext"].notna().mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "margin_market_ext"].notna().mean(),
    })
    combos.append({
        "combo": "parlgov_only",
        "margin_share": df["ideo_parlgov1_r"].notna().mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "ideo_parlgov1_r"].notna().mean(),
    })
    combos.append({
        "combo": "ches_only",
        "margin_share": df["ideo_ches1_r"].notna().mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "ideo_ches1_r"].notna().mean(),
    })
    combos.append({
        "combo": "elff_only",
        "margin_share": df["ideo_elff1_r"].notna().mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "ideo_elff1_r"].notna().mean(),
    })
    combos.append({
        "combo": "des_match_2y",
        "margin_share": df["des_match_2y"].mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "des_match_2y"].mean(),
    })
    combos.append({
        "combo": "des_match_5y",
        "margin_share": df["des_match_5y"].mean(),
        "clean_margin_share": df.loc[df["clean_flag"], "des_match_5y"].mean(),
    })
    return pd.DataFrame(combos)


def build_final(paths: PipelinePaths, logger: logging.Logger) -> pd.DataFrame:
    ext_path = paths.processed / "elections_master_external.parquet"
    if ext_path.exists():
        df = pd.read_parquet(ext_path)
    else:
        df = pd.read_parquet(paths.processed / "elections_master.parquet")

    if "margin_market_ext" in df.columns:
        df["margin_market_best"] = df["margin_market_ext"].combine_first(df.get("margin_market"))
    else:
        df["margin_market_best"] = df.get("margin_market")

    df["D_market_win_best"] = np.where(
        df["margin_market_best"].notna(), (df["margin_market_best"] >= 0).astype(int), np.nan
    )

    df["analysis_ready"] = (
        df.get("clean_flag", True)
        & df["iso3"].notna()
        & df["year"].notna()
        & df["share_1"].notna()
        & df["share_2"].notna()
        & df["margin_market_best"].notna()
    )
    if "sample_competitive" in df.columns:
        df["analysis_ready"] = df["analysis_ready"] & df["sample_competitive"].fillna(True)

    out_all = paths.processed / "elections_final.parquet"
    df.to_parquet(out_all, index=False)

    summary = pd.DataFrame([
        {
            "rows_all": len(df),
            "rows_final": int(df["analysis_ready"].sum()),
            "countries_all": df["iso3"].nunique(),
            "countries_final": df.loc[df["analysis_ready"], "iso3"].nunique(),
            "margin_best_share": df["margin_market_best"].notna().mean(),
            "analysis_ready_share": df["analysis_ready"].mean(),
        }
    ])
    summary.to_csv(paths.reports / "elections_final_summary.csv", index=False)

    validate_required_columns(
        df,
        ["iso3", "year", "analysis_ready", "margin_market_best"],
        "elections_final",
        False,
        logger,
    )

    return df


def fetch_electionguide(paths: PipelinePaths, logger: logging.Logger) -> None:
    import json
    import sys

    config_path = paths.config / "electionguide_api.json"
    if not config_path.exists():
        logger.error("Missing config/electionguide_api.json. Copy config/electionguide_api.example.json and fill it.")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_url = cfg.get("base_url")
    endpoint = cfg.get("endpoint")
    token = cfg.get("token")
    if not base_url or not endpoint or not token:
        logger.error("Config missing base_url, endpoint, or token.")
        sys.exit(1)

    auth_header = cfg.get("auth_header", "Authorization")
    auth_scheme = cfg.get("auth_scheme", "Bearer")
    params = cfg.get("params", {})
    output_path = cfg.get("output_path", "data/raw/electionguide/elections.json")

    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    headers = {auth_header: f"{auth_scheme} {token}"}

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()

    out_path = paths.root / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(resp.json(), f, ensure_ascii=False, indent=2)
    logger.info("Saved ElectionGuide data to %s", out_path)


def fetch_wdi(paths: PipelinePaths, logger: logging.Logger) -> None:
    raw = paths.raw / "wdi"
    raw.mkdir(parents=True, exist_ok=True)
    indicators = {
        "NY.GDP.PCAP.KD": "gdp_pc_const2015",
        "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp_const2017",
        "FP.CPI.TOTL.ZG": "inflation_cpi",
        "NE.GDI.FTOT.ZS": "investment_share_gdp",
        "NE.TRD.GNFS.ZS": "trade_share_gdp",
    }

    efw = pd.read_excel(paths.raw / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx", sheet_name="EFW Panel Dataset")
    iso_list = sorted(set(efw["ISO_Code"].dropna().unique()))

    def fetch_indicator(indicator: str, countries: list[str]) -> pd.DataFrame:
        rows = []
        for iso in countries:
            url = f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"
            params = {"format": "json", "per_page": 20000}
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data, list) or len(data) < 2:
                continue
            series = data[1]
            for item in series:
                if item.get("value") is None:
                    continue
                rows.append({
                    "iso3": iso,
                    "year": int(item["date"]),
                    "indicator": indicator,
                    "value": item["value"],
                })
            time.sleep(0.1)
        return pd.DataFrame(rows)

    all_frames = []
    for ind in indicators:
        df = fetch_indicator(ind, iso_list)
        all_frames.append(df)
        df.to_parquet(raw / f"wdi_{ind}.parquet", index=False)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined.to_parquet(raw / "wdi_selected.parquet", index=False)
    logger.info("WDI data written to %s", raw)
