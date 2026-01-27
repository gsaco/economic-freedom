from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import shutil
import unicodedata
import zipfile
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


try:  # pragma: no cover - optional dependency
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    raw: Path = field(init=False)
    interim: Path = field(init=False)
    processed: Path = field(init=False)
    reports: Path = field(init=False)
    external: Path = field(init=False)
    config: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.root / "data" / "raw")
        object.__setattr__(self, "interim", self.root / "data" / "interim")
        object.__setattr__(self, "processed", self.root / "data" / "processed")
        object.__setattr__(self, "reports", self.root / "reports")
        object.__setattr__(self, "external", self.root / "data" / "external")
        object.__setattr__(self, "config", self.root / "config")

    def ensure_dirs(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.interim.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.external.mkdir(parents=True, exist_ok=True)


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("elections_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_whitespace_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", "and")
    text = _non_alnum_re.sub(" ", text)
    text = _whitespace_re.sub(" ", text).strip()
    return text


def best_fuzzy_match(query: str, choices: Iterable[str], min_score: int = 80) -> Tuple[Optional[str], int]:
    q = normalize_name(query)
    if not q:
        return None, 0
    choices_list = list(choices)
    if not choices_list:
        return None, 0
    if process and fuzz:
        match, score, _ = process.extractOne(q, choices_list, scorer=fuzz.token_sort_ratio)
        return (match, score) if score >= min_score else (None, score)
    import difflib

    best = None
    best_score = 0
    for c in choices_list:
        score = int(difflib.SequenceMatcher(None, q, c).ratio() * 100)
        if score > best_score:
            best_score = score
            best = c
    return (best, best_score) if best_score >= min_score else (None, best_score)


def _handle_violation(message: str, strict: bool, logger: logging.Logger) -> None:
    if strict:
        raise ValueError(message)
    logger.warning(message)


def validate_required_columns(df: pd.DataFrame, columns: Sequence[str], name: str, strict: bool, logger: logging.Logger) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _handle_violation(f"{name}: missing required columns {missing}", strict, logger)


def validate_unique(df: pd.DataFrame, keys: Sequence[str], name: str, strict: bool, logger: logging.Logger) -> None:
    if not keys:
        return
    dupes = df.duplicated(list(keys), keep=False)
    if dupes.any():
        count = int(dupes.sum())
        _handle_violation(f"{name}: found {count} duplicate rows on keys {list(keys)}", strict, logger)


def validate_non_null_share(df: pd.DataFrame, columns: Sequence[str], name: str, strict: bool, logger: logging.Logger) -> None:
    missing = df[list(columns)].isna().any(axis=1)
    if missing.any():
        count = int(missing.sum())
        _handle_violation(f"{name}: {count} rows missing required keys {list(columns)}", strict, logger)


def log_key_stats(df: pd.DataFrame, keys: Sequence[str], name: str, logger: logging.Logger) -> None:
    keys = list(keys)
    if not keys:
        return
    missing = df[keys].isna().any(axis=1).mean() if not df.empty else 0
    dupes = df.duplicated(keys, keep=False).sum() if not df.empty else 0
    logger.info(
        "%s: rows=%s unique_keys=%s missing_key_share=%.3f dup_rows=%s",
        name,
        len(df),
        df[keys].dropna().drop_duplicates().shape[0] if not df.empty else 0,
        float(missing),
        int(dupes),
    )
    if dupes:
        logger.warning("%s: %s duplicate rows on keys %s", name, int(dupes), keys)


def harmonize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype("string").str.upper()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    return df


def dedupe_by_keys(
    df: pd.DataFrame,
    keys: Sequence[str],
    logger: Optional[logging.Logger] = None,
    name: str = "",
) -> pd.DataFrame:
    keys = list(keys)
    if df.empty:
        return df
    dup_mask = df.duplicated(keys, keep=False)
    if not dup_mask.any():
        return df
    df = df.copy()
    df["_non_null"] = df.notna().sum(axis=1)
    sort_cols = keys + ["_non_null"]
    ascending = [True] * len(keys) + [False]
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    deduped = df.drop_duplicates(keys, keep="first").drop(columns=["_non_null"])
    if logger:
        logger.warning(
            "%s: deduped %s -> %s rows on keys %s",
            name,
            len(df),
            len(deduped),
            keys,
        )
    return deduped


def merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Optional[Sequence[str]],
    how: str,
    name: str,
    logger: logging.Logger,
    dedupe_right: bool = True,
    suffixes: Tuple[str, str] = ("", "_y"),
    left_on: Optional[Sequence[str]] = None,
    right_on: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if keys is not None:
        left_on = list(keys)
        right_on = list(keys)
    if left_on is None or right_on is None:
        raise ValueError("merge_with_diagnostics requires keys or both left_on and right_on")
    left_on = list(left_on)
    right_on = list(right_on)
    if dedupe_right:
        right = dedupe_by_keys(right, right_on, logger=logger, name=f"{name}:right")
    log_key_stats(left, left_on, f"{name}:left", logger)
    log_key_stats(right, right_on, f"{name}:right", logger)
    merged = left.merge(
        right,
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        indicator="_merge_flag",
    )
    if not merged.empty:
        match_rate = (merged["_merge_flag"] == "both").mean()
    else:
        match_rate = 0
    logger.info("%s: merge match_rate=%.3f", name, float(match_rate))
    return merged.drop(columns=["_merge_flag"])


MANUAL_COUNTRY_OVERRIDES = {
    "bahamas": "Bahamas, The",
    "cape verde": "Cabo Verde",
    "cote divoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire",
    "czech republic": "Czechia",
    "gambia": "Gambia, The",
    "korea south": "Korea, Rep.",
    "korea, south": "Korea, Rep.",
    "north korea": "Korea, Dem. Rep.",
    "korea, north": "Korea, Dem. Rep.",
    "russia": "Russian Federation",
    "slovak republic": "Slovakia",
    "syria": "Syrian Arab Republic",
    "turkey": "Turkiye",
    "venezuela": "Venezuela, RB",
    "yemen": "Yemen, Rep.",
    "congo": "Congo, Rep.",
    "congo brazzaville": "Congo, Rep.",
    "congo kinshasa": "Congo, Dem. Rep.",
    "congo (drc)": "Congo, Dem. Rep.",
    "congo drc": "Congo, Dem. Rep.",
    "congo, democratic republic of": "Congo, Dem. Rep.",
    "congo, republic of": "Congo, Rep.",
    "egypt": "Egypt, Arab Rep.",
    "kyrgyzstan": "Kyrgyz Republic",
    "slovakia": "Slovak Republic",
    "brunei": "Brunei Darussalam",
    "hong kong": "Hong Kong SAR, China",
    "saudi arabia": "Saudi Arabia",
    "iran": "Iran, Islamic Rep.",
    "laos": "Lao PDR",
    "micronesia": "Micronesia, Fed. Sts.",
    "st. kitts and nevis": "St. Kitts and Nevis",
    "st. lucia": "St. Lucia",
    "st. vincent and the grenadines": "St. Vincent and the Grenadines",
    "trinidad & tobago": "Trinidad and Tobago",
    "united states": "United States",
    "united kingdom": "United Kingdom",
    "vietnam": "Viet Nam",
    "bolivia": "Bolivia",
    "bolivia (plurinational state of)": "Bolivia",
    "tanzania": "Tanzania",
}

STOPWORDS = {
    "party",
    "movement",
    "front",
    "union",
    "national",
    "democratic",
    "republican",
    "social",
    "labour",
    "labor",
    "liberal",
    "conservative",
    "progressive",
    "people",
    "peoples",
    "workers",
    "christian",
    "islamic",
    "new",
    "old",
    "green",
    "left",
    "right",
    "center",
    "centrist",
    "state",
    "country",
}

MERGE_STRATEGIES = (
    "date_exact",
    "date_fallback",
    "year_month",
    "year",
)
MERGE_SCORE_EPS = 0.002


def extract_raw(paths: PipelinePaths, logger: logging.Logger) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    zip_tasks = [
        ("CPD_V-Party_CSV_v2.zip", paths.raw / "vparty"),
        ("Coder_Level_V-Party_CSV_v2.zip", paths.raw / "vparty"),
        ("NELDA 6.0.zip", paths.raw / "nelda"),
        ("clea_lc_20251015.sav_.zip", paths.raw / "clea"),
        ("clea_uc_20220111_spss.zip", paths.raw / "clea"),
    ]
    copy_tasks = [
        ("presidential_elections_v2.dta", paths.raw / "ned"),
        ("parliamentary_elections_v2.dta", paths.raw / "ned"),
        ("efotw-2025-master-index-data-for-researchers-iso.xlsx", paths.raw / "efw"),
        ("ETAD_v_1_0_0.csv", paths.raw / "etad"),
    ]

    for zip_name, dest in zip_tasks:
        zip_path = paths.root / zip_name
        if not zip_path.exists():
            logger.warning("Missing zip: %s", zip_path)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        logger.info("Extracted %s -> %s", zip_path, dest)

    for fname, dest in copy_tasks:
        src = paths.root / fname
        if not src.exists():
            logger.warning("Missing file: %s", src)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)
        logger.info("Copied %s -> %s", src, dest / src.name)


def convert_clea_to_parquet(paths: PipelinePaths, logger: logging.Logger) -> None:
    import pyreadstat

    raw = paths.raw / "clea"
    interim = paths.interim / "clea"
    interim.mkdir(parents=True, exist_ok=True)
    sav_files = [p for p in raw.rglob("*.sav") if not p.name.startswith("._")]
    if not sav_files:
        raise FileNotFoundError("No .sav files found in data/raw/clea. Run extract_raw first.")

    for sav in sav_files:
        logger.info("Reading %s", sav)
        df, _ = pyreadstat.read_sav(sav)
        df.columns = [c.lower() for c in df.columns]
        for col in ["yr", "mn"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        out_path = interim / f"{sav.stem}.parquet"
        logger.info("Writing %s", out_path)
        df.to_parquet(out_path, index=False)


def load_efw_panel(paths: PipelinePaths) -> pd.DataFrame:
    path = paths.raw / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx"
    df = pd.read_excel(path, sheet_name="EFW Panel Dataset")
    df = df.rename(columns={"ISO_Code": "iso3", "Countries": "country_name", "Year": "year"})
    keep = ["iso3", "country_name", "year", "Summary", "Area 1", "Area 2", "Area 3", "Area 4", "Area 5"]
    return df[keep]


def load_efw_iso(paths: PipelinePaths) -> set[str]:
    df = load_efw_panel(paths)
    return set(df["iso3"].dropna().astype(str).str.upper().unique())


def load_vparty(paths: PipelinePaths) -> pd.DataFrame:
    path = paths.raw / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv"
    df = pd.read_csv(path)
    keep = [
        "v2paid",
        "country_name",
        "COWcode",
        "year",
        "v2paenname",
        "v2pashname",
        "v2paorname",
        "v2pariglef",
    ]
    return df[keep]


def load_nelda(paths: PipelinePaths) -> pd.DataFrame:
    import pyreadstat

    nelda_path = paths.raw / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta"
    nelda, _ = pyreadstat.read_dta(nelda_path, encoding="latin1")
    nelda = nelda.copy()
    nelda["office_type"] = nelda["types"].map({
        "Executive": "presidential",
        "Legislative/Parliamentary": "parliamentary",
    })
    nelda = nelda.dropna(subset=["office_type"])
    nelda["mmdd"] = nelda["mmdd"].astype(str).str.zfill(4)
    nelda["nelda_date"] = pd.to_datetime(
        nelda["year"].astype(str) + nelda["mmdd"], errors="coerce", format="%Y%m%d"
    )
    return nelda


def build_country_crosswalk(vparty: pd.DataFrame, efw_countries: pd.DataFrame) -> pd.DataFrame:
    efw_names = efw_countries.dropna(subset=["country_name"]).copy()
    efw_names["country_name_norm"] = efw_names["country_name"].map(normalize_name)
    efw_name_to_iso = dict(zip(efw_names["country_name_norm"], efw_names["iso3"]))

    rows = []
    for cowcode, grp in vparty.groupby("COWcode"):
        country_name = grp["country_name"].dropna().iloc[0]
        norm = normalize_name(country_name)
        override = MANUAL_COUNTRY_OVERRIDES.get(norm)
        if override:
            override_norm = normalize_name(override)
            iso = efw_names.loc[efw_names["country_name_norm"] == override_norm, "iso3"]
            if not iso.empty:
                rows.append({
                    "cowcode": cowcode,
                    "vparty_country_name": country_name,
                    "efw_country_name": override,
                    "iso3": iso.iloc[0],
                    "match_score": 100,
                    "match_method": "manual_override",
                })
                continue
        match, score = best_fuzzy_match(country_name, efw_names["country_name_norm"].tolist(), min_score=70)
        if match:
            rows.append({
                "cowcode": cowcode,
                "vparty_country_name": country_name,
                "efw_country_name": efw_names.loc[efw_names["country_name_norm"] == match, "country_name"].iloc[0],
                "iso3": efw_name_to_iso.get(match),
                "match_score": score,
                "match_method": "fuzzy",
            })
        else:
            rows.append({
                "cowcode": cowcode,
                "vparty_country_name": country_name,
                "efw_country_name": None,
                "iso3": None,
                "match_score": score,
                "match_method": "unmatched",
            })
    return pd.DataFrame(rows)


def map_country_names_to_iso(country_names: pd.Series, efw_countries: pd.DataFrame) -> pd.DataFrame:
    efw_names = efw_countries.dropna(subset=["country_name"]).copy()
    efw_names["country_name_norm"] = efw_names["country_name"].map(normalize_name)
    efw_name_to_iso = dict(zip(efw_names["country_name_norm"], efw_names["iso3"]))

    rows = []
    for name in country_names.dropna().unique():
        norm = normalize_name(name)
        override = MANUAL_COUNTRY_OVERRIDES.get(norm)
        if override:
            override_norm = normalize_name(override)
            iso = efw_names.loc[efw_names["country_name_norm"] == override_norm, "iso3"]
            rows.append({
                "country": name,
                "efw_country_name": override,
                "iso3": iso.iloc[0] if not iso.empty else None,
                "match_score": 100,
                "match_method": "manual_override",
            })
            continue
        match, score = best_fuzzy_match(name, efw_names["country_name_norm"].tolist(), min_score=70)
        if match:
            rows.append({
                "country": name,
                "efw_country_name": efw_names.loc[efw_names["country_name_norm"] == match, "country_name"].iloc[0],
                "iso3": efw_name_to_iso.get(match),
                "match_score": score,
                "match_method": "fuzzy",
            })
        else:
            rows.append({
                "country": name,
                "efw_country_name": None,
                "iso3": None,
                "match_score": score,
                "match_method": "unmatched",
            })
    return pd.DataFrame(rows)


def prep_party_name_index(vparty: pd.DataFrame) -> pd.DataFrame:
    name_cols = ["v2paenname", "v2pashname", "v2paorname"]
    records = []
    for _, row in vparty.iterrows():
        for col in name_cols:
            name = row[col]
            if pd.isna(name):
                continue
            records.append({
                "COWcode": row["COWcode"],
                "v2paid": row["v2paid"],
                "year": row["year"],
                "party_name": name,
                "party_name_norm": normalize_name(name),
            })
    return pd.DataFrame(records).drop_duplicates()


def match_party_name_basic(party_name: str, cowcode: float, party_index: pd.DataFrame) -> tuple:
    if pd.isna(party_name) or pd.isna(cowcode):
        return (None, 0)
    subset = party_index.loc[party_index["COWcode"] == cowcode, "party_name_norm"].unique().tolist()
    match, score = best_fuzzy_match(party_name, subset, min_score=80)
    if not match:
        return (None, score)
    v2paid = party_index.loc[
        (party_index["COWcode"] == cowcode) & (party_index["party_name_norm"] == match), "v2paid"
    ].mode()
    return (v2paid.iloc[0] if not v2paid.empty else None, score)


def get_party_ideology_basic(vparty: pd.DataFrame, v2paid, year) -> float:
    if v2paid is None or pd.isna(year):
        return np.nan
    subset = vparty.loc[vparty["v2paid"] == v2paid, ["year", "v2pariglef"]].dropna()
    if subset.empty:
        return np.nan
    subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
    subset = subset.dropna(subset=["year"])
    subset["year"] = subset["year"].astype(int)
    year = int(year)
    subset["year_diff"] = (subset["year"] - year).abs()
    closest = subset.sort_values(["year_diff"]).iloc[0]
    if closest["year_diff"] > 4:
        return np.nan
    return float(closest["v2pariglef"])


def compute_final_vote_share(df: pd.DataFrame, prefix: str) -> pd.Series:
    col1 = f"vote_share1_{prefix}"
    col2 = f"vote_share2_{prefix}"
    share1 = pd.to_numeric(df.get(col1), errors="coerce")
    share2 = pd.to_numeric(df.get(col2), errors="coerce")
    return share2.where(~share2.isna(), share1)


def build_presidential(ned: pd.DataFrame) -> pd.DataFrame:
    df = ned.copy()
    df["office_type"] = "presidential"
    df["share_1"] = compute_final_vote_share(df, "1")
    df["share_2"] = compute_final_vote_share(df, "2")
    df["margin"] = df["share_1"] - df["share_2"]
    df = df.rename(
        columns={
            "party_1": "party_1_name",
            "party_2": "party_2_name",
            "candidate_1": "candidate_1_name",
            "candidate_2": "candidate_2_name",
        }
    )
    return df


def build_parliamentary(ned: pd.DataFrame) -> pd.DataFrame:
    df = ned.copy()
    df["office_type"] = "parliamentary"
    df["share_1"] = pd.to_numeric(df.get("seat_share_1"), errors="coerce")
    df["share_2"] = pd.to_numeric(df.get("seat_share_2"), errors="coerce")
    df.loc[df["share_1"] < 0, "share_1"] = np.nan
    df.loc[df["share_2"] < 0, "share_2"] = np.nan
    df["margin"] = df["share_1"] - df["share_2"]
    df = df.rename(columns={"party_1": "party_1_name", "party_2": "party_2_name"})
    return df


def add_nelda_flags(df: pd.DataFrame, nelda: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        nelda[["ccode", "year", "office_type", "nelda_date", "nelda3", "nelda4", "nelda5"]],
        how="left",
        left_on=["country_cow", "year", "office_type"],
        right_on=["ccode", "year", "office_type"],
        suffixes=("", "_nelda"),
    )

    key_cols = ["country", "year", "office_type", "date"] if "country" in merged.columns else ["iso3", "year", "office_type", "date"]
    if all(col in merged.columns for col in key_cols) and merged.duplicated(subset=key_cols).any():
        def resolve(group: pd.DataFrame) -> pd.DataFrame:
            if group.shape[0] == 1:
                return group
            group = group.copy()
            group["date"] = pd.to_datetime(group["date"], errors="coerce")
            if group["date"].notna().any():
                group["date_diff"] = (group["nelda_date"] - group["date"]).abs()
                return group.sort_values("date_diff").head(1)
            return group.head(1)

        merged = merged.groupby(key_cols, dropna=False, as_index=False).apply(resolve).reset_index(drop=True)

    for col in ["nelda3", "nelda4", "nelda5"]:
        if col in merged.columns:
            merged[col] = merged[col].replace({"yes": 1, "no": 0, "unclear": np.nan})
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def build_elections_ned(paths: PipelinePaths, logger: logging.Logger, strict: bool = False) -> pd.DataFrame:
    paths.ensure_dirs()
    efw = load_efw_panel(paths)
    vparty = load_vparty(paths)

    crosswalk = build_country_crosswalk(vparty, efw[["iso3", "country_name"]].drop_duplicates())
    crosswalk.to_csv(paths.interim / "country_crosswalk_cow_iso.csv", index=False)

    party_index = prep_party_name_index(vparty)

    pres = pd.read_stata(paths.raw / "ned" / "presidential_elections_v2.dta", convert_categoricals=False)
    parl = pd.read_stata(paths.raw / "ned" / "parliamentary_elections_v2.dta", convert_categoricals=False)

    pres = build_presidential(pres)
    parl = build_parliamentary(parl)

    combined = pd.concat([pres, parl], ignore_index=True)

    combined = merge_with_diagnostics(
        combined,
        crosswalk[["cowcode", "iso3"]],
        keys=None,
        left_on=["country_cow"],
        right_on=["cowcode"],
        how="left",
        name="ned_iso_cowcode",
        logger=logger,
    )

    name_map = map_country_names_to_iso(combined["country"], efw[["iso3", "country_name"]].drop_duplicates())
    name_map.to_csv(paths.interim / "country_crosswalk_ned_name_iso.csv", index=False)

    combined = merge_with_diagnostics(
        combined,
        name_map[["country", "iso3"]].rename(columns={"iso3": "iso3_name"}),
        keys=["country"],
        how="left",
        name="ned_iso_name",
        logger=logger,
        dedupe_right=True,
        suffixes=("", "_name"),
    )
    combined["iso3"] = combined["iso3"].combine_first(combined["iso3_name"])
    combined = harmonize_keys(combined)

    matched_party1 = combined.apply(
        lambda r: match_party_name_basic(r.get("party_1_name"), r.get("country_cow"), party_index), axis=1
    )
    combined["party1_v2paid"] = [m[0] for m in matched_party1]
    combined["party1_match_score"] = [m[1] for m in matched_party1]

    matched_party2 = combined.apply(
        lambda r: match_party_name_basic(r.get("party_2_name"), r.get("country_cow"), party_index), axis=1
    )
    combined["party2_v2paid"] = [m[0] for m in matched_party2]
    combined["party2_match_score"] = [m[1] for m in matched_party2]

    combined["ideo_party1"] = combined.apply(
        lambda r: get_party_ideology_basic(vparty, r.get("party1_v2paid"), r.get("year")), axis=1
    )
    combined["ideo_party2"] = combined.apply(
        lambda r: get_party_ideology_basic(vparty, r.get("party2_v2paid"), r.get("year")), axis=1
    )

    combined["margin_market"] = np.where(
        combined["ideo_party1"] >= combined["ideo_party2"],
        combined["share_1"] - combined["share_2"],
        combined["share_2"] - combined["share_1"],
    )
    combined["D_market_win"] = np.where(
        combined["margin_market"].notna(), (combined["margin_market"] >= 0).astype(int), np.nan
    )

    nelda_path = paths.raw / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta"
    if nelda_path.exists():
        nelda = load_nelda(paths)
        combined = add_nelda_flags(combined, nelda)
    else:
        logger.warning("NELDA file not found: %s", nelda_path)

    out_path = paths.processed / "elections_ned.parquet"
    combined.to_parquet(out_path, index=False)

    efw_countries = efw["iso3"].dropna().unique()
    covered = combined["iso3"].dropna().unique()
    coverage_rate = len(set(covered)) / len(set(efw_countries)) if efw_countries.size else 0
    report = pd.DataFrame({
        "metric": ["efw_countries", "covered_by_ned", "coverage_rate"],
        "value": [len(set(efw_countries)), len(set(covered)), coverage_rate],
    })
    report.to_csv(paths.interim / "coverage_ned.csv", index=False)

    validate_required_columns(
        combined,
        ["country", "year", "iso3", "share_1", "share_2", "party_1_name", "party_2_name"],
        "elections_ned",
        strict,
        logger,
    )

    return combined


def map_clea_countries_to_iso(clea_countries: pd.Series, efw_countries: pd.DataFrame) -> pd.DataFrame:
    efw_names = efw_countries.dropna(subset=["country_name"]).copy()
    efw_names["country_name_norm"] = efw_names["country_name"].map(normalize_name)
    efw_name_to_iso = dict(zip(efw_names["country_name_norm"], efw_names["iso3"]))

    rows = []
    for name in clea_countries.dropna().unique():
        norm = normalize_name(name)
        override = MANUAL_COUNTRY_OVERRIDES.get(norm)
        if override:
            override_norm = normalize_name(override)
            iso = efw_names.loc[efw_names["country_name_norm"] == override_norm, "iso3"]
            rows.append({
                "clea_country": name,
                "efw_country_name": override,
                "iso3": iso.iloc[0] if not iso.empty else None,
                "match_score": 100,
                "match_method": "manual_override",
            })
            continue
        match, score = best_fuzzy_match(name, efw_names["country_name_norm"].tolist(), min_score=70)
        if match:
            rows.append({
                "clea_country": name,
                "efw_country_name": efw_names.loc[efw_names["country_name_norm"] == match, "country_name"].iloc[0],
                "iso3": efw_name_to_iso.get(match),
                "match_score": score,
                "match_method": "fuzzy",
            })
        else:
            rows.append({
                "clea_country": name,
                "efw_country_name": None,
                "iso3": None,
                "match_score": score,
                "match_method": "unmatched",
            })
    return pd.DataFrame(rows)


def aggregate_clea_lower(paths: PipelinePaths) -> pd.DataFrame:
    path = paths.interim / "clea" / "clea_lc_20251015.parquet"
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    for col in ["pv1", "seat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    party_totals = (
        df.groupby(["id", "ctr", "ctr_n", "yr", "mn", "pty_n"], dropna=False)
        .agg({"pv1": "sum", "seat": "sum"})
        .reset_index()
    )
    totals = (
        party_totals.groupby(["id", "ctr", "ctr_n", "yr", "mn"], dropna=False)
        .agg({"pv1": "sum", "seat": "sum"})
        .reset_index()
        .rename(columns={"pv1": "total_votes", "seat": "total_seats"})
    )

    merged = party_totals.merge(totals, on=["id", "ctr", "ctr_n", "yr", "mn"], how="left")
    merged["vote_share"] = merged["pv1"] / merged["total_votes"]
    merged["seat_share"] = merged["seat"] / merged["total_seats"]

    merged["vote_share_rank"] = merged.groupby(["id"])["vote_share"].rank(method="first", ascending=False)
    merged["seat_share_rank"] = merged.groupby(["id"])["seat_share"].rank(method="first", ascending=False)

    top_by_vote = merged[merged["vote_share_rank"] <= 2].copy()
    top_by_seat = merged[merged["seat_share_rank"] <= 2].copy()

    def pivot_top(df_top: pd.DataFrame, share_col: str, suffix: str) -> pd.DataFrame:
        df_top = df_top.sort_values(["id", share_col], ascending=[True, False], kind="mergesort")
        df_top["rank"] = df_top.groupby("id")[share_col].rank(method="first", ascending=False)
        df_top = df_top[df_top["rank"] <= 2]
        df_top = df_top.pivot_table(
            index=["id", "ctr", "ctr_n", "yr", "mn"],
            columns="rank",
            values=["pty_n", share_col],
            aggfunc="first",
        )
        df_top.columns = [f"{col[0]}_{int(col[1])}_{suffix}" for col in df_top.columns]
        return df_top.reset_index()

    vote_wide = pivot_top(top_by_vote, "vote_share", "vote")
    seat_wide = pivot_top(top_by_seat, "seat_share", "seat")

    elections = vote_wide.merge(seat_wide, on=["id", "ctr", "ctr_n", "yr", "mn"], how="outer")
    elections["office_type"] = "parliamentary"

    elections["share_1"] = elections["vote_share_1_vote"].combine_first(elections["seat_share_1_seat"])
    elections["share_2"] = elections["vote_share_2_vote"].combine_first(elections["seat_share_2_seat"])
    elections["party_1_name"] = elections["pty_n_1_vote"].combine_first(elections["pty_n_1_seat"])
    elections["party_2_name"] = elections["pty_n_2_vote"].combine_first(elections["pty_n_2_seat"])
    elections["margin"] = elections["share_1"] - elections["share_2"]

    elections["year"] = elections["yr"].astype("Int64")
    elections["month"] = elections["mn"].astype("Int64")
    elections["date"] = pd.to_datetime(
        elections["year"].astype(str) + "-" + elections["month"].astype(str) + "-01",
        errors="coerce",
    )

    return elections


def build_elections_clea(paths: PipelinePaths, logger: logging.Logger, strict: bool = False) -> pd.DataFrame:
    paths.ensure_dirs()
    efw = load_efw_panel(paths)
    vparty = load_vparty(paths)
    party_index = prep_party_name_index(vparty)

    clea_elections = aggregate_clea_lower(paths)

    country_map = map_clea_countries_to_iso(clea_elections["ctr_n"], efw[["iso3", "country_name"]].drop_duplicates())
    country_map.to_csv(paths.interim / "country_crosswalk_clea_iso.csv", index=False)

    clea_elections = merge_with_diagnostics(
        clea_elections,
        country_map[["clea_country", "iso3"]],
        keys=None,
        left_on=["ctr_n"],
        right_on=["clea_country"],
        how="left",
        name="clea_iso_name",
        logger=logger,
        suffixes=("", "_map"),
    )

    cow_iso_path = paths.interim / "country_crosswalk_cow_iso.csv"
    if not cow_iso_path.exists():
        _handle_violation("country_crosswalk_cow_iso.csv not found; run build_elections_ned first", strict, logger)
        cow_iso = pd.DataFrame(columns=["cowcode", "iso3"])
    else:
        cow_iso = pd.read_csv(cow_iso_path)

    iso_to_cow = (
        cow_iso.dropna(subset=["iso3", "cowcode"])
        .groupby("iso3")["cowcode"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    clea_elections = merge_with_diagnostics(
        clea_elections,
        iso_to_cow,
        keys=["iso3"],
        how="left",
        name="clea_iso_cow",
        logger=logger,
    )
    clea_elections = harmonize_keys(clea_elections)

    matched_party1 = clea_elections.apply(
        lambda r: match_party_name_basic(r.get("party_1_name"), r.get("cowcode"), party_index), axis=1
    )
    clea_elections["party1_v2paid"] = [m[0] for m in matched_party1]
    clea_elections["party1_match_score"] = [m[1] for m in matched_party1]

    matched_party2 = clea_elections.apply(
        lambda r: match_party_name_basic(r.get("party_2_name"), r.get("cowcode"), party_index), axis=1
    )
    clea_elections["party2_v2paid"] = [m[0] for m in matched_party2]
    clea_elections["party2_match_score"] = [m[1] for m in matched_party2]

    clea_elections["ideo_party1"] = clea_elections.apply(
        lambda r: get_party_ideology_basic(vparty, r.get("party1_v2paid"), r.get("year")), axis=1
    )
    clea_elections["ideo_party2"] = clea_elections.apply(
        lambda r: get_party_ideology_basic(vparty, r.get("party2_v2paid"), r.get("year")), axis=1
    )

    clea_elections["margin_market"] = np.where(
        clea_elections["ideo_party1"] >= clea_elections["ideo_party2"],
        clea_elections["share_1"] - clea_elections["share_2"],
        clea_elections["share_2"] - clea_elections["share_1"],
    )
    clea_elections["D_market_win"] = np.where(
        clea_elections["margin_market"].notna(), (clea_elections["margin_market"] >= 0).astype(int), np.nan
    )

    out_path = paths.processed / "elections_clea.parquet"
    clea_elections.to_parquet(out_path, index=False)

    efw_countries = efw["iso3"].dropna().unique()
    covered = clea_elections["iso3"].dropna().unique()
    coverage_rate = len(set(covered)) / len(set(efw_countries)) if efw_countries.size else 0
    report = pd.DataFrame({
        "metric": ["efw_countries", "covered_by_clea", "coverage_rate"],
        "value": [len(set(efw_countries)), len(set(covered)), coverage_rate],
    })
    report.to_csv(paths.interim / "coverage_clea.csv", index=False)

    validate_required_columns(
        clea_elections,
        ["year", "iso3", "share_1", "share_2", "party_1_name", "party_2_name"],
        "elections_clea",
        strict,
        logger,
    )

    return clea_elections


def compare_coverage(paths: PipelinePaths, logger: logging.Logger) -> None:
    efw_iso = load_efw_iso(paths)
    ned = set()
    clea = set()
    ned_path = paths.processed / "elections_ned.parquet"
    clea_path = paths.processed / "elections_clea.parquet"
    if ned_path.exists():
        df = pd.read_parquet(ned_path, columns=["iso3"])
        ned = set(df["iso3"].dropna().unique())
    if clea_path.exists():
        df = pd.read_parquet(clea_path, columns=["iso3"])
        clea = set(df["iso3"].dropna().unique())

    combined = ned.union(clea)
    summary = pd.DataFrame([
        {"dataset": "EFW", "countries": len(efw_iso)},
        {"dataset": "NED", "countries": len(ned)},
        {"dataset": "CLEA", "countries": len(clea)},
        {"dataset": "NED∪CLEA", "countries": len(combined)},
    ])
    summary["coverage_rate"] = summary["countries"] / len(efw_iso) if efw_iso else np.nan
    summary.to_csv(paths.reports / "coverage_summary.csv", index=False)

    missing = sorted(efw_iso - combined)
    pd.DataFrame({"iso3": missing}).to_csv(paths.reports / "missing_efw_countries.csv", index=False)
    logger.info("Coverage summary saved to %s", paths.reports)


def normalize_party_name(name: str) -> str:
    base = normalize_name(name)
    if not base:
        return ""
    tokens = [t for t in base.split() if t not in STOPWORDS]
    stripped = " ".join(tokens).strip()
    return stripped if stripped else base


def build_party_index(vparty: pd.DataFrame) -> pd.DataFrame:
    name_cols = ["v2paenname", "v2pashname", "v2paorname"]
    rows = []
    for _, row in vparty.iterrows():
        for col in name_cols:
            name = row[col]
            if pd.isna(name):
                continue
            norm = normalize_party_name(str(name))
            if not norm:
                continue
            rows.append({
                "COWcode": row["COWcode"],
                "v2paid": row["v2paid"],
                "party_name_norm": norm,
            })
    return pd.DataFrame(rows).drop_duplicates()


def match_party_name_master(party_name: str, cowcode: float, party_index: pd.DataFrame) -> tuple:
    if pd.isna(party_name) or pd.isna(cowcode):
        return (None, 0)
    name = str(party_name).strip().lower()
    if "independent" in name:
        return (None, 0)
    norm = normalize_party_name(name)
    subset = party_index.loc[party_index["COWcode"] == cowcode]
    if subset.empty:
        return (None, 0)
    if norm in subset["party_name_norm"].values:
        v2paid = subset.loc[subset["party_name_norm"] == norm, "v2paid"].mode()
        return (v2paid.iloc[0] if not v2paid.empty else None, 100)
    match, score = best_fuzzy_match(norm, subset["party_name_norm"].unique().tolist(), min_score=70)
    if not match:
        return (None, score)
    v2paid = subset.loc[subset["party_name_norm"] == match, "v2paid"].mode()
    return (v2paid.iloc[0] if not v2paid.empty else None, score)


def get_party_ideology_master(vparty: pd.DataFrame, v2paid, year) -> tuple:
    if v2paid is None or pd.isna(year):
        return (np.nan, np.nan)
    subset = vparty.loc[vparty["v2paid"] == v2paid, ["year", "v2pariglef"]].dropna()
    if subset.empty:
        return (np.nan, np.nan)
    subset["year"] = pd.to_numeric(subset["year"], errors="coerce").astype("Int64")
    subset = subset.dropna(subset=["year"])
    year = int(year)
    subset["year_diff"] = (subset["year"] - year).abs()
    closest = subset.sort_values("year_diff").iloc[0]
    return (float(closest["v2pariglef"]), int(closest["year_diff"]))


def standardize_shares(df: pd.DataFrame, share_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    max_val = df[list(share_cols)].max().max()
    if pd.notna(max_val) and max_val <= 1.5:
        df[list(share_cols)] = df[list(share_cols)] * 100
    for col in share_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
    return df


def reorder_top_two(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["share_2"] > df["share_1"]
    swap_cols = [
        ("share_1", "share_2"),
        ("party_1_name", "party_2_name"),
        ("party1_v2paid", "party2_v2paid"),
        ("party1_match_score", "party2_match_score"),
        ("ideo_party1", "ideo_party2"),
        ("candidate_1_name", "candidate_2_name"),
    ]
    for a, b in swap_cols:
        if a in df.columns and b in df.columns:
            tmp = df.loc[mask, a].copy()
            df.loc[mask, a] = df.loc[mask, b]
            df.loc[mask, b] = tmp
    return df


def compute_market_margin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["margin"] = df["share_1"] - df["share_2"]
    df["margin_abs"] = df["margin"].abs()
    df["ideo_gap"] = df["ideo_party1"] - df["ideo_party2"]
    df["margin_market"] = np.where(
        df["ideo_party1"].notna() & df["ideo_party2"].notna(),
        np.where(df["ideo_party1"] >= df["ideo_party2"], df["margin"], -df["margin"]),
        np.nan,
    )
    df["D_market_win"] = np.where(df["margin_market"].notna(), (df["margin_market"] >= 0).astype(int), np.nan)
    return df


def compute_merge_score(df: pd.DataFrame) -> pd.Series:
    has_shares = df["share_1"].notna() & df["share_2"].notna()
    has_ideology = df["ideo_party1"].notna() & df["ideo_party2"].notna()
    has_parties = df["party_1_name"].notna() & df["party_2_name"].notna()
    has_date = df["date"].notna()
    has_cow = df["country_cow"].notna()
    has_margin = df["margin_market"].notna()

    score = (
        3 * has_shares.astype(int)
        + 2 * has_margin.astype(int)
        + 2 * has_ideology.astype(int)
        + 1 * has_parties.astype(int)
        + 1 * has_date.astype(int)
        + 1 * has_cow.astype(int)
    )
    score = score + df["source"].map({"NED": 0.25, "CLEA": 0}).fillna(0)
    return score


def _to_str(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("")


def build_merge_key(df: pd.DataFrame, strategy: str) -> pd.Series:
    iso = _to_str(df["iso3"])
    office = _to_str(df["office_type"])
    row_id = _to_str(df["merge_row_id"])

    if strategy == "date_exact":
        date_str = df["date"].dt.strftime("%Y-%m-%d").fillna("")
        suffix = date_str
    elif strategy == "date_fallback":
        date_str = df["date"].dt.strftime("%Y-%m-%d").fillna("")
        year = _to_str(df["year"])
        month = _to_str(df["month"])
        ym = (year + "-" + month).str.rstrip("-")
        suffix = date_str.where(date_str != "", ym)
        suffix = suffix.where(suffix != "", year)
    elif strategy == "year_month":
        year = _to_str(df["year"])
        month = _to_str(df["month"])
        suffix = (year + "-" + month).str.rstrip("-")
    elif strategy == "year":
        suffix = _to_str(df["year"])
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    key = iso + "|" + office + "|" + suffix
    missing = (iso == "") | (office == "") | (suffix == "")
    key = key.where(~missing, key + "|row:" + row_id)
    return key


def evaluate_strategy(df: pd.DataFrame, strategy: str, efw_iso: set[str]) -> tuple[dict, pd.DataFrame]:
    tmp = df.copy()
    tmp["merge_key"] = build_merge_key(tmp, strategy)
    tmp = tmp.sort_values(["merge_key", "merge_score", "source_rank"], ascending=[True, False, True], kind="mergesort")
    best = tmp.drop_duplicates("merge_key", keep="first")

    dup_groups = int((tmp["merge_key"].value_counts() > 1).sum())
    total_rows = len(tmp)
    retention_rate = len(best) / total_rows if total_rows else np.nan
    dedupe_rate = (total_rows - len(best)) / total_rows if total_rows else np.nan
    has_shares = best["share_1"].notna() & best["share_2"].notna()
    has_ideology = best["ideo_party1"].notna() & best["ideo_party2"].notna()

    metrics = {
        "strategy": strategy,
        "rows": int(len(best)),
        "duplicate_groups": dup_groups,
        "rows_deduped": int(len(tmp) - len(best)),
        "coverage_rate": (best["iso3"].nunique() / len(efw_iso)) if efw_iso else np.nan,
        "share_complete_rate": float(has_shares.mean()),
        "margin_market_rate": float(best["margin_market"].notna().mean()),
        "ideology_complete_rate": float(has_ideology.mean()),
        "date_rate": float(best["date"].notna().mean()),
        "avg_merge_score": float(best["merge_score"].mean()),
        "retention_rate": float(retention_rate),
        "dedupe_rate": float(dedupe_rate),
    }

    metrics["strategy_score"] = (
        0.3 * metrics["coverage_rate"]
        + 0.2 * metrics["share_complete_rate"]
        + 0.2 * metrics["margin_market_rate"]
        + 0.1 * metrics["ideology_complete_rate"]
        + 0.05 * metrics["date_rate"]
        + 0.15 * metrics["retention_rate"]
    )
    return metrics, best


def apply_best_merge(df: pd.DataFrame, efw_iso: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = df.copy()
    df["merge_row_id"] = np.arange(len(df))
    df["source_rank"] = df["source"].map({"NED": 0, "CLEA": 1}).fillna(9)
    df["merge_score"] = compute_merge_score(df)

    metrics_list: list[dict] = []
    for strategy in MERGE_STRATEGIES:
        metrics, _ = evaluate_strategy(df, strategy, efw_iso)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    preference = {"date_exact": 4, "date_fallback": 3, "year_month": 2, "year": 1}
    metrics_df["preference"] = metrics_df["strategy"].map(preference).fillna(0)
    best_score = metrics_df["strategy_score"].max()
    candidates = metrics_df.loc[metrics_df["strategy_score"] >= (best_score - MERGE_SCORE_EPS)].copy()
    candidates = candidates.sort_values(["preference", "coverage_rate", "rows"], ascending=[False, False, False])
    best_strategy = str(candidates.iloc[0]["strategy"])
    metrics_df["selected"] = metrics_df["strategy"] == best_strategy

    full = df.copy()
    full["merge_strategy"] = best_strategy
    full["merge_key"] = build_merge_key(full, best_strategy)
    full = full.sort_values(["merge_key", "merge_score", "source_rank"], ascending=[True, False, True], kind="mergesort")
    full["merge_rank"] = full.groupby("merge_key", dropna=False).cumcount()
    full["merge_preferred"] = full["merge_rank"] == 0

    best = full.loc[full["merge_preferred"]].copy()
    return full, best, metrics_df, best_strategy


def build_master(paths: PipelinePaths, logger: logging.Logger, strict: bool = False) -> pd.DataFrame:
    ned = pd.read_parquet(paths.processed / "elections_ned.parquet")
    clea = pd.read_parquet(paths.processed / "elections_clea.parquet")

    ned["source"] = "NED"
    clea["source"] = "CLEA"

    for col in ["nelda3", "nelda4", "nelda5"]:
        if col in ned.columns:
            ned[col] = ned[col].replace({"yes": 1, "no": 0, "unclear": np.nan})
            ned[col] = pd.to_numeric(ned[col], errors="coerce")

    for col in [
        "candidate_1_name",
        "candidate_2_name",
        "party_1_name",
        "party_2_name",
        "party1_v2paid",
        "party2_v2paid",
        "party1_match_score",
        "party2_match_score",
        "ideo_party1",
        "ideo_party2",
        "country_cow",
    ]:
        if col not in clea.columns:
            clea[col] = np.nan

    if "country_cow" in clea.columns and clea["country_cow"].isna().all() and "cowcode" in clea.columns:
        clea["country_cow"] = clea["cowcode"]

    ned = standardize_shares(ned, ["share_1", "share_2"])
    clea = standardize_shares(clea, ["share_1", "share_2"])

    ned = reorder_top_two(ned)
    clea = reorder_top_two(clea)

    if "nelda3" not in clea.columns:
        nelda = load_nelda(paths)
        clea = add_nelda_flags(clea, nelda)

    combined = pd.concat([ned, clea], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
    if "month" in combined.columns:
        combined["month"] = pd.to_numeric(combined["month"], errors="coerce").astype("Int64")
    else:
        combined["month"] = pd.NA
    combined = harmonize_keys(combined)

    vparty = load_vparty(paths)
    party_index = build_party_index(vparty)

    matched1 = combined.apply(
        lambda r: match_party_name_master(r.get("party_1_name"), r.get("country_cow"), party_index), axis=1
    )
    combined["party1_v2paid"] = [m[0] for m in matched1]
    combined["party1_match_score"] = [m[1] for m in matched1]

    matched2 = combined.apply(
        lambda r: match_party_name_master(r.get("party_2_name"), r.get("country_cow"), party_index), axis=1
    )
    combined["party2_v2paid"] = [m[0] for m in matched2]
    combined["party2_match_score"] = [m[1] for m in matched2]

    ideo1 = combined.apply(lambda r: get_party_ideology_master(vparty, r.get("party1_v2paid"), r.get("year")), axis=1)
    combined["ideo_party1"] = [i[0] for i in ideo1]
    combined["ideo_year_diff1"] = [i[1] for i in ideo1]

    ideo2 = combined.apply(lambda r: get_party_ideology_master(vparty, r.get("party2_v2paid"), r.get("year")), axis=1)
    combined["ideo_party2"] = [i[0] for i in ideo2]
    combined["ideo_year_diff2"] = [i[1] for i in ideo2]

    def ideology_quality(score, diff):
        if pd.isna(score) or pd.isna(diff):
            return "missing"
        if score >= 85 and diff <= 2:
            return "high"
        if score >= 75 and diff <= 4:
            return "medium"
        return "low"

    combined["ideo_quality1"] = [ideology_quality(s, d) for s, d in zip(combined["party1_match_score"], combined["ideo_year_diff1"])]
    combined["ideo_quality2"] = [ideology_quality(s, d) for s, d in zip(combined["party2_match_score"], combined["ideo_year_diff2"])]

    quality_rank = {"high": 3, "medium": 2, "low": 1, "missing": 0}
    q1 = combined["ideo_quality1"].map(quality_rank).fillna(0)
    q2 = combined["ideo_quality2"].map(quality_rank).fillna(0)
    combined["ideo_quality"] = np.where(q1 <= q2, combined["ideo_quality1"], combined["ideo_quality2"])

    combined = compute_market_margin(combined)

    def key_for_row(row):
        iso_val = row.get("iso3")
        iso = "" if pd.isna(iso_val) else str(iso_val)
        office_val = row.get("office_type")
        office = "" if pd.isna(office_val) else str(office_val)
        if pd.notna(row.get("date")):
            date_str = row.get("date").strftime("%Y-%m-%d")
        else:
            y = row.get("year")
            m = row.get("month") if pd.notna(row.get("month")) else ""
            date_str = f"{y}-{m}" if pd.notna(y) else ""
        return f"{iso}|{office}|{date_str}"

    combined["election_key"] = combined.apply(key_for_row, axis=1)

    combined["clean_flag"] = (
        combined["iso3"].notna()
        & combined["year"].notna()
        & combined["share_1"].notna()
        & combined["share_2"].notna()
        & (combined["share_1"] >= combined["share_2"])
    )

    nelda_ok = (combined["nelda3"] == 1) & (combined["nelda4"] == 1) & (combined["nelda5"] == 1)
    for col in ["flag_coup", "flag_inconsequential", "flag_unopposed", "flag_indirect"]:
        if col not in combined.columns:
            combined[col] = np.nan
    quality_ok = (
        (combined["flag_coup"].fillna(0) != 1)
        & (combined["flag_inconsequential"].fillna(0) != 1)
        & (combined["flag_unopposed"].fillna(0) != 1)
        & (combined["flag_indirect"].fillna(0) != 1)
    )

    combined["competitive_flag"] = np.where(nelda_ok.notna(), nelda_ok, np.nan)
    combined["sample_competitive"] = combined["clean_flag"] & quality_ok & (nelda_ok.fillna(True))

    efw_iso = load_efw_iso(paths)
    combined["efw_coverage"] = combined["iso3"].isin(efw_iso)

    validate_required_columns(
        combined,
        ["iso3", "year", "share_1", "share_2", "source", "office_type"],
        "elections_master",
        strict,
        logger,
    )

    return combined


def combo_metrics(df: pd.DataFrame, efw_iso: set[str], name: str) -> dict:
    subset = df.copy()
    return {
        "combo": name,
        "rows": len(subset),
        "countries": subset["iso3"].nunique(),
        "coverage_rate": subset["iso3"].nunique() / len(efw_iso) if efw_iso else np.nan,
        "margin_market_share": subset["margin_market"].notna().mean(),
        "clean_share": subset["clean_flag"].mean(),
        "competitive_share": subset["sample_competitive"].mean(),
        "source_ned_share": (subset["source"] == "NED").mean(),
    }


def build_master_outputs(paths: PipelinePaths, logger: logging.Logger, strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = build_master(paths, logger, strict=strict)
    efw_iso = load_efw_iso(paths)

    full_df, best_df, strategy_metrics, best_strategy = apply_best_merge(full_df, efw_iso)

    out_full = paths.processed / "elections_master_full.parquet"
    out_best = paths.processed / "elections_master.parquet"
    full_df.to_parquet(out_full, index=False)
    best_df.to_parquet(out_best, index=False)

    strategy_metrics.to_csv(paths.reports / "merge_strategy_metrics.csv", index=False)
    (paths.reports / "merge_strategy_choice.txt").write_text(f"best_strategy={best_strategy}\n")

    combos = [
        combo_metrics(best_df, efw_iso, "combined_all_best"),
        combo_metrics(best_df[best_df["clean_flag"]], efw_iso, "combined_clean_best"),
        combo_metrics(best_df[best_df["sample_competitive"]], efw_iso, "combined_clean_competitive_best"),
        combo_metrics(best_df[best_df["source"] == "NED"], efw_iso, "ned_all_best"),
        combo_metrics(best_df[best_df["source"] == "CLEA"], efw_iso, "clea_all_best"),
        combo_metrics(best_df[(best_df["source"] == "NED") & (best_df["sample_competitive"])], efw_iso, "ned_competitive_best"),
    ]
    pd.DataFrame(combos).to_csv(paths.reports / "election_combinations_metrics.csv", index=False)

    missing_iso = sorted(efw_iso - set(best_df[best_df["clean_flag"]]["iso3"].dropna().unique()))
    pd.DataFrame({"iso3": missing_iso}).to_csv(paths.reports / "missing_efw_countries_clean.csv", index=False)

    return full_df, best_df
