from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pycountry


COUNTRY_OVERRIDES = {
    "BOLIVIA": "BOL",
    "BOLIVIA (PLURINATIONAL STATE OF)": "BOL",
    "BRUNEI": "BRN",
    "BRUNEI DARUSSALAM": "BRN",
    "CAPE VERDE": "CPV",
    "COTE D'IVOIRE": "CIV",
    "COTE DIVOIRE": "CIV",
    "CONGO, DEM. REP.": "COD",
    "CONGO, DEMOCRATIC REPUBLIC OF": "COD",
    "CONGO, REP.": "COG",
    "CONGO, REPUBLIC OF": "COG",
    "CZECH REPUBLIC": "CZE",
    "EGYPT, ARAB REP.": "EGY",
    "GAMBIA, THE": "GMB",
    "HONG KONG SAR, CHINA": "HKG",
    "IRAN, ISLAMIC REP.": "IRN",
    "KOREA, DEM. PEOPLE'S REP.": "PRK",
    "KOREA, REP.": "KOR",
    "KYRGYZ REPUBLIC": "KGZ",
    "LAO PDR": "LAO",
    "MACAO SAR, CHINA": "MAC",
    "MICRONESIA, FED. STS.": "FSM",
    "MOLDOVA": "MDA",
    "RUSSIAN FEDERATION": "RUS",
    "SLOVAK REPUBLIC": "SVK",
    "SYRIAN ARAB REPUBLIC": "SYR",
    "TANZANIA": "TZA",
    "UNITED STATES": "USA",
    "VENEZUELA, RB": "VEN",
    "VIET NAM": "VNM",
    "WEST BANK AND GAZA": "PSE",
    "YEMEN, REP.": "YEM",
}


def _normalize_country_name(name: str) -> str:
    clean = re.sub(r"\\([^)]*\\)", "", str(name)).strip()
    clean = re.sub(r"\\s+", " ", clean)
    return clean.upper()


def _map_country_to_iso3(name: str) -> str | None:
    if not name or str(name).strip() == "":
        return None
    key = _normalize_country_name(name)
    if key in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[key]
    try:
        match = pycountry.countries.search_fuzzy(name)[0]
        return match.alpha_3
    except LookupError:
        return None


def normalize_party_name(name: str) -> str:
    if not name or str(name).strip() == "":
        return ""
    cleaned = re.sub(r"[\\W_]+", " ", str(name).lower()).strip()
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned


@dataclass
class IdeologyBundle:
    dpi_exec: pd.DataFrame | None
    vparty: pd.DataFrame | None


def load_dpi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing DPI file: {path}")
    df = pd.read_excel(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "countryname" not in df.columns or "year" not in df.columns:
        raise ValueError("DPI file must include countryname and year columns.")

    df["iso3c"] = df["countryname"].apply(_map_country_to_iso3)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    exec_col = "execrlc" if "execrlc" in df.columns else None
    gov_col = "gov1rlc" if "gov1rlc" in df.columns else None

    def _clean_rlc(val: object) -> str | None:
        if pd.isna(val):
            return None
        txt = str(val).strip()
        if txt in {"-999", "0", ""}:
            return None
        return txt

    if exec_col:
        df["exec_rlc"] = df[exec_col].apply(_clean_rlc)
    if gov_col:
        df["gov1_rlc"] = df[gov_col].apply(_clean_rlc)

    out = df[["iso3c", "year", "exec_rlc", "gov1_rlc"]].dropna(subset=["iso3c", "year"])
    out["iso3c"] = out["iso3c"].astype(str).str.upper().str.strip()
    out["year"] = out["year"].astype(int)
    return out


def load_vparty(raw_dir: Path) -> pd.DataFrame | None:
    candidates = [
        raw_dir / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv",
        raw_dir / "vparty_party_positions.parquet",
        raw_dir / "vparty_party_positions.csv",
        raw_dir / "vparty_party_positions.dta",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_stata(path)
            df.columns = [str(col).strip().lower() for col in df.columns]
            return df
    return None


def prepare_vparty_positions(path: Path) -> pd.DataFrame:
    usecols = [
        "country_text_id",
        "year",
        "v2paenname",
        "v2paorname",
        "v2pashname",
        "v2pariglef",
        "v2pariglef_mean",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df["iso3c"] = df["country_text_id"].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["ideology_lr"] = df["v2pariglef_mean"].combine_first(df["v2pariglef"])

    name_cols = ["v2paenname", "v2paorname", "v2pashname"]
    rows = []
    for col in name_cols:
        subset = df[["iso3c", "year", "ideology_lr", col]].copy()
        subset = subset.rename(columns={col: "party_name"})
        subset["party_name_norm"] = subset["party_name"].apply(normalize_party_name)
        rows.append(subset)
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["iso3c", "year", "ideology_lr"])
    long_df = long_df[long_df["party_name_norm"] != ""].copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df.drop_duplicates(subset=["iso3c", "year", "party_name_norm"])
    return long_df[["iso3c", "year", "party_name_norm", "ideology_lr"]]


def classify_market_ideology(rlc: str | None) -> float | None:
    if rlc is None:
        return None
    rlc = rlc.strip().lower()
    if rlc == "right":
        return 1.0
    if rlc == "left":
        return 0.0
    return None


def classify_market_lr(value: float | None, *, threshold: float = 0.0) -> float | None:
    if value is None or pd.isna(value):
        return None
    if value > threshold:
        return 1.0
    if value < threshold:
        return 0.0
    return None


def build_ideology_bundle(dpi_path: Path, vparty_dir: Path) -> IdeologyBundle:
    dpi = load_dpi(dpi_path) if dpi_path.exists() else None
    vparty = load_vparty(vparty_dir)
    return IdeologyBundle(dpi_exec=dpi, vparty=vparty)
