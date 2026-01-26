from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pycountry


@dataclass
class CleaBundle:
    elections: pd.DataFrame
    results: pd.DataFrame


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


def _read_table(
    path: Path,
    *,
    zip_member: str | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CLEA file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".zip":
        if not zip_member:
            raise ValueError(f"zip_member is required for zipped CLEA file: {path}")
        with zipfile.ZipFile(path) as zf:
            with zf.open(zip_member) as handle:
                return pd.read_csv(handle, usecols=usecols)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=usecols)
    if suffix in {".sav"}:
        sidecars = [
            path.with_suffix(".parquet"),
            path.with_name(f"{path.stem}_subset.parquet"),
        ]
        for candidate in sidecars:
            if candidate.exists():
                try:
                    return pd.read_parquet(candidate, columns=usecols)
                except Exception as exc:
                    raise ValueError(
                        f"CLEA parquet sidecar {candidate} missing expected columns."
                    ) from exc
        import pyreadstat

        df, _ = pyreadstat.read_sav(path, usecols=usecols)
        return df
    if suffix in {".dta"}:
        return pd.read_stata(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, usecols=usecols)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported CLEA file type: {path}")


def load_clea_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"CLEA manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _standardize_columns(df: pd.DataFrame, mapping: dict, required: list[str]) -> pd.DataFrame:
    rename_map = {mapping[key]: key for key in mapping if mapping.get(key) in df.columns}
    out = df.rename(columns=rename_map).copy()
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in CLEA table: {missing}")
    return out


def build_clea_bundle(raw_dir: Path, manifest_path: Path) -> CleaBundle:
    manifest = load_clea_manifest(manifest_path)
    elections_file = raw_dir / manifest["election_file"]
    results_file = raw_dir / manifest["result_file"]

    election_usecols = list(manifest["election_columns"].values())
    result_usecols = list(manifest["result_columns"].values())

    if elections_file.resolve() == results_file.resolve():
        elections_raw = _read_table(
            elections_file,
            zip_member=manifest.get("election_zip_member"),
            usecols=election_usecols,
        )
        results_raw = _read_table(
            results_file,
            zip_member=manifest.get("result_zip_member"),
            usecols=result_usecols,
        )
    else:
        elections_raw = _read_table(
            elections_file,
            zip_member=manifest.get("election_zip_member"),
            usecols=election_usecols,
        )
        results_raw = _read_table(
            results_file,
            zip_member=manifest.get("result_zip_member"),
            usecols=result_usecols,
        )

    elections = _standardize_columns(
        elections_raw,
        manifest["election_columns"],
        required=["election_id"],
    )
    results = _standardize_columns(
        results_raw,
        manifest["result_columns"],
        required=["election_id"],
    )

    if "election_year" not in elections.columns:
        if "election_date" in elections.columns:
            elections["election_date"] = pd.to_datetime(elections["election_date"], errors="coerce")
            elections["election_year"] = elections["election_date"].dt.year
        else:
            raise ValueError("CLEA elections must include election_year or election_date.")

    elections = elections.dropna(subset=["election_id"]).copy()
    elections = elections.drop_duplicates(subset=["election_id"]).copy()

    if "iso3c" not in elections.columns:
        if "country_name" not in elections.columns:
            raise ValueError("CLEA elections must include iso3c or country_name.")
        unique_countries = elections["country_name"].dropna().unique()
        country_map = {name: _map_country_to_iso3(name) for name in unique_countries}
        elections["iso3c"] = elections["country_name"].map(country_map)
    elections["iso3c"] = elections["iso3c"].astype(str).str.upper().str.strip()

    elections = elections.dropna(subset=["iso3c", "election_year"]).copy()
    elections["election_year"] = elections["election_year"].astype(int)

    if "party_id" not in results.columns:
        if "party_name" in results.columns:
            results["party_id"] = results["party_name"].astype(str)
        else:
            raise ValueError("CLEA results must include party_id or party_name.")

    for col in ["votes", "seats", "vote_share"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    results = results.merge(
        elections[["election_id", "iso3c", "election_year"]],
        on="election_id",
        how="left",
    )
    results = results.dropna(subset=["iso3c", "election_year"]).copy()

    if "constituency_id" in results.columns:
        agg_cols = {
            "votes": "sum" if "votes" in results.columns else "sum",
            "seats": "sum" if "seats" in results.columns else "sum",
        }
        group_cols = ["election_id", "party_id", "party_name", "iso3c", "election_year"]
        group_cols = [col for col in group_cols if col in results.columns]
        results = results.groupby(group_cols, as_index=False).agg(agg_cols)

    if "vote_share" not in results.columns and "votes" in results.columns:
        totals = results.groupby("election_id")["votes"].sum().rename("vote_total")
        results = results.merge(totals, on="election_id", how="left")
        results["vote_share"] = results["votes"] / results["vote_total"].replace(0, pd.NA)
        results = results.drop(columns=["vote_total"])
    elif "vote_share" in results.columns:
        max_share = results["vote_share"].dropna().max()
        if pd.notna(max_share) and max_share > 1.5:
            results["vote_share"] = results["vote_share"] / 100.0

    if "seats" in results.columns:
        totals = results.groupby("election_id")["seats"].sum().rename("seat_total")
        results = results.merge(totals, on="election_id", how="left")
        results["seat_share"] = results["seats"] / results["seat_total"].replace(0, pd.NA)
        results = results.drop(columns=["seat_total"])

    return CleaBundle(elections=elections, results=results)


def write_clea_outputs(bundle: CleaBundle, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elections": out_dir / "clea_elections.parquet",
        "results": out_dir / "clea_results.parquet",
    }
    bundle.elections.to_parquet(paths["elections"], index=False)
    bundle.results.to_parquet(paths["results"], index=False)
    return paths


def clea_metadata(bundle: CleaBundle, manifest_path: Path) -> dict:
    return {
        "source": {
            "manifest_path": str(manifest_path),
        },
        "build": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tables": {
                "elections": len(bundle.elections),
                "results": len(bundle.results),
                "countries": bundle.elections["iso3c"].nunique(),
            },
        },
    }


def write_clea_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
