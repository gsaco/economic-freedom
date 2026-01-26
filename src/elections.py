from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PARLGOV_ZIP_URL = (
    "https://raw.githubusercontent.com/hdigital/parlgov/main/static/data/"
    "parlgov-development_csv-utf-8.zip"
)
PARLGOV_CODEBOOK_URL = "https://raw.githubusercontent.com/hdigital/parlgov/main/static/data/codebook.pdf"


@dataclass
class ParlGovBundle:
    elections: pd.DataFrame
    election_results: pd.DataFrame
    cabinets: pd.DataFrame
    cabinet_parties: pd.DataFrame
    parties: pd.DataFrame
    countries: pd.DataFrame


def download_parlgov_zip(dest_dir: Path, url: str = PARLGOV_ZIP_URL) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "parlgov.zip"
    if zip_path.exists():
        return zip_path

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    return zip_path


def download_parlgov_codebook(dest_dir: Path, url: str = PARLGOV_CODEBOOK_URL) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    codebook_path = dest_dir / "parlgov_codebook.pdf"
    if codebook_path.exists():
        return codebook_path

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    codebook_path.write_bytes(resp.content)
    return codebook_path


def _read_csv_from_zip(zip_path: Path, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as handle:
            return pd.read_csv(handle)


def _load_parlgov_tables(zip_path: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "country": "country.csv",
        "external_country_iso": "external_country_iso.csv",
        "info_id": "info_id.csv",
        "election": "election.csv",
        "election_result": "election_result.csv",
        "cabinet": "cabinet.csv",
        "cabinet_party": "cabinet_party.csv",
        "party": "party.csv",
        "party_family": "party_family.csv",
        "party_position": "viewcalc_party_position.csv",
    }
    return {key: _read_csv_from_zip(zip_path, name) for key, name in tables.items()}


def _build_country_map(country: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    iso_map = iso.rename(columns={"isonumeric": "iso_numeric"})[["iso_numeric", "iso3", "iso2", "country"]]
    mapped = country.merge(iso_map, on="iso_numeric", how="left")
    mapped["iso3c"] = mapped["iso3"].astype(str).str.upper().str.strip()
    mapped = mapped.rename(columns={"name": "country_name"})
    return mapped[["id", "country_name", "iso3c", "iso_numeric"]]


def _map_info_labels(info_id: pd.DataFrame, table_variable: str) -> dict[int, str]:
    subset = info_id[info_id["table_variable"] == table_variable]
    return dict(zip(subset["id"].astype(int), subset["name"].astype(str)))


def load_parlgov_bundle(zip_path: Path) -> ParlGovBundle:
    tables = _load_parlgov_tables(zip_path)
    country_map = _build_country_map(tables["country"], tables["external_country_iso"])

    election_types = _map_info_labels(tables["info_id"], "election_type")
    party_families = _map_info_labels(tables["info_id"], "party_family")

    elections = tables["election"].merge(country_map, left_on="country_id", right_on="id", how="left")
    elections["election_date"] = pd.to_datetime(elections["date"], errors="coerce")
    elections["year"] = elections["election_date"].dt.year
    elections["election_type"] = elections["type_id"].map(election_types)
    elections = elections.rename(columns={"id_x": "election_id"})
    elections = elections.drop(columns=["id_y"], errors="ignore")

    election_results = tables["election_result"].merge(
        elections[["election_id", "iso3c", "year"]],
        left_on="election_id",
        right_on="election_id",
        how="left",
    )

    cabinets = tables["cabinet"].merge(country_map, left_on="country_id", right_on="id", how="left")
    cabinets["start_date"] = pd.to_datetime(cabinets["start_date"], errors="coerce")
    cabinets = cabinets.rename(columns={"id_x": "cabinet_id"})
    cabinets = cabinets.drop(columns=["id_y"], errors="ignore")

    cabinet_parties = tables["cabinet_party"].rename(columns={"id": "cabinet_party_id"})

    parties = tables["party"].merge(country_map, left_on="country_id", right_on="id", how="left")
    parties = parties.rename(columns={"id_x": "party_id"})
    parties = parties.drop(columns=["id_y"], errors="ignore")
    parties["family_name"] = parties["family_id"].map(party_families)

    positions = tables["party_position"]
    parties = parties.merge(
        positions[["party_id", "left_right", "state_market", "liberty_authority", "eu_anti_pro"]],
        on="party_id",
        how="left",
    )

    countries = country_map.rename(columns={"id": "country_id"})

    return ParlGovBundle(
        elections=elections,
        election_results=election_results,
        cabinets=cabinets,
        cabinet_parties=cabinet_parties,
        parties=parties,
        countries=countries,
    )


def write_parlgov_outputs(bundle: ParlGovBundle, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elections": out_dir / "parlgov_elections.parquet",
        "election_results": out_dir / "parlgov_election_results.parquet",
        "cabinets": out_dir / "parlgov_cabinets.parquet",
        "cabinet_parties": out_dir / "parlgov_cabinet_parties.parquet",
        "parties": out_dir / "parlgov_parties.parquet",
        "countries": out_dir / "parlgov_countries.parquet",
    }

    bundle.elections.to_parquet(paths["elections"], index=False)
    bundle.election_results.to_parquet(paths["election_results"], index=False)
    bundle.cabinets.to_parquet(paths["cabinets"], index=False)
    bundle.cabinet_parties.to_parquet(paths["cabinet_parties"], index=False)
    bundle.parties.to_parquet(paths["parties"], index=False)
    bundle.countries.to_parquet(paths["countries"], index=False)

    return paths


def parlgov_metadata(bundle: ParlGovBundle, zip_path: Path, codebook_path: Path | None) -> dict:
    return {
        "source": {
            "zip_path": str(zip_path),
            "zip_url": PARLGOV_ZIP_URL,
            "codebook_path": str(codebook_path) if codebook_path else None,
            "codebook_url": PARLGOV_CODEBOOK_URL,
        },
        "build": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tables": {
                "elections": len(bundle.elections),
                "election_results": len(bundle.election_results),
                "cabinets": len(bundle.cabinets),
                "cabinet_parties": len(bundle.cabinet_parties),
                "parties": len(bundle.parties),
                "countries": len(bundle.countries),
            },
        },
    }


def write_parlgov_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
