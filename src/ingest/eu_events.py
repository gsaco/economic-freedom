from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


def _default_events() -> pd.DataFrame:
    data = [
        # 2004 wave
        {"iso3": "CYP", "country_name": "Cyprus", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "CZE", "country_name": "Czech Republic", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "EST", "country_name": "Estonia", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "HUN", "country_name": "Hungary", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "LVA", "country_name": "Latvia", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "LTU", "country_name": "Lithuania", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "MLT", "country_name": "Malta", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "POL", "country_name": "Poland", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "SVK", "country_name": "Slovakia", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        {"iso3": "SVN", "country_name": "Slovenia", "eu_wave": "2004", "cand_year": 1997, "nego_open_year": 1998, "accession_year": 2004},
        # 2007 wave
        {"iso3": "BGR", "country_name": "Bulgaria", "eu_wave": "2007", "cand_year": 1999, "nego_open_year": 2000, "accession_year": 2007},
        {"iso3": "ROU", "country_name": "Romania", "eu_wave": "2007", "cand_year": 1999, "nego_open_year": 2000, "accession_year": 2007},
        # 2013 wave
        {"iso3": "HRV", "country_name": "Croatia", "eu_wave": "2013", "cand_year": 2004, "nego_open_year": 2005, "accession_year": 2013},
        # earlier accessions and founders
        {"iso3": "AUT", "country_name": "Austria", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1995},
        {"iso3": "FIN", "country_name": "Finland", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1995},
        {"iso3": "SWE", "country_name": "Sweden", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1995},
        {"iso3": "ESP", "country_name": "Spain", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1986},
        {"iso3": "PRT", "country_name": "Portugal", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1986},
        {"iso3": "GRC", "country_name": "Greece", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1981},
        {"iso3": "DNK", "country_name": "Denmark", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1973},
        {"iso3": "IRL", "country_name": "Ireland", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1973},
        {"iso3": "GBR", "country_name": "United Kingdom", "eu_wave": "other", "cand_year": None, "nego_open_year": None, "accession_year": 1973},
        {"iso3": "BEL", "country_name": "Belgium", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
        {"iso3": "FRA", "country_name": "France", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
        {"iso3": "DEU", "country_name": "Germany", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
        {"iso3": "ITA", "country_name": "Italy", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
        {"iso3": "LUX", "country_name": "Luxembourg", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
        {"iso3": "NLD", "country_name": "Netherlands", "eu_wave": "founder", "cand_year": None, "nego_open_year": None, "accession_year": 1958},
    ]
    return pd.DataFrame(data)


def ingest_eu_events() -> Path:
    raw_path = RAW_DIR / "eu" / "eu_milestones.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        df = _default_events()
        df.to_csv(raw_path, index=False)

    required = {"iso3", "country_name", "eu_wave", "cand_year", "nego_open_year", "accession_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EU milestones missing columns: {missing}")

    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    for col in ["cand_year", "nego_open_year", "accession_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = PROCESSED_DIR / "eu_events.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ingest_eu_events()
