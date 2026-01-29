from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


def write_fixtures(root: Path) -> None:
    raw = root / "data" / "raw"
    external = root / "data" / "external"

    # EFW panel dataset
    efw_dir = raw / "efw"
    efw_dir.mkdir(parents=True, exist_ok=True)
    efw_df = pd.DataFrame({
        "ISO_Code": ["USA", "FRA"],
        "Countries": ["United States", "France"],
        "Year": [2000, 2000],
        "Summary": [5.0, 6.0],
        "Area 1": [5.0, 6.0],
        "Area 2": [5.0, 6.0],
        "Area 3": [5.0, 6.0],
        "Area 4": [5.0, 6.0],
        "Area 5": [5.0, 6.0],
    })
    with pd.ExcelWriter(efw_dir / "efotw-2025-master-index-data-for-researchers-iso.xlsx") as writer:
        efw_df.to_excel(writer, sheet_name="EFW Panel Dataset", index=False)

    # V-Party
    vparty_dir = raw / "vparty" / "CPD_V-Party_CSV_v2"
    vparty_dir.mkdir(parents=True, exist_ok=True)
    vparty_df = pd.DataFrame({
        "v2paid": [1, 2, 3, 4],
        "country_name": ["United States", "United States", "France", "France"],
        "COWcode": [2, 2, 220, 220],
        "year": [2000, 2000, 2000, 2000],
        "v2paenname": ["Alpha Party", "Beta Party", "Gamma Party", "Delta Party"],
        "v2pashname": [np.nan, np.nan, np.nan, np.nan],
        "v2paorname": [np.nan, np.nan, np.nan, np.nan],
        "v2pariglef": [2.0, 4.0, np.nan, np.nan],
    })
    vparty_df.to_csv(vparty_dir / "V-Dem-CPD-Party-V2.csv", index=False)

    # NED presidential/parliamentary
    ned_dir = raw / "ned"
    ned_dir.mkdir(parents=True, exist_ok=True)
    pres_df = pd.DataFrame({
        "country": ["United States"],
        "country_cow": [2],
        "year": [2000],
        "month": [1],
        "date": ["2000-01-15"],
        "party_1": ["Alpha Party"],
        "party_2": ["Beta Party"],
        "candidate_1": ["Alice"],
        "candidate_2": ["Bob"],
        "vote_share1_1": [0.6],
        "vote_share2_1": [0.4],
        "vote_share1_2": [np.nan],
        "vote_share2_2": [np.nan],
        "flag_coup": [0],
        "flag_inconsequential": [0],
        "flag_unopposed": [0],
        "flag_indirect": [0],
    })
    pres_df.to_stata(ned_dir / "presidential_elections_v2.dta", write_index=False)

    parl_df = pd.DataFrame({
        "country": ["United States"],
        "country_cow": [2],
        "year": [2000],
        "month": [1],
        "date": ["2000-01-01"],
        "party_1": ["Alpha Party"],
        "party_2": ["Beta Party"],
        "seat_share_1": [0.52],
        "seat_share_2": [0.48],
    })
    parl_df.to_stata(ned_dir / "parliamentary_elections_v2.dta", write_index=False)

    # NELDA
    nelda_dir = raw / "nelda" / "NELDA 6.0"
    nelda_dir.mkdir(parents=True, exist_ok=True)
    nelda_df = pd.DataFrame({
        "ccode": [2, 2, 220],
        "year": [2000, 2000, 2000],
        "mmdd": [115, 101, 101],
        "types": ["Executive", "Legislative/Parliamentary", "Legislative/Parliamentary"],
        "nelda3": [1, 1, 0],
        "nelda4": [1, 1, 0],
        "nelda5": [1, 1, 0],
    })
    pyreadstat.write_dta(nelda_df, nelda_dir / "id & q-wide_share.dta")

    # CLEA
    clea_dir = raw / "clea"
    clea_dir.mkdir(parents=True, exist_ok=True)
    clea_df = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "ctr": ["USA", "USA", "FRA", "FRA"],
        "ctr_n": ["United States", "United States", "France", "France"],
        "yr": [2000, 2000, 2000, 2000],
        "mn": [1, 1, 3, 3],
        "pty_n": ["Alpha Party", "Beta Party", "Gamma Party", "Delta Party"],
        "pv1": [600, 400, 300, 700],
        "seat": [55, 45, 40, 60],
    })
    pyreadstat.write_sav(clea_df, clea_dir / "clea_lc_20251015.sav")

    # External datasets
    external.mkdir(parents=True, exist_ok=True)
    pf_df = pd.DataFrame({
        "country": ["United States", "United States", "United States", "United States"],
        "dataset_key": ["vparty", "vparty", "ches", "ches"],
        "dataset_party_id": [1, 2, 201, 202],
        "partyfacts_id": [101, 102, 101, 102],
        "name_short": ["Alpha", "Beta", "Alpha", "Beta"],
        "name": ["Alpha Party", "Beta Party", "Alpha Party", "Beta Party"],
    })
    pf_df.to_csv(external / "partyfacts_external_parties.csv", index=False)

    ches_df = pd.DataFrame({
        "year": [2000, 2000],
        "country": ["United States", "United States"],
        "party_id": [201, 202],
        "party": ["Alpha Party", "Beta Party"],
        "electionyear": [2000, 2000],
        "lrecon": [2.5, 4.5],
    })
    ches_df.to_csv(external / "ches_1999_2024.csv", index=False)

    elff_df = pd.DataFrame({
        "countryname": ["France"],
        "partyname": ["Delta Party"],
        "year": [2000],
        "econlr": [5.5],
    })
    elff_df.to_csv(external / "elff_partypos_summaries.csv", index=False)

    parlgov_df = pd.DataFrame({
        "country_name": ["United States", "United States", "France", "France"],
        "party_name_english": ["Alpha Party", "Beta Party", "Gamma Party", "Delta Party"],
        "party_name": ["Alpha Party", "Beta Party", "Gamma Party", "Delta Party"],
        "state_market": [2.0, 4.0, 1.0, 5.0],
        "left_right": [2.0, 4.0, 2.0, 6.0],
    })
    parlgov_zip = external / "parlgov.zip"
    with zipfile.ZipFile(parlgov_zip, "w") as zf:
        zf.writestr("view_party.csv", parlgov_df.to_csv(index=False))

    des_df = pd.DataFrame({
        "country": ["United States", "France"],
        "date": ["2000-01-01", "2000-03-01"],
        "elecrule": ["PR", "PR"],
        "tier1_formula": ["D'Hondt", "Sainte-Lague"],
        "tier1_avemag": [10, 8],
        "mixed_type": ["mixed", "mixed"],
    })
    des_zip = external / "des_es_data_v50.zip"
    with zipfile.ZipFile(des_zip, "w") as zf:
        zf.writestr("es_data-v5_0.csv", des_df.to_csv(index=False))

    idea_df = pd.DataFrame({
        "ISO3": ["USA", "FRA"],
        "Year": [2000, 2000],
        "Electoral system family": ["Mixed", "PR"],
        "Electoral system for national legislature": ["Mixed", "PR"],
        "Electoral system for the president": ["Majority", "Majority"],
        "Number of tiers": [2, 1],
        "Legislative size (directly elected)": [100, 80],
        "Legislative size (voting members)": [100, 80],
    })
    with pd.ExcelWriter(external / "idea_export_electoral_system_design_database.xlsx") as writer:
        idea_df.to_excel(writer, index=False)

    dpi_dir = external / "dpi" / "DPI2020"
    dpi_dir.mkdir(parents=True, exist_ok=True)
    dpi_df = pd.DataFrame({
        "countryname": ["United States", "France"],
        "year": [2000, 2000],
        "system": [1, 2],
        "execme": [1, 1],
        "execrlc": [1, 1],
        "gov1rlc": [1, 1],
        "checks": [2, 3],
        "checks_lax": [1, 1],
        "military": [0, 0],
    })
    dpi_df.to_csv(dpi_dir / "dpi2020.csv", index=False)
