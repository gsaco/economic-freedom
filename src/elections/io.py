from __future__ import annotations

from pathlib import Path
import zipfile
import pandas as pd
import pyreadstat


def _read_dta(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if usecols is None:
        df, _ = pyreadstat.read_dta(path)
        return df
    # filter to existing columns
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    cols = [c for c in usecols if c in meta.column_names]
    df, _ = pyreadstat.read_dta(path, usecols=cols)
    return df


def _read_sav(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if usecols is None:
        df, _ = pyreadstat.read_sav(path)
        return df
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    cols = [c for c in usecols if c in meta.column_names]
    df, _ = pyreadstat.read_sav(path, usecols=cols)
    return df


def read_ned_pres(path: Path) -> pd.DataFrame:
    usecols = [
        "country",
        "country_cow",
        "country_abb",
        "year",
        "month",
        "date",
        "type_election",
        "candidate_1",
        "candidate_2",
        "party_1",
        "party_2",
        "vote_share1_1",
        "vote_share1_2",
        "vote_share2_1",
        "vote_share2_2",
        "flag_coup",
        "flag_inconsequential",
        "flag_unopposed",
        "flag_indirect",
        "flag_two_round",
    ]
    return _read_dta(path, usecols=usecols)


def read_ned_parl(path: Path) -> pd.DataFrame:
    usecols = [
        "country",
        "country_cow",
        "country_abb",
        "year",
        "month",
        "date",
        "type_election",
        "party_1",
        "party_2",
        "seat_share_1",
        "seat_share_2",
        "flag_coup",
        "flag_inconsequential",
        "flag_constituent",
        "flag_vacant_seats",
    ]
    return _read_dta(path, usecols=usecols)


def read_nelda(path: Path) -> pd.DataFrame:
    usecols = [
        "electionid",
        "ccode",
        "country",
        "year",
        "mmdd",
        "types",
        "nelda3",
        "nelda4",
        "nelda5",
    ]
    return _read_dta(path, usecols=usecols)


def read_clea_lc(path: Path) -> pd.DataFrame:
    usecols = [
        "id",
        "ctr",
        "ctr_n",
        "yr",
        "mn",
        "pty",
        "pty_n",
        "pv1",
        "pvs1",
        "seat",
        "vv1",
        "ivv1",
        "cv1",
        "cvs1",
    ]
    return _read_sav(path, usecols=usecols)


def read_efw_panel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="EFW Panel Dataset")
    return df


def read_vparty(path: Path) -> pd.DataFrame:
    usecols = ["v2paid", "pf_party_id", "country_name", "country_id", "COWcode", "year", "v2pariglef"]
    return pd.read_csv(path, usecols=usecols)


def read_partyfacts_core(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_partyfacts_external(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_ches(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_elff(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_parlgov_zip(path: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as z:
        for name in [
            "external_country_iso.csv",
            "party.csv",
            "election.csv",
            "election_result.csv",
            "cabinet.csv",
            "cabinet_party.csv",
            "viewcalc_party_position.csv",
        ]:
            if name in z.namelist():
                out[name] = pd.read_csv(z.open(name))
    return out


def read_des_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open("es_data-v5_0.csv") as f:
            return pd.read_csv(f)


def read_idea(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def read_dpi(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def read_cow2iso(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize key fields
    if "cow_id" in df.columns:
        df["cow_id"] = pd.to_numeric(df["cow_id"], errors="coerce").astype("Int64")
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype(str).str.upper()
        df.loc[df["iso3"].isin(["", "NAN", "NONE"]), "iso3"] = pd.NA
    for col in ["valid_from", "valid_until"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df
