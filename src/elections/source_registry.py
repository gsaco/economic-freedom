from __future__ import annotations

from pathlib import Path
import pandas as pd

from .io import read_ned_pres, read_ned_parl, read_clea_lc, read_vparty, read_cow2iso


def _year_range(series) -> str | None:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return None
    return f"{int(s.min())}-{int(s.max())}"


def build_source_registry(raw_root: Path, external_root: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    repo_root = raw_root.parent.parent

    # NED pres
    ned_pres = read_ned_pres(raw_root / "ned" / "presidential_elections_v2.dta")
    rows.append({
        "source_id": "ned_pres_v2",
        "file": str(raw_root / "ned" / "presidential_elections_v2.dta"),
        "format": "Stata .dta",
        "unit_of_observation": "National-level presidential election",
        "time_coverage": _year_range(ned_pres.get("year")),
        "geo_coverage": f"countries={ned_pres['country'].nunique()}" if "country" in ned_pres else None,
        "core_IDs_present": ", ".join([c for c in ["country_cow", "country_abb", "country", "date"] if c in ned_pres.columns]),
        "suspected_duplicates": "duplicates on (country,date) possible if date missing",
        "key_variables": "candidate_1/2, party_1/2, vote_share1_1/2, vote_share2_1/2",
        "notes": f"rows={len(ned_pres)} cols={len(ned_pres.columns)}",
    })

    # NED parl
    ned_parl = read_ned_parl(raw_root / "ned" / "parliamentary_elections_v2.dta")
    rows.append({
        "source_id": "ned_parl_v2",
        "file": str(raw_root / "ned" / "parliamentary_elections_v2.dta"),
        "format": "Stata .dta",
        "unit_of_observation": "National-level parliamentary election",
        "time_coverage": _year_range(ned_parl.get("year")),
        "geo_coverage": f"countries={ned_parl['country'].nunique()}" if "country" in ned_parl else None,
        "core_IDs_present": ", ".join([c for c in ["country_cow", "country_abb", "country", "date"] if c in ned_parl.columns]),
        "suspected_duplicates": "duplicates on (country,date) possible if date missing",
        "key_variables": "party_1/2, seat_share_1/2",
        "notes": f"rows={len(ned_parl)} cols={len(ned_parl.columns)}",
    })

    # CLEA LC
    clea = read_clea_lc(raw_root / "clea" / "clea_lc_20251015.sav")
    rows.append({
        "source_id": "clea_lc_20251015",
        "file": str(raw_root / "clea" / "clea_lc_20251015.sav"),
        "format": "SPSS .sav",
        "unit_of_observation": "Constituency × party (lower chamber)",
        "time_coverage": _year_range(clea.get("yr")),
        "geo_coverage": f"countries={clea['ctr_n'].nunique()}" if "ctr_n" in clea else None,
        "core_IDs_present": "id, ctr, ctr_n, yr, mn, pty, pty_n",
        "suspected_duplicates": "many rows per election id; aggregate required",
        "key_variables": "pv1/pvs1/seat",
        "notes": f"rows={len(clea)} cols={len(clea.columns)}",
    })

    # V-Party
    vparty = read_vparty(raw_root / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv")
    rows.append({
        "source_id": "vparty_v2_party",
        "file": str(raw_root / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv"),
        "format": "CSV",
        "unit_of_observation": "Party × year",
        "time_coverage": _year_range(vparty.get("year")),
        "geo_coverage": f"countries={vparty['country_name'].nunique()}" if "country_name" in vparty else None,
        "core_IDs_present": "v2paid, pf_party_id, COWcode, year",
        "suspected_duplicates": "unique on (v2paid,year) expected",
        "key_variables": "v2pariglef",
        "notes": f"rows={len(vparty)} cols={len(vparty.columns)}",
    })

    # External key datasets (prefer repo-level PartyFacts files if present)
    pf_core_path = (repo_root / "partyfacts-core-parties.csv")
    if not pf_core_path.exists():
        pf_core_path = external_root / "partyfacts_core_parties.csv"
    pf_core = pd.read_csv(pf_core_path)
    rows.append({
        "source_id": "partyfacts_core",
        "file": str(pf_core_path),
        "format": "CSV",
        "unit_of_observation": "Party",
        "time_coverage": None,
        "geo_coverage": f"iso3={pf_core['country'].nunique()}" if "country" in pf_core else None,
        "core_IDs_present": "partyfacts_id, country",
        "suspected_duplicates": "partyfacts_id unique expected",
        "key_variables": "party names, year_first/last",
        "notes": f"rows={len(pf_core)} cols={len(pf_core.columns)}",
    })

    pf_external_path = (repo_root / "partyfacts-external-parties.csv")
    if not pf_external_path.exists():
        pf_external_path = external_root / "partyfacts_external_parties.csv"
    pf_external = pd.read_csv(pf_external_path)
    rows.append({
        "source_id": "partyfacts_external",
        "file": str(pf_external_path),
        "format": "CSV",
        "unit_of_observation": "Crosswalk to PartyFacts",
        "time_coverage": None,
        "geo_coverage": f"iso3={pf_external['country'].nunique()}" if "country" in pf_external else None,
        "core_IDs_present": "dataset_key, dataset_party_id, partyfacts_id",
        "suspected_duplicates": "many-to-one mappings possible",
        "key_variables": "name, name_english, name_short",
        "notes": f"rows={len(pf_external)} cols={len(pf_external.columns)}",
    })

    # cow2iso crosswalk (if present)
    cow2iso_path = repo_root / "cow2iso.csv"
    if cow2iso_path.exists():
        cow2iso = read_cow2iso(cow2iso_path)
        rows.append({
            "source_id": "cow2iso_crosswalk",
            "file": str(cow2iso_path),
            "format": "CSV",
            "unit_of_observation": "COW code × ISO3 mapping (time-bounded)",
            "time_coverage": _year_range(cow2iso.get("valid_from")) if "valid_from" in cow2iso.columns else None,
            "geo_coverage": f"iso3={cow2iso['iso3'].nunique()}" if "iso3" in cow2iso else None,
            "core_IDs_present": "cow_id, iso3, valid_from, valid_until",
            "suspected_duplicates": "multiple iso3 per cow_id across time",
            "key_variables": "cow_id, iso3, valid_from, valid_until, statenme",
            "notes": f"rows={len(cow2iso)} cols={len(cow2iso.columns)}",
        })

    # EFW panel dataset (if present)
    efw_candidates: list[Path] = []
    for root in [repo_root, external_root, raw_root, repo_root / "data"]:
        if root.exists():
            efw_candidates += sorted(root.rglob("*efw*.xls*"))
            efw_candidates += sorted(root.rglob("*EFW*.xls*"))
    seen = set()
    efw_candidates = [p for p in efw_candidates if not (p in seen or seen.add(p))]
    if efw_candidates:
        efw_path = efw_candidates[0]
        efw = pd.read_excel(efw_path, sheet_name="EFW Panel Dataset")
        rows.append({
            "source_id": "efw_panel",
            "file": str(efw_path),
            "format": "Excel",
            "unit_of_observation": "Country × year",
            "time_coverage": _year_range(efw.get("Year")),
            "geo_coverage": f"countries={efw['Countries'].nunique()}" if "Countries" in efw else None,
            "core_IDs_present": "ISO_Code, Countries, Year",
            "suspected_duplicates": "unique on (ISO_Code,Year) expected",
            "key_variables": "Summary, Area 1-5",
            "notes": f"rows={len(efw)} cols={len(efw.columns)}",
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
