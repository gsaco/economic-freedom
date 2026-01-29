from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elections.io import read_nelda, read_efw_panel

DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_INT = ROOT / "data" / "interim"
DATA_OUT = ROOT / "data" / "processed"
AUDIT = ROOT / "reports" / "audit"

rows: list[dict] = []


def add(
    step: str,
    step_type: str,
    left_name: str,
    left_obs: int | None,
    left_grain: str,
    right_name: str,
    right_obs: int | None,
    right_grain: str,
    join_type: str,
    join_keys: str,
    matched_obs: int | None,
    match_definition: str,
    output_table: str,
    notes: str = "",
) -> None:
    left_obs_i = int(left_obs) if left_obs is not None else None
    matched_obs_i = int(matched_obs) if matched_obs is not None else None
    match_rate_left = None
    left_unmatched = None
    if left_obs_i is not None and matched_obs_i is not None and left_obs_i > 0:
        match_rate_left = matched_obs_i / left_obs_i
        if matched_obs_i <= left_obs_i:
            left_unmatched = left_obs_i - matched_obs_i
    rows.append({
        "step": step,
        "step_type": step_type,
        "left_dataset": left_name,
        "left_obs": left_obs_i,
        "left_grain": left_grain,
        "right_dataset": right_name,
        "right_obs": int(right_obs) if right_obs is not None else None,
        "right_grain": right_grain,
        "join_type": join_type,
        "join_keys": join_keys,
        "matched_obs": matched_obs_i,
        "match_rate_left": match_rate_left,
        "left_unmatched": left_unmatched,
        "match_definition": match_definition,
        "output_table": output_table,
        "notes": notes,
    })


def _count_columns(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            return len(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            df = pd.read_parquet(path)
            return len(df.columns)
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=1)
        return len(df.columns)
    if path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=1)
        return len(df.columns)
    return None


# Load key tables
ned_pres = pd.read_parquet(DATA_INT / "elections_ned_pres.parquet")
ned_parl = pd.read_parquet(DATA_INT / "elections_ned_parl.parquet")
ned_union = pd.read_parquet(DATA_INT / "election_records_ned.parquet")
clea = pd.read_parquet(DATA_INT / "elections_clea_lc.parquet")

sources_base = pd.read_parquet(DATA_INT / "election_event_sources_base.parquet")
sources = pd.read_parquet(DATA_OUT / "election_event_sources.parquet")
canonical = pd.read_parquet(DATA_OUT / "election_event.parquet")
horizon = pd.read_parquet(DATA_OUT / "election_event_horizon_efw.parquet")

# APPENDS
add(
    step="Append: NED pres + NED parl",
    step_type="append",
    left_name="NED presidential elections",
    left_obs=len(ned_pres),
    left_grain="election record",
    right_name="NED parliamentary elections",
    right_obs=len(ned_parl),
    right_grain="election record",
    join_type="row_bind",
    join_keys="N/A",
    matched_obs=len(ned_union),
    match_definition="Row-bind union; matched_obs is union row count",
    output_table="data/interim/election_records_ned.parquet",
)

ned_iso_count = sources_base[sources_base["source"].isin(["ned_pres", "ned_parl"])].shape[0]
clea_iso_count = sources_base[sources_base["source"] == "clea_lc"].shape[0]
add(
    step="Append: NED + CLEA (source records)",
    step_type="append",
    left_name="NED elections (ISO3-mapped)",
    left_obs=ned_iso_count,
    left_grain="election record",
    right_name="CLEA LC elections (ISO3-mapped)",
    right_obs=clea_iso_count,
    right_grain="election record",
    join_type="row_bind",
    join_keys="N/A",
    matched_obs=len(sources_base),
    match_definition="Row-bind union; matched_obs is combined source records",
    output_table="data/interim/election_event_sources_base.parquet",
)

# ISO mapping for NED
ned_left = sources_base[sources_base["source"].isin(["ned_pres", "ned_parl"])]
add(
    step="ISO3 mapping (NED via cow2iso + name map)",
    step_type="merge",
    left_name="NED elections (pres + parl)",
    left_obs=len(ned_left),
    left_grain="election record",
    right_name="cow2iso crosswalk + ISO name map",
    right_obs=(pd.read_csv(ROOT / "cow2iso.csv")["cow_id"].notna().sum() + pd.read_csv(AUDIT / "04_country_mapping_ned.csv").shape[0]),
    right_grain="country crosswalk row",
    join_type="left",
    join_keys="country_cow → cow2iso.cow_id; country_name → ISO name map",
    matched_obs=ned_left["iso3"].notna().sum(),
    match_definition="iso3 non-null after cow2iso/name mapping",
    output_table="data/interim/election_event_sources_base.parquet",
)

# ISO mapping for CLEA
clea_left = sources_base[sources_base["source"] == "clea_lc"]
add(
    step="ISO3 mapping (CLEA via ctr + name map)",
    step_type="merge",
    left_name="CLEA LC elections (aggregated)",
    left_obs=len(clea_left),
    left_grain="election record",
    right_name="ISO name map for CLEA",
    right_obs=pd.read_csv(AUDIT / "04_country_mapping_clea.csv").shape[0],
    right_grain="country crosswalk row",
    join_type="left",
    join_keys="ctr (ISO3 if valid) or ctr_n → ISO name map",
    matched_obs=clea_left["iso3"].notna().sum(),
    match_definition="iso3 non-null after ctr/name mapping",
    output_table="data/interim/election_event_sources_base.parquet",
)

# NELDA merge
nelda_raw = read_nelda(DATA_RAW / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta")
nelda_report = pd.read_csv(AUDIT / "09_nelda_match_report.csv")
matched_nelda = nelda_report["nelda_match_status"].isin(["matched", "matched_outside_tolerance"]).sum()
add(
    step="NELDA merge (competitiveness flags)",
    step_type="merge",
    left_name="Election source records (NED + CLEA)",
    left_obs=len(sources_base),
    left_grain="election record",
    right_name="NELDA 6.0 elections",
    right_obs=len(nelda_raw),
    right_grain="election event",
    join_type="left",
    join_keys="cow_code + election_year + office_type (date tolerance)",
    matched_obs=matched_nelda,
    match_definition="matched or matched_outside_tolerance in nelda_match_report",
    output_table="reports/audit/09_nelda_match_report.csv",
)

# PartyFacts matching
party_match = pd.read_parquet(AUDIT / "party_match_results.parquet")
pf_core = pd.read_csv(ROOT / "partyfacts-core-parties.csv")
pf_ext = pd.read_csv(ROOT / "partyfacts-external-parties.csv")
matched_party = party_match["match_status"].isin(["matched", "matched_tiebreak"]).sum()
add(
    step="PartyFacts matching (party names → partyfacts_id)",
    step_type="match",
    left_name="Unique party strings (from NED/CLEA top-two)",
    left_obs=len(party_match),
    left_grain="party string",
    right_name="PartyFacts core + external",
    right_obs=len(pf_core) + len(pf_ext),
    right_grain="party",
    join_type="fuzzy",
    join_keys="iso3 + election_year window + party_name",
    matched_obs=matched_party,
    match_definition="match_status in {matched, matched_tiebreak}",
    output_table="reports/audit/party_match_results.parquet",
)

# Ideology assignment
ideology_obs = pd.read_parquet(DATA_OUT / "party_ideology_obs.parquet")
matched_ideo = sources["ideo_source"].notna().sum()
add(
    step="Ideology assignment (party IDs → ideology)",
    step_type="merge",
    left_name="Election source records with party IDs",
    left_obs=len(sources),
    left_grain="election record",
    right_name="Party ideology observations (V-Party/CHES/ParlGov/ELFF)",
    right_obs=len(ideology_obs),
    right_grain="party-year observation",
    join_type="left",
    join_keys="partyfacts_id + nearest year within window; same-source-per-election",
    matched_obs=matched_ideo,
    match_definition="ideo_source non-null",
    output_table="data/processed/election_event_sources.parquet",
)

# Deduplication
add(
    step="Deduplication (source records → canonical events)",
    step_type="dedupe",
    left_name="Election source records",
    left_obs=len(sources),
    left_grain="election record",
    right_name="N/A (grouping only)",
    right_obs=None,
    right_grain="N/A",
    join_type="groupby",
    join_keys="iso3|office_type|date_key (day/month/year precision)",
    matched_obs=len(canonical),
    match_definition="preferred record per merge_key",
    output_table="data/processed/election_event.parquet",
)

# Controls merges
if "dpi_match_status" in canonical.columns:
    dpi = pd.read_csv(DATA_EXT / "dpi" / "DPI2020" / "dpi2020.csv", low_memory=False)
    add(
        step="Controls merge (DPI)",
        step_type="merge",
        left_name="Canonical elections",
        left_obs=len(canonical),
        left_grain="election event",
        right_name="DPI 2020 (country-year)",
        right_obs=len(dpi),
        right_grain="country-year",
        join_type="left",
        join_keys="iso3 + baseline_year (t-1)",
        matched_obs=(canonical["dpi_match_status"] == "matched").sum(),
        match_definition="dpi_match_status==matched",
        output_table="data/processed/election_event.parquet",
    )

if "idea_match_status" in canonical.columns:
    idea = pd.read_excel(DATA_EXT / "idea_export_electoral_system_design_database.xlsx")
    add(
        step="Controls merge (IDEA ESD)",
        step_type="merge",
        left_name="Canonical elections",
        left_obs=len(canonical),
        left_grain="election event",
        right_name="IDEA ESD (country-year)",
        right_obs=len(idea),
        right_grain="country-year",
        join_type="left",
        join_keys="iso3 + election_year",
        matched_obs=(canonical["idea_match_status"] == "matched").sum(),
        match_definition="idea_match_status==matched",
        output_table="data/processed/election_event.parquet",
    )

if "des_match_status" in canonical.columns:
    des_rows = None
    try:
        import zipfile
        with zipfile.ZipFile(DATA_EXT / "des_es_data_v50.zip") as z:
            with z.open("es_data-v5_0.csv") as f:
                des_rows = sum(1 for _ in f) - 1
    except Exception:
        des_rows = None
    add(
        step="Controls merge (DES electoral system)",
        step_type="merge",
        left_name="Canonical elections",
        left_obs=len(canonical),
        left_grain="election event",
        right_name="DES v5.0 (election-level)",
        right_obs=des_rows,
        right_grain="election event",
        join_type="left",
        join_keys="iso3 + election_year (date tolerance)",
        matched_obs=(canonical["des_match_status"] == "matched").sum(),
        match_definition="des_match_status==matched",
        output_table="data/processed/election_event.parquet",
    )

# EFW baseline
if "efw_match_status" in canonical.columns:
    efw = read_efw_panel(DATA_RAW / "efw" / "efotw-2025-master-index-data-for-researchers-iso.xlsx")
    add(
        step="EFW baseline merge",
        step_type="merge",
        left_name="Canonical elections",
        left_obs=len(canonical),
        left_grain="election event",
        right_name="EFW panel (iso3-year)",
        right_obs=len(efw),
        right_grain="country-year",
        join_type="left",
        join_keys="iso3 + baseline_year (t-1)",
        matched_obs=(canonical["efw_match_status"] == "matched").sum(),
        match_definition="efw_match_status==matched",
        output_table="data/processed/election_event.parquet",
    )

# EFW horizons (long)
if "efw_horizon_match_status" in horizon.columns:
    add(
        step="EFW horizons build",
        step_type="expand",
        left_name="Canonical elections",
        left_obs=len(canonical),
        left_grain="election event",
        right_name="EFW panel (iso3-year)",
        right_obs=len(efw) if "efw" in locals() else None,
        right_grain="country-year",
        join_type="expand+left",
        join_keys="iso3 + outcome_year (t+h)",
        matched_obs=(horizon["efw_horizon_match_status"] == "matched").sum(),
        match_definition="efw_horizon_match_status==matched",
        output_table="data/processed/election_event_horizon_efw.parquet",
    )

out = pd.DataFrame(rows)
def _resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path

out["output_cols"] = out["output_table"].apply(lambda p: _count_columns(_resolve_path(p)) if _resolve_path(p) else None)
out_path = AUDIT / "merge_step_table.csv"
out.to_csv(out_path, index=False)
print(out)
print(f"\nWrote {out_path}")
