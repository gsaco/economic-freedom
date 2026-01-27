from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "elections_final_clean_2000_tidy.csv"
PROFILE_PATH = ROOT / "reports" / "elections_profile_2000_tidy.json"
OUT_TEX = ROOT / "data_dictionary.tex"
APPENDIX_TABLE = ROOT / "reports" / "data_dictionary_profile_table.tex"


EXPECTED_COLUMNS = [
    "country",
    "year",
    "type_election",
    "date",
    "source",
    "top1_party",
    "top2_party",
    "party_3",
    "flag_inconsequential_note",
    "country_abb",
    "cow_code",
    "office_type",
    "top1_share",
    "top2_share",
    "top1_margin",
    "cowcode",
    "iso3",
    "party1_v2paid",
    "party1_match_score",
    "party2_v2paid",
    "party2_match_score",
    "ideo_party1",
    "ideo_party2",
    "margin_market",
    "D_market_win",
    "ccode",
    "nelda_date",
    "nelda_opposition_allowed",
    "nelda_multiparty_legal",
    "nelda_candidate_choice",
    "ideo_year_diff1",
    "ideo_year_diff2",
    "ideo_quality1",
    "ideo_quality2",
    "ideo_quality",
    "margin_abs",
    "ideo_gap",
    "election_key",
    "clean_flag",
    "competitive_flag",
    "competitive_sample",
    "efw_coverage",
    "partyfacts_id1",
    "partyfacts_id2",
    "ches_score",
    "elff_score",
    "parlgov_score",
    "des_country_norm_x",
    "des_score",
    "dpi_country_norm",
    "dpi_score",
    "party1_norm",
    "party2_norm",
    "top1_ideology",
    "top1_ideology_source",
    "top2_ideology",
    "top2_ideology_source",
    "margin_market_ext",
    "D_market_win_ext",
    "des_country_norm_y",
    "des_electoral_rule",
    "des_tier1_formula",
    "des_tier1_avg_magnitude",
    "des_date_diff",
    "des_match",
    "des_match_2y",
    "des_match_5y",
    "idea_system_family",
    "idea_legislative_system",
    "idea_presidential_system",
    "idea_tiers",
    "idea_leg_size_direct",
    "idea_leg_size_voting",
    "idea_match",
    "country_norm",
    "dpi_system",
    "dpi_execme",
    "dpi_execrlc",
    "dpi_gov1rlc",
    "dpi_checks",
    "dpi_checks_lax",
    "dpi_military",
    "dpi_match",
    "market_margin",
    "market_win",
    "analysis_ready",
]


def latex_escape(text: str) -> str:
    if text is None:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = ""
    for ch in str(text):
        out += replacements.get(ch, ch)
    return out


def format_num(val: float | int | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    return f"\\num{{{val:.4g}}}"


def is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique().tolist()
    if not vals:
        return False
    allowed = {0, 1, True, False}
    return all(v in allowed for v in vals)


def load_labels() -> dict:
    labels = {}
    pres_path = ROOT / "data" / "raw" / "ned" / "presidential_elections_v2.dta"
    parl_path = ROOT / "data" / "raw" / "ned" / "parliamentary_elections_v2.dta"
    if pres_path.exists():
        _, meta = pyreadstat.read_dta(pres_path, metadataonly=True)
        labels.update(dict(zip(meta.column_names, meta.column_labels)))
    if parl_path.exists():
        _, meta = pyreadstat.read_dta(parl_path, metadataonly=True)
        for k, v in zip(meta.column_names, meta.column_labels):
            labels.setdefault(k, v)
    nelda_path = ROOT / "data" / "raw" / "nelda" / "NELDA 6.0" / "id & q-wide_share.dta"
    if nelda_path.exists():
        _, meta = pyreadstat.read_dta(nelda_path, metadataonly=True)
        labels.update(dict(zip(meta.column_names, meta.column_labels)))
    return labels


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing {PROFILE_PATH}. Run scripts/profile_dataset.py first.")
    return json.loads(PROFILE_PATH.read_text())


def domain_text(col: str, prof: dict, series: pd.Series) -> str:
    inferred = prof.get("inferred_type")
    if col == "date":
        dates = pd.to_datetime(series, errors="coerce")
        if dates.dropna().empty:
            return "No non-missing dates."
        return f"Min date {latex_escape(str(dates.min().date()))}; max date {latex_escape(str(dates.max().date()))}."

    if inferred in {"int", "float"} and not is_binary(series):
        stats = prof.get("numeric_stats", {})
        if not stats:
            return "No non-missing numeric values."
        return (
            f"min {format_num(stats.get('min'))}, max {format_num(stats.get('max'))}, "
            f"mean {format_num(stats.get('mean'))}, sd {format_num(stats.get('std'))}, "
            f"p1 {format_num(stats.get('p1'))}, p50 {format_num(stats.get('p50'))}, p99 {format_num(stats.get('p99'))}."
        )

    cat = prof.get("categorical_stats", {})
    levels = cat.get("levels", [])
    unique_count = cat.get("unique_count", 0)
    if not levels:
        return "No non-missing values."
    parts = []
    for lvl in levels:
        val = latex_escape(lvl["value"])
        parts.append(f"{val} ({lvl['count']}, {lvl['pct']:.2f}\\%)")
    if unique_count > len(levels):
        return "Top levels: " + "; ".join(parts) + f". Total unique: {unique_count}."
    return "Levels: " + "; ".join(parts) + "."


def missing_text(prof: dict, n_rows: int) -> str:
    miss = prof["missing_count"]
    pct = prof["missing_pct"]
    return f"{miss} / {n_rows} ({pct:.2f}\\%)"


def value_type_text(col: str, prof: dict, series: pd.Series) -> str:
    inferred = prof.get("inferred_type")
    if col == "date":
        return "date (ISO-8601 string in CSV; parsed as calendar date)."
    if is_binary(series):
        return "binary indicator encoded as 0/1."
    if inferred == "int":
        return "integer (pandas Int64)."
    if inferred == "float":
        return "float (numeric)."
    return "string / categorical."


def meta_for(col: str, labels: dict) -> dict:
    src_ned = "elections_core.py (build_elections_ned)"
    src_clea = "elections_core.py (build_elections_clea)"
    src_master = "elections_core.py (build_master/build_master_outputs)"
    src_rename = "legacy tidy step (removed)"
    src_utils = "elections_core.py (normalize_name/best_fuzzy_match)"
    src_external = "elections_external.py (build_external_master)"
    src_final = "elections_external.py (build_final)"
    src_des = "elections_external.py (attach_des)"
    src_idea = "elections_external.py (load_idea)"
    src_dpi = "elections_external.py (load_dpi)"

    if col == "country":
        return {
            "label": "Country name (source label)",
            "meaning": f"Country name as reported in the source election dataset. NED label: {labels.get('country','')}.",
            "construction": "Carried through from NED/CLEA without transformation.",
            "source": f"NED/CLEA raw files; see {src_ned} and {src_clea}. NED dataset: \\cite{{NEDv2}}; CLEA dataset: \\cite{{CLEA2025}}.",
            "notes": "Missing for CLEA-only rows if the raw file does not include a country name.",
        }
    if col == "year":
        return {
            "label": "Election year",
            "meaning": "Calendar year of the election.",
            "construction": "From NED/CLEA date fields or year variables; CLEA year is set from `yr`.",
            "source": f"NED/CLEA raw files; see {src_ned} and {src_clea}.",
            "notes": "Used for joins to ideology and external datasets.",
        }
    if col == "type_election":
        return {
            "label": "Election type (NED label)",
            "meaning": f"NED election-type descriptor. NED label: {labels.get('type_election','')}.",
            "construction": "Carried through from NED; not constructed for CLEA.",
            "source": f"NED raw file; see {src_ned}. NED dataset: \\cite{{NEDv2}}.",
            "notes": "May be missing for CLEA-derived rows.",
        }
    if col == "date":
        return {
            "label": "Election date",
            "meaning": "Election date from NED or constructed for CLEA.",
            "construction": "NED uses its reported date. CLEA uses year-month with day=01 (see script).",
            "source": f"NED/CLEA raw files; see {src_ned} and {src_clea}.",
            "notes": "Used for DES matching and election key construction.",
        }
    if col == "source":
        return {
            "label": "Record source",
            "meaning": "Originating dataset for the election record (NED or CLEA).",
            "construction": "Assigned in master build.",
            "source": f"{src_master}.",
            "notes": "Used to trace provenance of each row.",
        }
    if col in {"top1_party", "top2_party"}:
        rank = "1" if col == "top1_party" else "2"
        return {
            "label": f"Top-{rank} party name",
            "meaning": f"Party name for the {rank}st/2nd ranked competitor by share (after top-two ordering).",
            "construction": "Derived from NED party_1/party_2 or CLEA aggregated party names, then renamed in the tidy step.",
            "source": f"{src_ned}; {src_clea}; rename in {src_rename}.",
            "notes": "Top-two ordering is enforced so share_1 ≥ share_2 before renaming.",
        }
    if col == "party_3":
        return {
            "label": "Third party name",
            "meaning": f"{labels.get('party_3','Third party/coalition')} (NED label).",
            "construction": "Carried through from NED raw files.",
            "source": f"NED raw file; see {src_ned}. NED dataset: \\cite{{NEDv2}}.",
            "notes": "Typically missing for CLEA rows.",
        }
    if col == "flag_inconsequential_note":
        return {
            "label": "Inconsequential flag note",
            "meaning": f"{labels.get('flag_inconsequential_note','Reason for inconsequential flag')} (NED label).",
            "construction": "Carried through from NED raw files.",
            "source": f"NED raw file; see {src_ned}. NED dataset: \\cite{{NEDv2}}.",
            "notes": "Only populated when NED `flag_inconsequential` is 1.",
        }
    if col == "country_abb":
        return {
            "label": "Country abbreviation",
            "meaning": f"{labels.get('country_abb','Country code')} (NED label).",
            "construction": "Carried through from NED raw files.",
            "source": f"NED raw file; see {src_ned}. NED dataset: \\cite{{NEDv2}}.",
            "notes": "May be missing for CLEA rows.",
        }
    if col == "cow_code":
        return {
            "label": "COW code (NED)",
            "meaning": f"{labels.get('country_cow','COW country code')} (NED label).",
            "construction": "Renamed from NED `country_cow` in the tidy step.",
            "source": f"NED raw file; rename in {src_rename}. NED dataset: \\cite{{NEDv2}}.",
            "notes": "Numeric COW code from NED.",
        }
    if col == "office_type":
        return {
            "label": "Office type",
            "meaning": "Standardized office type for the election (presidential/parliamentary).",
            "construction": "Assigned during NED/CLEA processing.",
            "source": f"{src_ned}; {src_clea}.",
            "notes": "Used in election matching and RD specifications.",
        }
    if col in {"top1_share", "top2_share"}:
        rank = "1" if col == "top1_share" else "2"
        return {
            "label": f"Top-{rank} share",
            "meaning": f"Vote/seat share of the {rank}st/2nd ranked competitor.",
            "construction": "From NED: presidential final-round vote shares; parliamentary seat shares. From CLEA: vote share if available, otherwise seat share.",
            "source": f"{src_ned}; {src_clea}; rename in {src_rename}.",
            "notes": "Shares are expressed in percentage points (0–100).",
        }
    if col == "top1_margin":
        return {
            "label": "Top-1 minus top-2 margin",
            "meaning": "Difference between top1 and top2 shares.",
            "construction": "Computed as share_1 − share_2 in NED/CLEA processing; renamed in tidy step.",
            "source": f"{src_ned}; {src_clea}; rename in {src_rename}.",
            "notes": "Positive by construction after top-two ordering.",
        }
    if col == "cowcode":
        return {
            "label": "COW code (crosswalk)",
            "meaning": "COW country code from the ISO3↔COW crosswalk used for mapping.",
            "construction": "Merged from country crosswalk in NED/CLEA build.",
            "source": f"{src_ned}; {src_clea}.",
            "notes": "May match `cow_code` but originates from the crosswalk file.",
        }
    if col == "iso3":
        return {
            "label": "ISO3 country code",
            "meaning": "ISO3 code assigned to each election record.",
            "construction": "Mapped from COW code and country names during NED/CLEA processing.",
            "source": f"{src_ned}; {src_clea}.",
            "notes": "Used for merges to EFW and external datasets.",
        }
    if col in {"party1_v2paid", "party2_v2paid"}:
        rank = "1" if col == "party1_v2paid" else "2"
        return {
            "label": f"V-Party ID (party {rank})",
            "meaning": "V-Party party identifier matched to the top-two party name.",
            "construction": "Fuzzy match of party names to V-Party units; stored as v2paid.",
            "source": f"{src_ned}; {src_clea}; matching in {src_master}. V-Party dataset: \\cite{{VPartyV2}}.",
            "notes": f"Match quality tracked in `party{rank}_match_score`.",
        }
    if col in {"party1_match_score", "party2_match_score"}:
        rank = "1" if col == "party1_match_score" else "2"
        return {
            "label": f"Party-name match score ({rank})",
            "meaning": "Fuzzy match score (0–100) for mapping party name to V-Party.",
            "construction": "Computed by `best_fuzzy_match` during party matching.",
            "source": f"{src_ned}; {src_clea}; matcher in {src_utils}.",
            "notes": "Higher scores indicate closer name matches.",
        }
    if col in {"ideo_party1", "ideo_party2"}:
        return {
            "label": "V-Party ideology",
            "meaning": "Economic left-right score (V-Party `v2pariglef`) for the matched party.",
            "construction": "Nearest-year V-Party score selected for the matched `v2paid`.",
            "source": f"{src_master}. V-Party dataset: \\cite{{VPartyV2}}.",
            "notes": "Scale is 0–6 in V-Party; higher values = more market-oriented.",
        }
    if col in {"margin_market", "D_market_win"}:
        return {
            "label": "Ideology-based margin / win",
            "meaning": "Margin oriented so positive values indicate a more-market win; indicator is 1 if margin ≥ 0.",
            "construction": "If ideo_party1 ≥ ideo_party2 then margin_market = margin; else margin_market = −margin. Indicator defined as margin_market ≥ 0.",
            "source": f"{src_master} (compute_market_margin).",
            "notes": "Requires non-missing V-Party ideology scores.",
        }
    if col in {"ccode", "nelda_date", "nelda_opposition_allowed", "nelda_multiparty_legal", "nelda_candidate_choice"}:
        label_map = {
            "ccode": "NELDA COW code",
            "nelda_date": "NELDA election date",
            "nelda_opposition_allowed": "Opposition allowed",
            "nelda_multiparty_legal": "Multiple parties legal",
            "nelda_candidate_choice": "Choice of candidates",
        }
        meaning_map = {
            "ccode": "COW country code from NELDA used for merge.",
            "nelda_date": "Date parsed from NELDA year + mmdd (YYYYMMDD).",
            "nelda_opposition_allowed": f"{labels.get('nelda3','Opposition allowed?')} (NELDA 3).",
            "nelda_multiparty_legal": f"{labels.get('nelda4','More than one party legal?')} (NELDA 4).",
            "nelda_candidate_choice": f"{labels.get('nelda5','Choice of candidates?')} (NELDA 5).",
        }
        return {
            "label": label_map[col],
            "meaning": meaning_map[col],
            "construction": "Merged from NELDA by ccode/year/office_type; nelda_date built from year+mmdd.",
            "source": "elections_core.py (build_elections_ned/build_master). NELDA dataset/codebook: \\cite{NELDAv6}.",
            "notes": "NELDA indicators are coded 1/0; missing if no match.",
        }
    if col in {"ideo_year_diff1", "ideo_year_diff2"}:
        return {
            "label": "Ideology year distance",
            "meaning": "Absolute year difference between election year and the V-Party year used for ideology.",
            "construction": "Computed in get_party_ideology as |year_vparty − election_year|.",
            "source": f"{src_master}.",
            "notes": "Used to assess ideology match quality.",
        }
    if col in {"ideo_quality1", "ideo_quality2", "ideo_quality"}:
        if col == "ideo_quality":
            meaning = "Overall ideology match quality, taking the lower-quality of the two top parties."
            construction = "Min-quality based on party1/party2 quality rank."
        else:
            meaning = "Ideology match quality (high/medium/low/missing) for the top party."
            construction = "High if match_score ≥85 and year_diff ≤2; Medium if score ≥75 and year_diff ≤4; Low otherwise."
        return {
            "label": "Ideology match quality",
            "meaning": meaning,
            "construction": construction,
            "source": f"{src_master} (ideology_quality function).",
            "notes": "Qualitative diagnostic of party-ideology mapping.",
        }
    if col == "margin_abs":
        return {
            "label": "Absolute margin",
            "meaning": "Absolute value of the top-two share margin.",
            "construction": "Computed as abs(margin).",
            "source": f"{src_master} (compute_market_margin).",
            "notes": "Useful for bandwidth selection and density tests.",
        }
    if col == "ideo_gap":
        return {
            "label": "Ideology gap",
            "meaning": "Difference between party 1 and party 2 V-Party ideology scores.",
            "construction": "Computed as ideo_party1 − ideo_party2.",
            "source": f"{src_master} (compute_market_margin).",
            "notes": "Positive values indicate party 1 is more market-oriented.",
        }
    if col == "election_key":
        return {
            "label": "Election key",
            "meaning": "Unique string key for deduplication across sources.",
            "construction": "Concatenation of iso3, office_type, and date (or year-month).",
            "source": f"{src_master} (key_for_row).",
            "notes": "Used to drop duplicate elections with NED priority.",
        }
    if col == "clean_flag":
        return {
            "label": "Core clean flag",
            "meaning": "Basic validity flag for key fields and share ordering.",
            "construction": "True if iso3/year/share_1/share_2 non-missing and share_1 ≥ share_2.",
            "source": f"{src_master}.",
            "notes": "Baseline for analysis-ready sample.",
        }
    if col == "competitive_flag":
        return {
            "label": "NELDA competitive flag",
            "meaning": "Indicator that NELDA competition conditions are satisfied.",
            "construction": "nelda3==1 & nelda4==1 & nelda5==1 (NA if NELDA missing).",
            "source": f"{src_master}.",
            "notes": "Used in competitive sample definition.",
        }
    if col == "competitive_sample":
        return {
            "label": "Competitive sample flag",
            "meaning": "Indicator for baseline competitive-election sample.",
            "construction": "clean_flag & quality_ok & (nelda_ok if available).",
            "source": f"{src_master}.",
            "notes": "quality_ok excludes coups/inconsequential/unopposed/indirect elections.",
        }
    if col == "efw_coverage":
        return {
            "label": "EFW coverage flag",
            "meaning": "Whether iso3 is present in the EFW dataset.",
            "construction": "iso3 ∈ EFW ISO list.",
            "source": f"{src_master}. EFW dataset: \\cite{{EFW2025}}.",
            "notes": "Used to define coverage for institutional outcomes.",
        }
    if col in {"partyfacts_id1", "partyfacts_id2"}:
        return {
            "label": "PartyFacts ID",
            "meaning": "PartyFacts unique party identifier mapped from V-Party party IDs.",
            "construction": "Mapped via PartyFacts external-parties table (vparty → partyfacts).",
            "source": f"{src_external}. PartyFacts dataset: \\cite{{PartyFacts}}.",
            "notes": "Used to link CHES party-level ideology where available.",
        }
    if col in {"ches_score", "elff_score", "parlgov_score", "des_score", "dpi_score"}:
        src = col.split("_")[0].upper()
        return {
            "label": f"{src} country match score",
            "meaning": "Fuzzy match score (0–100) for mapping ISO3 country names to external dataset country names.",
            "construction": "Computed by best_fuzzy_match in country map construction.",
            "source": f"{src_external}; matcher in {src_utils}.",
            "notes": "Higher scores indicate closer name matches.",
        }
    if col in {"des_country_norm_x", "des_country_norm_y"}:
        return {
            "label": "DES country name (normalized)",
            "meaning": "Normalized DES country name used in matching; suffixes reflect merge collisions.",
            "construction": "From fuzzy country map merges used in DES matching.",
            "source": f"{src_external}.",
            "notes": "Intermediate columns retained for auditability.",
        }
    if col == "dpi_country_norm":
        return {
            "label": "DPI country name (normalized)",
            "meaning": "Normalized country name for DPI mapping.",
            "construction": "From fuzzy country map matching ISO3 to DPI country names.",
            "source": f"{src_external}; DPI loader in {src_dpi}.",
            "notes": "Used as merge key with DPI country-year data.",
        }
    if col in {"party1_norm", "party2_norm"}:
        return {
            "label": "Normalized party name",
            "meaning": "Normalized party name used for fuzzy matching to CHES/ELFF/ParlGov.",
            "construction": "normalize_name applied to party_1_name/party_2_name.",
            "source": f"{src_external}; normalizer in {src_utils}.",
            "notes": "Lowercased, stripped of punctuation and accents.",
        }
    if col in {"top1_ideology", "top2_ideology"}:
        return {
            "label": "Final ideology score",
            "meaning": "Final market-orientation score used for RD, after prioritizing ideology sources.",
            "construction": "Priority: V-Party → ParlGov → CHES → ELFF; rescaled to 0–6 where needed.",
            "source": f"{src_external}; rename in {src_rename}.",
            "notes": "Source indicator stored in top1_ideology_source/top2_ideology_source.",
        }
    if col in {"top1_ideology_source", "top2_ideology_source"}:
        return {
            "label": "Ideology source tag",
            "meaning": "Source used for the final ideology score (vparty/parlgov/ches/elff/missing).",
            "construction": "Assigned during ideology priority selection.",
            "source": f"{src_external}; rename in {src_rename}.",
            "notes": "Use for robustness by ideology source.",
        }
    if col in {"margin_market_ext", "D_market_win_ext"}:
        return {
            "label": "External-priority margin / win",
            "meaning": "Ideology-based margin and win indicator using final ideology scores.",
            "construction": "Same formula as margin_market but using ideo_final1/2; indicator is margin_market_ext ≥ 0.",
            "source": f"{src_external}.",
            "notes": "Used when extending ideology sources beyond V-Party.",
        }
    if col in {"des_electoral_rule", "des_tier1_formula", "des_tier1_avg_magnitude", "des_date_diff", "des_match", "des_match_2y", "des_match_5y"}:
        return {
            "label": "DES electoral system field",
            "meaning": "Electoral-system metadata from the Database of Electoral Systems (DES).",
            "construction": "Matched by country and nearest date; des_date_diff is days between election date and DES record; match flags denote proximity.",
            "source": f"{src_des}. DES dataset: \\cite{{DESv50}}.",
            "notes": "des_match_2y/5y indicate matches within 2 or 5 years.",
        }
    if col in {"idea_system_family", "idea_legislative_system", "idea_presidential_system", "idea_tiers", "idea_leg_size_direct", "idea_leg_size_voting", "idea_match"}:
        return {
            "label": "IDEA ESD electoral system field",
            "meaning": "Electoral-system metadata from the IDEA Electoral System Design (ESD) database.",
            "construction": "Merged by iso3-year; idea_match indicates non-missing system family.",
            "source": f"{src_idea}. IDEA ESD dataset: \\cite{{IDEAESD}}.",
            "notes": "Coverage depends on IDEA reporting year.",
        }
    if col == "country_norm":
        return {
            "label": "Normalized DPI country name",
            "meaning": "Normalized country name from DPI dataset.",
            "construction": "normalize_name applied to DPI country_name.",
            "source": f"{src_dpi}. DPI dataset: \\cite{{DPI2020}}.",
            "notes": "Used as merge key for DPI country-year data.",
        }
    if col in {"dpi_system", "dpi_execme", "dpi_execrlc", "dpi_gov1rlc", "dpi_checks", "dpi_checks_lax", "dpi_military", "dpi_match"}:
        return {
            "label": "DPI institutional variable",
            "meaning": "Political institutions/control variable from DPI 2020.",
            "construction": "Merged by dpi_country_norm and year; dpi_match indicates non-missing dpi_system.",
            "source": f"{src_dpi}. DPI dataset: \\cite{{DPI2020}}.",
            "notes": "Interpretation follows DPI documentation in the bundled dictionary files.",
        }
    if col in {"market_margin", "market_win"}:
        return {
            "label": "Best-available market margin / win",
            "meaning": "Final RD running variable and treatment indicator using best available ideology sources.",
            "construction": "market_margin = margin_market_ext if available else margin_market; market_win = 1 if market_margin ≥ 0.",
            "source": f"{src_final}; rename in {src_rename}.",
            "notes": "Primary running variable for RD analysis.",
        }
    if col == "analysis_ready":
        return {
            "label": "Analysis-ready flag",
            "meaning": "Indicator that record passes core and competitive filters for analysis.",
            "construction": "Requires clean_flag, non-missing iso3/year/share_1/share_2/margin_market_best, and competitive_sample if available.",
            "source": f"{src_final}; rename in {src_rename}.",
            "notes": "Defines the default estimation sample.",
        }

    return {
        "label": "See source",
        "meaning": "Variable retained from upstream processing; see source/provenance.",
        "construction": "Carried through without modification in the tidy step.",
        "source": f"{src_master}; {src_rename}.",
        "notes": "See pipeline scripts for details.",
    }


def build_appendix_table(profile: dict) -> None:
    rows = []
    for col, p in profile["columns"].items():
        rows.append({
            "variable": col,
            "dtype": p.get("dtype"),
            "missing_pct": p.get("missing_pct"),
            "nunique": p.get("nunique"),
        })
    df = pd.DataFrame(rows)
    df["missing_pct"] = df["missing_pct"].map(lambda x: f"{x:.2f}")

    lines = []
    lines.append("\\begin{longtable}{llrr}")
    lines.append("\\toprule")
    lines.append("Variable & Dtype & Missing (\\%) & Unique \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append("Variable & Dtype & Missing (\\%) & Unique \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for _, row in df.iterrows():
        var = latex_escape(row["variable"])
        dtype = latex_escape(row["dtype"])
        lines.append(f"\\texttt{{{var}}} & {dtype} & {row['missing_pct']} & {int(row['nunique'])} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    APPENDIX_TABLE.write_text("\n".join(lines))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    profile = load_profile()
    labels = load_labels()

    cols = df.columns.tolist()
    if cols != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in cols]
        extra = [c for c in cols if c not in EXPECTED_COLUMNS]
        raise ValueError(f"Column mismatch. Missing: {missing}. Extra: {extra}.")
    if len(cols) != 86:
        raise ValueError(f"Expected 86 columns, found {len(cols)}.")

    n_rows, n_cols = df.shape
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    iso3_count = df["iso3"].nunique()

    build_appendix_table(profile)

    git_hash = "unknown"
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        pass

    tex = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{microtype}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{tabularx}")
    tex.append(r"\usepackage{enumitem}")
    tex.append(r"\providecommand{\IfDocumentMetadataT}[1]{#1}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage{siunitx}")
    tex.append(r"\sisetup{detect-weight=true,detect-family=true}")
    tex.append("")
    tex.append(r"\begin{filecontents*}{data_dictionary_refs.bib}")
    tex.append(r"@misc{NEDv2,")
    tex.append(r"  author = {National Elections Database},")
    tex.append(r"  year = {2025},")
    tex.append(r"  title = {Elections Database v2.0 (presidential and parliamentary)},")
    tex.append(r"  howpublished = {Local replication package (Stata files)},")
    tex.append(r"  url = {file:data/raw/ned/presidential_elections_v2.dta},")
    tex.append(r"  note = {Used for national election results and flags.}")
    tex.append(r"}")
    tex.append(r"@misc{CLEA2025,")
    tex.append(r"  author = {Constituency-Level Elections Archive (CLEA)},")
    tex.append(r"  year = {2025},")
    tex.append(r"  title = {CLEA lower-chamber election results},")
    tex.append(r"  howpublished = {Local replication package (SPSS file)},")
    tex.append(r"  url = {file:data/raw/clea/clea_lc_20251015.sav},")
    tex.append(r"  note = {Used for parliamentary election aggregation.}")
    tex.append(r"}")
    tex.append(r"@misc{VPartyV2,")
    tex.append(r"  author = {V-Dem Institute},")
    tex.append(r"  year = {2022},")
    tex.append(r"  title = {V-Dem V-Party Dataset v2},")
    tex.append(r"  howpublished = {Local replication package (CSV)},")
    tex.append(r"  url = {file:data/raw/vparty/CPD_V-Party_CSV_v2/V-Dem-CPD-Party-V2.csv},")
    tex.append(r"  note = {Economic left-right score v2pariglef.}")
    tex.append(r"}")
    tex.append(r"@misc{NELDAv6,")
    tex.append(r"  author = {NELDA Project},")
    tex.append(r"  year = {2011},")
    tex.append(r"  title = {National Elections across Democracy and Autocracy (NELDA) v6},")
    tex.append(r"  howpublished = {Local replication package (Stata + PDF codebook)},")
    tex.append(r"  url = {file:data/raw/nelda/NELDA 6.0/NELDA_Codebook_V6.pdf},")
    tex.append(r"  note = {Election competitiveness indicators nelda3--nelda5.}")
    tex.append(r"}")
    tex.append(r"@misc{EFW2025,")
    tex.append(r"  author = {Fraser Institute},")
    tex.append(r"  year = {2025},")
    tex.append(r"  title = {Economic Freedom of the World (EFW) dataset},")
    tex.append(r"  howpublished = {Local replication package (Excel)},")
    tex.append(r"  url = {file:data/raw/efw/efotw-2025-master-index-data-for-researchers-iso.xlsx},")
    tex.append(r"  note = {Used to define EFW coverage.}")
    tex.append(r"}")
    tex.append(r"@misc{PartyFacts,")
    tex.append(r"  author = {PartyFacts Project},")
    tex.append(r"  year = {n.d.},")
    tex.append(r"  title = {PartyFacts external parties table},")
    tex.append(r"  howpublished = {Downloaded CSV},")
    tex.append(r"  url = {https://partyfacts.herokuapp.com/download/external-parties-csv/},")
    tex.append(r"  note = {Used to map V-Party to CHES party IDs.}")
    tex.append(r"}")
    tex.append(r"@misc{CHES2024,")
    tex.append(r"  author = {Chapel Hill Expert Survey},")
    tex.append(r"  year = {2024},")
    tex.append(r"  title = {CHES 1999--2024 dataset (means)},")
    tex.append(r"  howpublished = {Downloaded CSV},")
    tex.append(r"  url = {https://www.chesdata.eu/s/1999-2024_CHES_dataset_means.csv},")
    tex.append(r"  note = {Economic left-right variable lrecon.}")
    tex.append(r"}")
    tex.append(r"@misc{ELFF2024,")
    tex.append(r"  author = {European Left-Right and Party Positions (ELFF)},")
    tex.append(r"  year = {2024},")
    tex.append(r"  title = {Party positions summary dataset},")
    tex.append(r"  howpublished = {Downloaded CSV},")
    tex.append(r"  url = {https://www.elff.eu/data/party-positions/partypos-summaries.csv},")
    tex.append(r"  note = {Economic left-right variable econlr.}")
    tex.append(r"}")
    tex.append(r"@misc{ParlGov2023,")
    tex.append(r"  author = {D\"oring, Holger and Manow, Philip},")
    tex.append(r"  year = {2023},")
    tex.append(r"  title = {ParlGov database},")
    tex.append(r"  howpublished = {Downloaded ZIP},")
    tex.append(r"  url = {https://www.parlgov.org/data/parlgov-development_csv-utf-8.zip},")
    tex.append(r"  note = {Used for party position state\_market.}")
    tex.append(r"}")
    tex.append(r"@misc{DESv50,")
    tex.append(r"  author = {Golder, Matt},")
    tex.append(r"  year = {2020},")
    tex.append(r"  title = {Database of Electoral Systems v5.0},")
    tex.append(r"  howpublished = {Downloaded ZIP},")
    tex.append(r"  url = {https://mattgolder.com/files/research/es_data-v50.zip},")
    tex.append(r"  note = {Electoral-system rules matched by date.}")
    tex.append(r"}")
    tex.append(r"@misc{IDEAESD,")
    tex.append(r"  author = {International IDEA},")
    tex.append(r"  year = {n.d.},")
    tex.append(r"  title = {Electoral System Design (ESD) database export},")
    tex.append(r"  howpublished = {Local replication package (Excel export)},")
    tex.append(r"  url = {file:data/external/idea_export_electoral_system_design_database.xlsx},")
    tex.append(r"  note = {Electoral system family and tiers.}")
    tex.append(r"}")
    tex.append(r"@misc{DPI2020,")
    tex.append(r"  author = {Database of Political Institutions (DPI)},")
    tex.append(r"  year = {2020},")
    tex.append(r"  title = {DPI 2020},")
    tex.append(r"  howpublished = {Local replication package (CSV + dictionary)},")
    tex.append(r"  url = {file:data/external/dpi/DPI2020/dpi2020.csv},")
    tex.append(r"  note = {Institutional controls: system, execme, checks, etc.}")
    tex.append(r"}")
    tex.append(r"\end{filecontents*}")

    tex.append(r"\title{Data Dictionary: elections\_final\_clean\_2000\_tidy.csv}")
    tex.append(r"\author{Data Dictionary Strike Team}")
    tex.append(f"\\date{{Build date: {date.today().isoformat()} \\\\ Git commit: {latex_escape(git_hash)}}}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    tex.append(r"\section*{Overview}")
    tex.append(f"Dataset: \\texttt{{{latex_escape(str(DATA_PATH))}}}\\\\")
    tex.append(f"Rows: {n_rows}\\\\")
    tex.append(f"Columns: {n_cols} (confirmed)\\\\")
    tex.append(f"Time span (year): {year_min}--{year_max}\\\\")
    tex.append(f"Countries (ISO3): {iso3_count}\\\\")
    tex.append(r"Key unit: election event (national presidential or parliamentary election).")

    tex.append(r"\section*{Variable Dictionary}")

    for col in cols:
        prof = profile["columns"][col]
        series = df[col]
        meta = meta_for(col, labels)
        tex.append(f"\\subsection*{{\\texttt{{{latex_escape(col)}}}}}")
        tex.append(r"\begin{itemize}[leftmargin=*]")
        tex.append(f"  \\item \\textbf{{Variable name}} \\texttt{{{latex_escape(col)}}}")
        tex.append(f"  \\item \\textbf{{Short label}} {latex_escape(meta['label'])}")
        tex.append(f"  \\item \\textbf{{Meaning / definition}} {latex_escape(meta['meaning'])}")
        tex.append(f"  \\item \\textbf{{Value type}} {latex_escape(value_type_text(col, prof, series))}")
        tex.append(f"  \\item \\textbf{{Allowed values / domain}} {domain_text(col, prof, series)}")
        tex.append(f"  \\item \\textbf{{Missingness}} {missing_text(prof, n_rows)}")
        tex.append(f"  \\item \\textbf{{Operationalization / construction}} {latex_escape(meta['construction'])}")
        tex.append(f"  \\item \\textbf{{Source / provenance}} {latex_escape(meta['source'])}")
        tex.append(f"  \\item \\textbf{{Notes \\& validation checks}} {latex_escape(meta['notes'])}")
        tex.append(r"\end{itemize}")

    tex.append(r"\section*{Data Quality / Validation}")
    tex.append(r"We verified the dataset dimensions from the CSV and computed per-column missingness and domains using scripts/profile_dataset.py. Quality flags are defined by \texttt{clean\_flag}, \texttt{competitive\_flag}, \texttt{competitive\_sample}, and \texttt{analysis\_ready} (see code references above). Electoral-system matches include proximity flags (\texttt{des\_match\_2y} and \texttt{des\_match\_5y}).")

    tex.append(r"\appendix")
    tex.append(r"\section*{Appendix: Profile Summary}")
    tex.append(r"\input{reports/data_dictionary_profile_table.tex}")

    tex.append(r"\bibliographystyle{plain}")
    tex.append(r"\bibliography{data_dictionary_refs}")
    tex.append(r"\end{document}")

    tex_content = "\n".join(tex)
    OUT_TEX.write_text(tex_content)

    found = []
    for line in tex:
        if line.startswith("\\subsection*{\\texttt{") or line.startswith("\\\\subsection*{\\\\texttt{"):
            name = line.split("{\\texttt{", 1)[1].rsplit("}}", 1)[0]
            name = name.replace("\\\\_", "_").replace(r"\_", "_")
            found.append(name)
    if found != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in found]
        extra = [c for c in found if c not in EXPECTED_COLUMNS]
        raise ValueError(f"LaTeX variable list mismatch. Missing: {missing}. Extra: {extra}.")

    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
