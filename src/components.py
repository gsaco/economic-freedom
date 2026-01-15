"""EFW component extraction, shocks, and decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


CORE_COLS = {
    "Year": "year",
    "ISO_Code": "iso3",
    "Countries": "country",
    "ECONOMIC FREEDOM ALL AREAS": "efw_summary",
    "Summary": "efw_summary",
    "World Bank Region": "wb_region",
    "World Bank Current Income Classification, 1990-Present": "wb_income_class",
}


@dataclass
class EfwComponentResult:
    df: pd.DataFrame
    component_cols: List[str]
    area_cols: List[str]
    iso3_issues: pd.DataFrame


def _normalize_columns(cols: Iterable[str]) -> list[str]:
    normalized = []
    seen = {}
    for col in cols:
        clean = str(col).strip()
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}__{seen[clean]}"
        else:
            seen[clean] = 0
        normalized.append(clean)
    return normalized


def _find_header_row(raw: pd.DataFrame) -> int:
    for idx, row in raw.iterrows():
        values = [str(val).strip().lower() for val in row.tolist()]
        if "year" in values and "iso_code" in values and "countries" in values:
            return idx
    raise ValueError("Header row not found in EFW index sheet.")


def _slugify(text: str) -> str:
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _dedupe_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    suffix = 2
    while f"{name}_{suffix}" in used:
        suffix += 1
    deduped = f"{name}_{suffix}"
    used.add(deduped)
    return deduped


def _fix_column_label(label: str) -> str:
    label = label.strip()
    if label.startswith("IE "):
        label = label.replace("IE ", "1E ", 1)
    return label


def _match_area_columns(columns: Iterable[str]) -> dict[str, str]:
    area_map: dict[str, str] = {}
    cols = list(columns)
    lower_map = {col.lower(): col for col in cols}

    def pick_area(area_num: int) -> str | None:
        key = f"area {area_num}"
        matches = [c for c in cols if c.lower().startswith(key) and "rank" not in c.lower()]
        if not matches:
            return None
        if area_num == 2:
            with_gender = [c for c in matches if "with gender" in c.lower()]
            without_gender = [c for c in matches if "without gender" in c.lower()]
            if with_gender:
                area_map[with_gender[0]] = "efw_area2"
                if without_gender:
                    area_map[without_gender[0]] = "efw_area2_nogender"
                return with_gender[0]
            if without_gender:
                area_map[without_gender[0]] = "efw_area2"
                return without_gender[0]
        area_map[matches[0]] = f"efw_area{area_num}"
        return matches[0]

    for area_num in range(1, 6):
        pick_area(area_num)

    return area_map


def _extract_component_map(columns: Iterable[str]) -> dict[str, str]:
    component_map: dict[str, str] = {}
    used: set[str] = set()
    area_map = _match_area_columns(columns)
    skip_tokens = {"year", "iso_code", "countries"}

    for col in columns:
        clean = _fix_column_label(str(col))
        lower = clean.lower()
        if lower in skip_tokens:
            continue
        if lower.startswith("data"):
            continue
        if "rank" in lower or "quartile" in lower:
            continue
        if "gender disparity index" in lower:
            continue
        if col in CORE_COLS:
            continue
        if col in area_map:
            continue

        token = clean.split()[0]
        if token.lower() == "ie":
            token = "1E"
        if not re.match(r"^[1-5][A-Z][ivx]*$", token, flags=re.IGNORECASE):
            continue

        label = clean[len(token):].strip()
        if not label:
            label = "component"
        var_name = f"efw_{token.lower()}_{_slugify(label)}"
        var_name = _dedupe_name(var_name, used)
        component_map[col] = var_name

    return component_map


def load_efw_index_sheet(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="EFW Index 1970-2023", header=None)
    header_row = _find_header_row(raw)
    header = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = _normalize_columns(header)
    return df


def clean_efw_index(df: pd.DataFrame) -> EfwComponentResult:
    df = df.copy()
    df.columns = [_fix_column_label(str(col)) for col in df.columns]

    area_map = _match_area_columns(df.columns)
    component_map = _extract_component_map(df.columns)
    rename_map = {**CORE_COLS, **area_map, **component_map}

    df = df.rename(columns=rename_map)

    keep_cols = ["year", "iso3", "country", "wb_region", "wb_income_class"]
    keep_cols += [col for col in ["efw_summary", "efw_area1", "efw_area2", "efw_area2_nogender", "efw_area3", "efw_area4", "efw_area5"] if col in df.columns]
    component_cols = list(component_map.values())
    keep_cols += component_cols

    df = df[keep_cols]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["country"] = df["country"].astype(str).str.strip()

    iso3_issues = df[(df["iso3"].str.len() != 3) | df["iso3"].isna()].copy()
    df = df[df["iso3"].str.len() == 3].copy()
    df = df.dropna(subset=["year"])

    value_cols = [col for col in df.columns if col.startswith("efw_")]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    area_cols = [col for col in ["efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"] if col in df.columns]
    return EfwComponentResult(df=df, component_cols=component_cols, area_cols=area_cols, iso3_issues=iso3_issues)


def to_quinquennial(df: pd.DataFrame, start_year: int = 1970, end_year: int = 2020) -> pd.DataFrame:
    quin_years = set(range(start_year, end_year + 1, 5))
    return df[df["year"].isin(quin_years)].copy()


def compute_changes(df: pd.DataFrame, value_cols: Iterable[str]) -> pd.DataFrame:
    df = df.sort_values(["iso3", "year"]).copy()
    for col in value_cols:
        df[f"d_{col}"] = df.groupby("iso3")[col].diff()
    return df


def get_core_component_cols(df: pd.DataFrame) -> List[str]:
    cols = ["efw_summary", "efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]
    if "efw_area2_nogender" in df.columns:
        cols.append("efw_area2_nogender")
    return [col for col in cols if col in df.columns]


def add_shock_definitions(
    df: pd.DataFrame,
    change_cols: List[str],
    quantiles: Tuple[float, ...] = (0.9, 0.85),
    sd_thresholds: Tuple[float, ...] = (1.0, 1.5),
    abs_quantile: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    thresholds_global_records = []
    thresholds_year_records = []

    for col in change_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        std = float(series.std())
        abs_threshold = float(series.abs().quantile(abs_quantile))
        record = {
            "variable": col,
            "n_obs": int(series.shape[0]),
            "std": std,
            "abs_quantile": abs_quantile,
            "abs_threshold": abs_threshold,
        }

        for q in quantiles:
            low_q = 1 - q
            pos_thresh = float(series.quantile(q))
            neg_thresh = float(series.quantile(low_q))
            record[f"q{int(q*100)}_pos"] = pos_thresh
            record[f"q{int(q*100)}_neg"] = neg_thresh

            df[f"shock_pos_{col}_q{int(q*100)}p"] = df[col] >= pos_thresh
            df[f"shock_neg_{col}_q{int(q*100)}p"] = df[col] <= neg_thresh

            by_year = df.groupby("year")[col].quantile([low_q, q]).unstack()
            pos_map = by_year[q].to_dict()
            neg_map = by_year[low_q].to_dict()
            pos_year = df["year"].map(pos_map)
            neg_year = df["year"].map(neg_map)

            df[f"shock_pos_{col}_q{int(q*100)}y"] = df[col] >= pos_year
            df[f"shock_neg_{col}_q{int(q*100)}y"] = df[col] <= neg_year

            for year, row in by_year.iterrows():
                thresholds_year_records.append(
                    {
                        "variable": col,
                        "year": year,
                        "q": q,
                        "pos_threshold": row[q],
                        "neg_threshold": row[low_q],
                    }
                )

        for k in sd_thresholds:
            df[f"shock_pos_{col}_sd{k}"] = df[col] >= k * std
            df[f"shock_neg_{col}_sd{k}"] = df[col] <= -k * std

        df[f"shock_pos_{col}_abs"] = df[col] >= abs_threshold
        df[f"shock_neg_{col}_abs"] = df[col] <= -abs_threshold

        thresholds_global_records.append(record)

    thresholds_global = pd.DataFrame.from_records(thresholds_global_records)
    thresholds_by_year = pd.DataFrame.from_records(thresholds_year_records)
    return df, thresholds_global, thresholds_by_year


def compute_shock_decomposition(
    df: pd.DataFrame,
    summary_change_col: str,
    area_change_cols: List[str],
    shock_flag_col: str,
) -> pd.DataFrame:
    events = df[df[shock_flag_col]].copy()
    records = []

    for _, row in events.iterrows():
        area_changes = row[area_change_cols]
        total_abs = float(area_changes.abs().sum())
        summary_change = row[summary_change_col]

        for area_col in area_change_cols:
            area_change = row[area_col]
            abs_share = np.nan
            signed_share = np.nan
            if total_abs > 0:
                abs_share = float(abs(area_change) / total_abs)
            if pd.notna(summary_change) and summary_change != 0:
                signed_share = float(area_change / (summary_change * len(area_change_cols)))
            records.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "year": row["year"],
                    "summary_change": summary_change,
                    "area": area_col,
                    "area_change": area_change,
                    "abs_share": abs_share,
                    "signed_share": signed_share,
                }
            )

    if not records:
        return pd.DataFrame(columns=["iso3", "country", "year", "summary_change", "area", "area_change", "abs_share", "signed_share", "rank_abs"])

    result = pd.DataFrame.from_records(records)
    result["rank_abs"] = result.groupby(["iso3", "year"])["abs_share"].rank(ascending=False, method="dense")
    return result.sort_values(["year", "iso3", "rank_abs"])


def build_efw_component_panel(
    path: Path | str,
    start_year: int = 1970,
    end_year: int = 2020,
) -> EfwComponentResult:
    raw = load_efw_index_sheet(path)
    cleaned = clean_efw_index(raw)
    df = to_quinquennial(cleaned.df, start_year=start_year, end_year=end_year)

    value_cols = [col for col in df.columns if col.startswith("efw_")]
    df = compute_changes(df, value_cols)

    return EfwComponentResult(
        df=df,
        component_cols=cleaned.component_cols,
        area_cols=cleaned.area_cols,
        iso3_issues=cleaned.iso3_issues,
    )


def build_reform_bundles(
    df: pd.DataFrame,
    change_cols: List[str],
    k: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df[["iso3", "year"] + change_cols].dropna().copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[change_cols])
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    data["bundle"] = kmeans.fit_predict(X)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=change_cols)
    centers = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=change_cols,
    )
    centers["bundle"] = range(k)
    centers = centers.set_index("bundle")
    return data, centers
