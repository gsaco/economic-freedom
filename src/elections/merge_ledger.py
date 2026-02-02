from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "run_id",
    "step_id",
    "step_name",
    "left_table",
    "right_table",
    "join_type",
    "keys_left",
    "keys_right",
    "left_rows_pre",
    "right_rows_pre",
    "left_key_dupe_rows",
    "right_key_dupe_rows",
    "rows_post",
    "matched_left_rows",
    "match_rate_left",
    "row_expansion_factor",
    "notes",
]


@dataclass
class MergeLedger:
    run_id: str
    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ledger_path(self) -> Path:
        return self.out_dir / "merge_ledger.csv"

    def append(self, row: dict) -> None:
        record = {col: row.get(col) for col in LEDGER_COLUMNS}
        df = pd.DataFrame([record])
        df.to_csv(self.ledger_path, mode="a", header=not self.ledger_path.exists(), index=False)

    def write_df(self, df: pd.DataFrame, filename: str) -> None:
        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy() if keep else df.copy()


def _count_dupes(df: pd.DataFrame, keys: list[str]) -> int:
    if not keys:
        return 0
    return int(df.duplicated(keys, keep=False).sum())


def logged_merge(
    ledger: MergeLedger,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    how: str,
    step_id: str,
    step_name: str,
    left_table: str,
    right_table: str,
    on: list[str] | None = None,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
    context_cols_left: list[str] | None = None,
    context_cols_right: list[str] | None = None,
    notes: str | None = None,
    keep_merge_indicator: bool = False,
) -> pd.DataFrame:
    left_keys = _as_list(on) if on is not None else _as_list(left_on)
    right_keys = _as_list(on) if on is not None else _as_list(right_on)

    left_rows_pre = len(left)
    right_rows_pre = len(right)
    left_dupes = _count_dupes(left, left_keys)
    right_dupes = _count_dupes(right, right_keys)

    left_tmp = left.copy()
    right_tmp = right.copy()
    left_tmp["__left_row_id"] = np.arange(len(left_tmp))
    right_tmp["__right_row_id"] = np.arange(len(right_tmp))

    merged = left_tmp.merge(
        right_tmp,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        indicator=True,
        suffixes=("", "_right"),
    )

    rows_post = len(merged)
    matched_left_rows = int(merged.loc[merged["_merge"] != "left_only", "__left_row_id"].nunique())
    match_rate_left = float(matched_left_rows / left_rows_pre) if left_rows_pre else 0.0
    row_expansion = float(rows_post / left_rows_pre) if left_rows_pre else 0.0

    note_parts = []
    if left_dupes:
        note_parts.append("left_keys_not_unique")
    if right_dupes:
        note_parts.append("right_keys_not_unique")
    if notes:
        note_parts.append(notes)
    note_text = "; ".join(note_parts) if note_parts else ""

    ledger.append({
        "run_id": ledger.run_id,
        "step_id": step_id,
        "step_name": step_name,
        "left_table": left_table,
        "right_table": right_table,
        "join_type": how,
        "keys_left": ",".join(left_keys),
        "keys_right": ",".join(right_keys),
        "left_rows_pre": left_rows_pre,
        "right_rows_pre": right_rows_pre,
        "left_key_dupe_rows": left_dupes,
        "right_key_dupe_rows": right_dupes,
        "rows_post": rows_post,
        "matched_left_rows": matched_left_rows,
        "match_rate_left": match_rate_left,
        "row_expansion_factor": row_expansion,
        "notes": note_text,
    })

    if left_dupes:
        ledger.write_df(left_tmp[left_tmp.duplicated(left_keys, keep=False)].drop(columns=["__left_row_id"]), f"{step_id}__left_key_dupes.csv")
    else:
        ledger.write_df(pd.DataFrame(columns=left.columns), f"{step_id}__left_key_dupes.csv")
    if right_dupes:
        ledger.write_df(right_tmp[right_tmp.duplicated(right_keys, keep=False)].drop(columns=["__right_row_id"]), f"{step_id}__right_key_dupes.csv")
    else:
        ledger.write_df(pd.DataFrame(columns=right.columns), f"{step_id}__right_key_dupes.csv")

    if how in {"left", "outer"}:
        unmatched_left = merged.loc[merged["_merge"] == "left_only"].drop_duplicates("__left_row_id")
        left_cols = list(dict.fromkeys(left_keys + _as_list(context_cols_left)))
        if left_cols:
            unmatched_left = _select_cols(unmatched_left, left_cols)
        else:
            unmatched_left = unmatched_left.drop(columns=["__left_row_id", "__right_row_id", "_merge"], errors="ignore")
        ledger.write_df(unmatched_left, f"{step_id}__unmatched_left.csv")
    if how in {"right", "outer"}:
        unmatched_right = merged.loc[merged["_merge"] == "right_only"].drop_duplicates("__right_row_id")
        right_cols = list(dict.fromkeys(right_keys + _as_list(context_cols_right)))
        if right_cols:
            # map right columns with _right suffix if needed
            rename_map = {}
            cols = []
            for col in right_cols:
                if col in unmatched_right.columns:
                    cols.append(col)
                else:
                    col_right = f"{col}_right"
                    if col_right in unmatched_right.columns:
                        cols.append(col_right)
                        rename_map[col_right] = col
            unmatched_right = unmatched_right[cols].rename(columns=rename_map)
        else:
            unmatched_right = unmatched_right.drop(columns=["__left_row_id", "__right_row_id", "_merge"], errors="ignore")
        ledger.write_df(unmatched_right, f"{step_id}__unmatched_right.csv")

    drop_cols = ["__left_row_id", "__right_row_id"]
    if not keep_merge_indicator:
        drop_cols.append("_merge")
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def logged_match(
    ledger: MergeLedger,
    match_df: pd.DataFrame,
    *,
    step_id: str,
    step_name: str,
    left_table: str,
    right_table: str,
    key_cols: list[str] | None = None,
    match_status_col: str = "match_status",
    context_cols: list[str] | None = None,
    notes: str | None = None,
) -> None:
    left_rows_pre = len(match_df)
    right_rows_pre = 0
    left_keys = _as_list(key_cols)
    left_dupes = _count_dupes(match_df, left_keys)

    if match_status_col in match_df.columns:
        matched_mask = match_df[match_status_col].isin(["matched", "matched_tiebreak", "manual_override"])
        matched_left_rows = int(matched_mask.sum())
    else:
        matched_left_rows = 0
    match_rate_left = float(matched_left_rows / left_rows_pre) if left_rows_pre else 0.0

    note_parts = []
    if left_dupes:
        note_parts.append("left_keys_not_unique")
    if notes:
        note_parts.append(notes)
    note_text = "; ".join(note_parts) if note_parts else ""

    ledger.append({
        "run_id": ledger.run_id,
        "step_id": step_id,
        "step_name": step_name,
        "left_table": left_table,
        "right_table": right_table,
        "join_type": "match",
        "keys_left": ",".join(left_keys),
        "keys_right": "",
        "left_rows_pre": left_rows_pre,
        "right_rows_pre": right_rows_pre,
        "left_key_dupe_rows": left_dupes,
        "right_key_dupe_rows": 0,
        "rows_post": left_rows_pre,
        "matched_left_rows": matched_left_rows,
        "match_rate_left": match_rate_left,
        "row_expansion_factor": 1.0,
        "notes": note_text,
    })

    if left_dupes:
        ledger.write_df(match_df[match_df.duplicated(left_keys, keep=False)], f"{step_id}__left_key_dupes.csv")
    else:
        ledger.write_df(pd.DataFrame(columns=match_df.columns), f"{step_id}__left_key_dupes.csv")

    if match_status_col in match_df.columns:
        unmatched = match_df[match_df[match_status_col] == "unmatched"].copy()
        ambiguous = match_df[match_df[match_status_col] == "ambiguous"].copy()
    else:
        unmatched = pd.DataFrame(columns=match_df.columns)
        ambiguous = pd.DataFrame(columns=match_df.columns)

    if context_cols:
        cols = list(dict.fromkeys(left_keys + _as_list(context_cols)))
        unmatched = _select_cols(unmatched, cols)
        ambiguous = _select_cols(ambiguous, cols)

    ledger.write_df(unmatched, f"{step_id}__unmatched_left.csv")
    ledger.write_df(ambiguous, f"{step_id}__ambiguous.csv")
