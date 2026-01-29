from __future__ import annotations

from pathlib import Path
import pandas as pd


def apply_patches(df: pd.DataFrame, patch_file: Path, target_table: str) -> pd.DataFrame:
    if not patch_file.exists():
        return df
    patches = pd.read_csv(patch_file)
    patches = patches[patches["target_table"] == target_table]
    if patches.empty:
        return df

    out = df.copy()
    for _, patch in patches.iterrows():
        key_type = patch["key_type"]
        key_value = patch["key_value"]
        field = patch["field_name"]
        old_value = patch.get("old_value")
        new_value = patch.get("new_value")
        patch_id = patch["patch_id"]

        if key_type not in out.columns:
            continue
        mask = out[key_type].astype(str) == str(key_value)
        if not mask.any():
            continue
        if pd.notna(old_value):
            current = out.loc[mask, field]
            if not (current.astype(str) == str(old_value)).all():
                raise ValueError(f"Patch {patch_id} old_value mismatch for {field}")
        out.loc[mask, field] = new_value
        out[f"{field}_patch_applied"] = out.get(f"{field}_patch_applied", False)
        out.loc[mask, f"{field}_patch_applied"] = True
        out[f"{field}_patch_id"] = out.get(f"{field}_patch_id", pd.NA)
        out.loc[mask, f"{field}_patch_id"] = patch_id
    return out
