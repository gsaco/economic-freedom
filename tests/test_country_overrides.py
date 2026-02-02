import pandas as pd

from elections.country_map import map_names_to_iso
from elections.standardize import normalize_name


def test_country_overrides_apply_by_source() -> None:
    iso_ref = pd.DataFrame({
        "country_name": ["United Kingdom"],
        "iso3": ["GBR"],
    })
    iso_ref["country_name_norm"] = iso_ref["country_name"].map(normalize_name)

    overrides = pd.DataFrame({
        "source_system": ["DPI"],
        "source_name_raw": ["UK"],
        "iso3": ["GBR"],
        "notes": ["abbrev"],
    })

    out = map_names_to_iso(pd.Series(["UK"]), iso_ref, min_score=90, tie_delta=0.5, overrides=overrides, source_system="DPI")
    assert out.loc[0, "iso3"] == "GBR"
    assert out.loc[0, "match_method"] == "override"

    out_other = map_names_to_iso(pd.Series(["UK"]), iso_ref, min_score=90, tie_delta=0.5, overrides=overrides, source_system="NED")
    assert out_other.loc[0, "match_status"] == "unmatched"
