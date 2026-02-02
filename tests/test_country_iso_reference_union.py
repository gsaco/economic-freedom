from pathlib import Path
import pandas as pd

from elections.country_map import build_iso_reference


def test_iso_reference_union_contains_expected_codes() -> None:
    cow2iso = pd.read_csv("cow2iso.csv")
    iso_ref = build_iso_reference(
        Path("data/external/parlgov.zip"),
        cow2iso=cow2iso,
        manual_path=Path("data/manual/iso_reference_manual.csv"),
    )
    iso3 = set(iso_ref["iso3"].dropna().astype(str))
    for code in ["SRB", "MNE", "MKD", "TLS", "CUW", "XKX"]:
        assert code in iso3
