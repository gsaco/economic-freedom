"""Country name normalization and ISO3 mapping helpers."""

from __future__ import annotations

import unicodedata
from typing import Dict, Optional

import pycountry


BASE_OVERRIDES = {
    "Bahamas, The": "BHS",
    "Brunei": "BRN",
    "Cabo Verde": "CPV",
    "Cape Verde": "CPV",
    "Central African Rep.": "CAF",
    "Central African Republic": "CAF",
    "China, P.R.": "CHN",
    "China, P.R.: Hong Kong": "HKG",
    "Congo, Dem. Rep.": "COD",
    "Congo, Dem. Rep. of": "COD",
    "Congo, Rep.": "COG",
    "Congo, Rep. of": "COG",
    "Congo-Brazzaville": "COG",
    "Congo-Kinshasa": "COD",
    "Cote dIvoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "CA te d'Ivoire": "CIV",
    "Czech Republic": "CZE",
    "Egypt, Arab Rep.": "EGY",
    "Eswatini": "SWZ",
    "Gambia, The": "GMB",
    "Hong Kong SAR, China": "HKG",
    "Iran, I.R. of": "IRN",
    "Iran, Islamic Rep.": "IRN",
    "Korea": "KOR",
    "Korea, Dem. Rep.": "PRK",
    "Korea, Rep.": "KOR",
    "Lao PDR": "LAO",
    "Lao Peoples Dem. Rep.": "LAO",
    "Lao People's Dem. Rep.": "LAO",
    "Macao SAR, China": "MAC",
    "Micronesia": "FSM",
    "Macedonia": "MKD",
    "North Macedonia": "MKD",
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "Sao Tome and Principe": "STP",
    "SAo TomA and PrAncipe": "STP",
    "Serbia, Republic of": "SRB",
    "St. Vincent and Grenadines": "VCT",
    "Slovak Republic": "SVK",
    "St. Kitts and Nevis": "KNA",
    "St. Lucia": "LCA",
    "St. Vincent and the Grenadines": "VCT",
    "Swaziland": "SWZ",
    "Turkey": "TUR",
    "Venezuela, RB": "VEN",
    "Yemen, Rep.": "YEM",
    "Yugoslavia, SFR": "YUG",
    "Yugoslavia": "YUG",
    "Soviet Union": "SUN",
    "Czechoslovakia": "CSK",
    "Kosovo": "XKX",
    "Palestinian Territories": "PSE",
}


def normalize_country_name(name: str) -> str:
    name = str(name).strip()
    name = " ".join(name.split())
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name


def country_to_iso3(name: str, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    if name is None:
        return None
    normalized = normalize_country_name(name)
    if overrides and normalized in overrides:
        return overrides[normalized]
    if normalized in BASE_OVERRIDES:
        return BASE_OVERRIDES[normalized]
    try:
        return pycountry.countries.lookup(normalized).alpha_3
    except LookupError:
        return None
