from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import requests

from src.paths import RAW_DIR

NE_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"


def _download_naturalearth(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ne_110m_admin_0_countries.zip"
    shp_path = dest_dir / "ne_110m_admin_0_countries.shp"
    if shp_path.exists():
        return shp_path
    if not zip_path.exists():
        resp = requests.get(NE_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        zf.extractall(dest_dir)
    return shp_path


def load_world_geometries() -> gpd.GeoDataFrame:
    geo_dir = RAW_DIR / "geodata"
    shp_path = _download_naturalearth(geo_dir)
    world = gpd.read_file(shp_path)
    iso_col = "ISO_A3" if "ISO_A3" in world.columns else "iso_a3"
    world = world.rename(columns={iso_col: "iso3c"})
    world["iso3c"] = world["iso3c"].astype(str).str.upper().str.strip()
    world = world[world["iso3c"].str.len() == 3]
    world = world[world["iso3c"] != "-99"]
    return world[["iso3c", "NAME", "geometry"]].rename(columns={"NAME": "country_name"})


def merge_world_data(
    world: gpd.GeoDataFrame,
    df,
    iso_col: str,
    value_col: str,
) -> gpd.GeoDataFrame:
    merged = world.merge(df[[iso_col, value_col]], left_on="iso3c", right_on=iso_col, how="left")
    return merged


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: str = "viridis",
    missing_color: str = "lightgrey",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=True,
        missing_kwds={"color": missing_color, "label": "Missing"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    return fig, ax
