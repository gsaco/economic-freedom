from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    external: Path
    interim: Path
    processed: Path
    reports: Path
    audit: Path
    manifests: Path
    manual_patch: Path
    cow2iso: Path
    partyfacts_core: Path
    partyfacts_external: Path


@dataclass(frozen=True)
class Params:
    country_match: dict
    party_match: dict
    ideology: dict
    dedupe: dict
    efw: dict
    standardization: dict
    flags: dict


@dataclass(frozen=True)
class Config:
    paths: Paths
    params: Params


def load_config(root: Path, config_path: Path | None = None) -> Config:
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    root_path = root
    def p(key: str, default: str | None = None) -> Path:
        if key in paths:
            return (root_path / paths[key]).resolve()
        if default is None:
            raise KeyError(f"Missing required path key: {key}")
        return (root_path / default).resolve()

    def pf_path(key: str, preferred: str, fallback: str) -> Path:
        candidate = (root_path / paths.get(key, preferred)).resolve()
        if candidate.exists():
            return candidate
        fallback_path = (root_path / fallback).resolve()
        return fallback_path

    paths_obj = Paths(
        root=root_path,
        raw=p("raw"),
        external=p("external"),
        interim=p("interim"),
        processed=p("processed"),
        reports=p("reports"),
        audit=p("audit"),
        manifests=p("manifests"),
        manual_patch=(root_path / paths.get("manual_patch", "manual_patch.csv")).resolve(),
        cow2iso=(root_path / paths.get("cow2iso", "cow2iso.csv")).resolve(),
        partyfacts_core=pf_path("partyfacts_core", "partyfacts-core-parties.csv", "data/external/partyfacts_core_parties.csv"),
        partyfacts_external=pf_path("partyfacts_external", "partyfacts-external-parties.csv", "data/external/partyfacts_external_parties.csv"),
    )

    params_obj = Params(
        country_match=params.get("country_match", {}),
        party_match=params.get("party_match", {}),
        ideology=params.get("ideology", {}),
        dedupe=params.get("dedupe", {}),
        efw=params.get("efw", {}),
        standardization=params.get("standardization", {}),
        flags=params.get("flags", {}),
    )

    return Config(paths=paths_obj, params=params_obj)
