from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleSpec:
    name: str
    running_var: str
    instrument: str | None = None
    filter_query: str | None = None
    pooled: bool = False


@dataclass(frozen=True)
class EFWSpec:
    mode: str
    efw_col: str
    pre_window: tuple[int, int] = (-3, -1)
    post_window: tuple[int, int] = (1, 3)
    min_pre_obs: int = 2
    min_post_obs: int = 2
    standardize: bool = False
    winsorize: tuple[float, float] | None = None


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    base_col: str
    transform: str
    horizons: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    min_avg_obs: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    controls: tuple[str, ...] = ()
    window: float | None = 0.03
    windows_grid: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)
    select_window_by_balance: bool = False
    cutoff: float = 0.0
    cluster: str | None = "iso3c"
    year_fe: bool = False
    year_col: str | None = "election_year"
    kernel: str = "triangular"
    order: int = 1


@dataclass(frozen=True)
class DiagnosticSpec:
    balance_vars: tuple[str, ...] = ()
    balance_p: float = 0.15
    density_bw: float = 0.1
    pretrend_p: float = 0.1
    fstat_min: float = 10.0
    min_n: int = 10


@dataclass(frozen=True)
class GroupSpec:
    column: str
    value: int | str


@dataclass(frozen=True)
class Spec:
    name: str
    sample: SampleSpec
    efw: EFWSpec
    outcome: OutcomeSpec
    model: ModelSpec
    diagnostics: DiagnosticSpec
    group: GroupSpec | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample": _dataclass_to_dict(self.sample),
            "efw": _dataclass_to_dict(self.efw),
            "outcome": _dataclass_to_dict(self.outcome),
            "model": _dataclass_to_dict(self.model),
            "diagnostics": _dataclass_to_dict(self.diagnostics),
            "group": _dataclass_to_dict(self.group) if self.group else None,
            "tags": list(self.tags),
        }


def spec_from_dict(payload: dict[str, Any]) -> Spec:
    sample = SampleSpec(**payload["sample"])
    efw = EFWSpec(**payload["efw"])
    outcome = OutcomeSpec(**payload["outcome"])
    model = ModelSpec(**payload["model"])
    diagnostics = DiagnosticSpec(**payload["diagnostics"])
    group = None
    if payload.get("group"):
        group = GroupSpec(**payload["group"])
    tags = tuple(payload.get("tags", []))
    return Spec(
        name=payload["name"],
        sample=sample,
        efw=efw,
        outcome=outcome,
        model=model,
        diagnostics=diagnostics,
        group=group,
        tags=tags,
    )


def spec_id(spec: Spec, *, prefix: str = "spec") -> str:
    payload = spec.to_dict()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def flatten_dict(data: dict[str, Any], *, parent_key: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, parent_key=new_key))
        else:
            flat[new_key] = value
    return flat


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if not hasattr(obj, "__dict__"):
        raise TypeError("Expected dataclass instance")
    payload: dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload
