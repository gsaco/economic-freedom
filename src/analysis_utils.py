"""Analysis helpers for reproducible spec logging."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import numpy as np


def hash_file(path: Path | str, algo: str = "sha256") -> str:
    path = Path(path)
    hasher = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def bh_fdr(p_values: Iterable[float]) -> List[float]:
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        qval = ranked[i] * n / rank
        prev = min(prev, qval)
        q[order[i]] = prev
    return q.tolist()
