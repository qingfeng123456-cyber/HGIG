"""Bounded, JSON-serializable mechanism traces for offline paper figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import torch


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class MechanismTrace:
    """Collect raw mechanism variables, not pre-rendered chart values."""

    def __init__(self, metadata: Mapping[str, Any], sample_limit: int = 4096):
        self.sample_limit = int(sample_limit)
        self.payload: Dict[str, Any] = {
            "schema_version": "4.0",
            "metadata": _plain(metadata),
            "epochs": [],
            "module_1_samples": [],
            "module_2_edges": [],
        }

    def epoch(self, record: Mapping[str, Any]) -> None:
        self.payload["epochs"].append(_plain(record))

    def extend_module_1(self, records: Iterable[Mapping[str, Any]]) -> None:
        target: List[Dict[str, Any]] = self.payload["module_1_samples"]
        remaining = max(self.sample_limit - len(target), 0)
        if remaining:
            target.extend(_plain(list(records)[:remaining]))

    def extend_module_2(self, records: Iterable[Mapping[str, Any]]) -> None:
        target: List[Dict[str, Any]] = self.payload["module_2_edges"]
        remaining = max(self.sample_limit - len(target), 0)
        if remaining:
            target.extend(_plain(list(records)[:remaining]))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(self.payload, output, ensure_ascii=False, indent=2)
        temporary.replace(destination)
