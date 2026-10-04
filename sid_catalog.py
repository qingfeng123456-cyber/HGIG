"""Tokenizer-neutral catalog preparation for fixed semantic identifiers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def catalog_hash(codes: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(codes, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base_digits(value: int, width: int, base: int) -> List[int]:
    digits = [0] * width
    for position in reversed(range(width)):
        digits[position] = int(value % base)
        value //= base
    if value:
        raise ValueError("Collision bucket exceeds the configured SID suffix capacity")
    return digits


@torch.no_grad()
def build_fixed_sids(
    tokenizer: torch.nn.Module,
    item_embeddings: torch.Tensor,
    semantic_depth: int,
    code_length: int,
    code_number: int,
) -> List[List[int]]:
    """Quantize once, then deterministically make every full SID unique.

    The tokenizer is never optimized by the recommendation objective.  Extra
    positions, when configured, encode only the within-bucket instance index.
    """

    if item_embeddings.ndim != 2 or item_embeddings.size(0) < 2:
        raise ValueError("item_embeddings must include a padding row and catalog rows")
    semantic_depth = int(semantic_depth)
    suffix_width = int(code_length) - semantic_depth
    if suffix_width < 0:
        raise ValueError("code_length cannot be smaller than semantic_depth")

    tokenizer.eval()
    device = next(tokenizer.parameters()).device
    semantic_codes = tokenizer.get_indices(item_embeddings[1:].to(device)).detach().cpu().long()
    if semantic_codes.ndim != 2 or semantic_codes.size(1) != semantic_depth:
        raise ValueError(
            f"Tokenizer returned shape {tuple(semantic_codes.shape)}; expected [items, {semantic_depth}]"
        )

    bucket_counts = defaultdict(int)
    result: List[List[int]] = [[-1] * int(code_length)]
    for row in semantic_codes.tolist():
        key = tuple(int(x) for x in row)
        collision_index = bucket_counts[key]
        bucket_counts[key] += 1
        if suffix_width == 0:
            if collision_index:
                raise ValueError(
                    "Tokenizer produced duplicate SIDs but no instance-resolution position is configured"
                )
            suffix = []
        else:
            suffix = _base_digits(collision_index, suffix_width, int(code_number))
        result.append(list(key) + suffix)
    return result


def load_sids(path: str | Path) -> List[List[int]]:
    with open(path, "r", encoding="utf-8") as source:
        values = json.load(source)
    return [[int(token) for token in sid] for sid in values]


def save_sids(path: str | Path, codes: Iterable[Iterable[int]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump([[int(x) for x in sid] for sid in codes], output, ensure_ascii=False)
    temporary.replace(destination)
