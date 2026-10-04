"""Collision-aware item ranking and hierarchy diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch


def dcg(scores: Sequence[float]) -> float:
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.size == 0:
        return 0.0
    return float(
        np.sum((2.0**scores_array - 1.0) / np.log2(np.arange(2, scores_array.size + 2)))
    )


def ndcg_at_k(relevance: Sequence[float], k: int) -> float:
    relevance_array = np.asarray(relevance, dtype=np.float64)[: int(k)]
    ideal = dcg(np.sort(relevance_array)[::-1])
    return 0.0 if ideal <= 0 else dcg(relevance_array) / ideal


def ranking_metric_sums(
    predicted_item_ids: torch.Tensor,
    target_item_ids: torch.Tensor,
    metrics: Iterable[str],
    return_tensors: bool = False,
) -> Dict[str, float]:
    """Rank metrics without removing invalid or duplicate prediction positions.

    A noncatalog SID occupies a rank in the released unconstrained beam search.
    Removing -1 entries before computing rank would inflate the comparison.
    Each target receives credit once, at its first matching position.
    """
    predictions = predicted_item_ids.detach()
    targets = target_item_ids.detach().reshape(-1, 1)
    matches = predictions.eq(targets) & predictions.ge(0)
    positions = torch.arange(1, predictions.size(1) + 1, device=predictions.device).unsqueeze(0)
    rank = torch.where(matches, positions, predictions.size(1) + 1).min(dim=1).values
    output = {}
    for metric in metrics:
        name, cutoff = metric.lower().split("@", 1)
        cutoff = int(cutoff)
        hit = rank.le(cutoff) & rank.le(predictions.size(1))
        if name in {"recall", "hit", "hr"}:
            value = hit.float()
        elif name == "ndcg":
            value = hit.float() / torch.log2(rank.float() + 1)
        else:
            raise ValueError(f"Unsupported ranking metric: {metric}")
        total = value.sum()
        output[metric.lower()] = total if return_tensors else float(total.item())
    return output


def hierarchy_metric_sums(
    predicted_sids: torch.Tensor,
    target_sids: torch.Tensor,
    semantic_depth: int,
    return_tensors: bool = False,
) -> Dict[str, float]:
    """Top-1 prefix accuracy and first-divergence counts."""

    predictions = predicted_sids.detach().long()
    targets = target_sids.detach().long()
    if predictions.ndim == 3:
        predictions = predictions[:, 0]
    if targets.ndim == 3:
        targets = targets[:, 0]
    output = {}
    equal = predictions.eq(targets)
    for depth in range(1, semantic_depth + 1):
        output[f"prefix_acc@{depth}"] = equal[:, :depth].all(-1).float().sum()
    complete = equal.all(-1)
    output["full_sid_accuracy"] = complete.float().sum()
    divergence = (~equal).long().argmax(-1)
    for depth in range(predictions.size(1)):
        output[f"first_divergence@{depth + 1}"] = (divergence.eq(depth) & ~complete).float().sum()
    output["no_divergence"] = complete.float().sum()
    return output if return_tensors else {key: float(value.item()) for key, value in output.items()}
