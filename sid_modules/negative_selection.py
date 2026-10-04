"""History-safe, decision-diverse mining from an existing beam pool.

No extra generation pass and no future validation/test labels are required.
Unobserved candidates remain uncertain negatives, not verified dislikes.
"""

import torch


@torch.no_grad()
def select_repair_negatives(decoded, labels, history, history_mask, count, vocabulary):
    batch, pool_size, depth = decoded.sids.shape
    if not 0 < count <= pool_size:
        raise ValueError("Negative count must fit the decoded candidate pool")
    mismatch = decoded.sids.ne(labels.unsqueeze(1))
    divergence = mismatch.long().argmax(-1)
    tokens = decoded.sids.gather(-1, divergence.unsqueeze(-1)).squeeze(-1)
    edge_key = divergence * int(vocabulary) + tokens
    past = history.reshape(batch, -1, depth)
    observed = history_mask.reshape(batch, -1, depth).all(-1)
    seen = (decoded.sids.unsqueeze(2).eq(past.unsqueeze(1)).all(-1)
            & observed.unsqueeze(1)).any(-1)
    finite = decoded.scores.isfinite()
    valid = finite & mismatch.any(-1) & ~seen
    selected = torch.zeros_like(valid)
    used_depth = torch.zeros(batch, depth, dtype=torch.bool, device=labels.device)
    ranks = torch.arange(pool_size, device=labels.device).unsqueeze(0).expand(batch, -1)
    selected_keys = []
    indices = []
    masks = []
    for _slot in range(int(count)):
        new_depth = ~used_depth.gather(1, divergence)
        if selected_keys:
            new_edge = ~(edge_key.unsqueeze(-1).eq(torch.stack(selected_keys, -1).unsqueeze(1))).any(-1)
        else:
            new_edge = torch.ones_like(valid)
        # Keep hardness order within each priority tier: uncovered depth,
        # uncovered sibling decision, then a different leaf on a used edge.
        priority = ranks + (~new_depth).long() * (2 * pool_size) + (~new_edge).long() * pool_size
        priority = priority.masked_fill(~valid | selected, 5 * pool_size)
        index = priority.argmin(1, keepdim=True)
        usable = valid.gather(1, index) & ~selected.gather(1, index)
        selected.scatter_(1, index, selected.gather(1, index) | usable)
        chosen_depth = divergence.gather(1, index)
        used_depth.scatter_(1, chosen_depth, used_depth.gather(1, chosen_depth) | usable)
        selected_keys.append(edge_key.gather(1, index).squeeze(1).masked_fill(~usable.squeeze(1), -1))
        indices.append(index)
        masks.append(usable)
    order = torch.cat(indices, 1)
    usable = torch.cat(masks, 1)
    negatives = decoded.sids.gather(1, order.unsqueeze(-1).expand(-1, -1, depth))
    negatives = torch.where(usable.unsqueeze(-1), negatives, labels.unsqueeze(1))
    selected_item_ids = decoded.item_ids.gather(1, order)
    metrics = {"negative_pool_seen_fraction": (seen & finite).float().mean(),
               "negative_pool_usable_fraction": valid.float().mean(),
               "negative_selected_depths": used_depth.float().sum(-1).mean(),
               "negative_selected_catalog_fraction":
                   (selected_item_ids.gt(0) & usable).float().sum() / usable.sum().clamp_min(1)}
    diagnostics = {"negative_beam_rank": (order + 1).float(),
                   "negative_is_catalog_item": selected_item_ids.gt(0).float()}
    return negatives, usable, metrics, diagnostics
