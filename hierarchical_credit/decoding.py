"""Batched SID beam search with an explicit, reproducible score protocol."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from .sid_trie import SIDTrie


@dataclass
class DecodedBatch:
    item_ids: torch.Tensor
    sids: torch.Tensor
    scores: torch.Tensor


@torch.no_grad()
def beam_search(
    policy: torch.nn.Module,
    history_codes: torch.Tensor,
    attention_mask: torch.Tensor,
    trie: SIDTrie,
    num_beams: int,
    topk: int,
    decoder_start_token_id: int = 0,
    protocol: str = "full_vocabulary",
) -> DecodedBatch:
    """Rank fixed-length SIDs without reading labels or target-dependent state.

    full_vocabulary applies log softmax over every token, no catalog pruning,
    and no length penalty. Invalid
    final identifiers occupy their original ranks and map to item -1.
    catalog_full_vocabulary masks illegal expansions AFTER the full softmax.
    catalog_siblings masks them BEFORE softmax (the older experiment protocol).
    Equal-score tie ordering may differ between device top-k kernels.
    """
    if not 0 < topk <= num_beams:
        raise ValueError("Require 0 < topk <= num_beams")
    if protocol not in {"full_vocabulary", "catalog_full_vocabulary", "catalog_siblings"}:
        raise ValueError(f"Unknown decoding protocol: {protocol}")
    if history_codes.ndim != 2 or attention_mask.shape != history_codes.shape:
        raise ValueError("history/mask must have matching [batch, sequence] shape")
    batch_size = history_codes.size(0)
    if batch_size == 0:
        raise ValueError("Cannot decode an empty batch")
    device = history_codes.device
    # Run the expensive history encoder once per user, BEFORE beam expansion.
    cached = None
    if callable(getattr(policy, "encode", None)):
        encoded = policy.encode(history_codes, attention_mask)
        cached = encoded.last_hidden_state
    prefixes = torch.empty(batch_size, 1, 0, dtype=torch.long, device=device)
    beam_scores = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
    node_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    tables = None
    expanded_hidden = expanded_mask = expanded_history = None
    previous_width = -1

    for depth in range(trie.code_length):
        width = prefixes.size(1)
        if width != previous_width:
            owners = torch.arange(batch_size, device=device).repeat_interleave(width)
            expanded_mask = attention_mask.index_select(0, owners)
            if cached is not None:
                expanded_hidden = cached.index_select(0, owners)
            else:
                expanded_history = history_codes.index_select(0, owners)
            previous_width = width
        decoder_ids = torch.cat(
            [torch.full((batch_size * width, 1), int(decoder_start_token_id),
                        dtype=torch.long, device=device),
             prefixes.reshape(batch_size * width, depth)], dim=1
        )
        kwargs = dict(attention_mask=expanded_mask, decoder_input_ids=decoder_ids)
        if getattr(policy, "supports_last_depth_routing", False):
            # Past routed logits are discarded by beam search. Skip only their
            # redundant expert work, leaving T5 states and current scores intact.
            kwargs["routing_depth"] = depth
        if cached is not None:
            kwargs["encoder_outputs"] = (expanded_hidden,)
        else:
            kwargs["input_ids"] = expanded_history
        logits = policy(**kwargs).logits[:, -1].float()
        vocabulary = logits.size(-1)
        if tables is None:
            tables = trie.tensor_tables(vocabulary, device)
        transitions = tables.transitions[depth].index_select(
            0, node_ids.reshape(-1).clamp_min(0)
        ).long()
        legal = transitions.ge(0) & node_ids.reshape(-1, 1).ge(0)
        if protocol == "catalog_siblings":
            # Dead padded beams use finite logits to avoid all--inf softmax.
            safe = logits.masked_fill(~legal, -1.0e9)
            log_probs = safe.log_softmax(-1).masked_fill(~legal, -torch.inf)
        else:
            log_probs = logits.log_softmax(-1)
            if protocol == "catalog_full_vocabulary":
                log_probs = log_probs.masked_fill(~legal, -torch.inf)
        candidates = log_probs.view(batch_size, width, vocabulary) + beam_scores.unsqueeze(-1)
        next_width = min(num_beams, width * vocabulary)
        next_scores, selection = candidates.flatten(1).topk(next_width, dim=1)
        parent = torch.div(selection, vocabulary, rounding_mode="floor")
        token = selection.remainder(vocabulary)
        prefixes = torch.cat(
            [prefixes.gather(1, parent.unsqueeze(-1).expand(-1, -1, depth)),
             token.unsqueeze(-1)], dim=-1
        )
        next_nodes = transitions.view(batch_size, width * vocabulary).gather(1, selection)
        parent_live = node_ids.gather(1, parent).ge(0)
        node_ids = next_nodes.masked_fill(~parent_live, -1)
        beam_scores = next_scores

    actual = min(topk, beam_scores.size(1))
    out_scores = beam_scores[:, :actual]
    out_sids = prefixes[:, :actual]
    nodes = node_ids[:, :actual]
    item_ids = tables.terminal_item_ids.index_select(0, nodes.clamp_min(0).reshape(-1)).view_as(nodes)
    item_ids = item_ids.masked_fill(nodes.lt(0) | ~out_scores.isfinite(), -1)
    out_sids = out_sids.masked_fill(~out_scores.isfinite().unsqueeze(-1), -1)
    if actual < topk:
        padding = topk - actual
        item_ids = torch.cat([item_ids, torch.full((batch_size, padding), -1,
                             dtype=torch.long, device=device)], dim=1)
        out_sids = torch.cat([out_sids, torch.full((batch_size, padding, trie.code_length),
                             -1, dtype=torch.long, device=device)], dim=1)
        out_scores = torch.cat([out_scores, torch.full((batch_size, padding), -torch.inf,
                               device=device)], dim=1)
    return DecodedBatch(item_ids, out_sids, out_scores)


@torch.no_grad()
def constrained_beam_search(
    policy, history_codes, attention_mask, trie, num_beams, topk,
    decoder_start_token_id=0,
) -> DecodedBatch:
    """Backward-compatible entry point for catalog-local beam scoring."""
    return beam_search(
        policy, history_codes, attention_mask, trie, num_beams, topk,
        decoder_start_token_id, protocol="catalog_siblings",
    )
