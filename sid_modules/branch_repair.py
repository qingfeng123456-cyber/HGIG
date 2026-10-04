"""Module 2: router-aware contrastive branch repair (RCBR).

Hard-negative preference learning is established prior work (S-DPO/ReRe), and
prefix-aware optimization is established prior work (APAO/BEAR). The ranking
term here is a leaf-derived proxy: prefixes of sampled final beam candidates
do not reconstruct every intermediate beam or exactly model beam pruning.
This implementation also audits the router at the first divergent decision,
guards against a harmful routing correction, and retains the trainer's strong
full-vocabulary CE anchor. It uses logged training targets, never test labels.
The objectives are empirical hypotheses, not improvement guarantees.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_log_probabilities(logits, labels):
    return logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def cumulative_prefix_ranking(token_logp, candidates, valid_negatives, margin=0.5):
    """Rank distinct sampled negative prefixes against the logged prefix.

    Scores are SUMS of full-vocabulary token log probabilities, exactly the
    score definition used by the decoder; there is no length normalization.
    Equal positive/negative prefixes are not ranking examples. Multiple leaves
    sharing one negative prefix contribute one unit of softmax mass, achieved
    by subtracting the log of their multiplicity. The mean is over user-depth
    decisions with at least one usable competitor, not over individual leaves.

    This is not the complete set of prefixes considered during beam search.
    Both policies must score candidates with the same deterministic context,
    so identical prefixes have identical probabilities.
    """
    cumulative = token_logp.float().cumsum(-1)
    prefix_margins = cumulative[:, :1] - cumulative[:, 1:]
    negative = candidates[:, 1:]
    diverged = negative.ne(candidates[:, :1]).long().cumsum(-1).gt(0)
    valid_prefix = valid_negatives.unsqueeze(-1) & diverged
    same_prefix = negative.unsqueeze(2).eq(negative.unsqueeze(1)).long().cumprod(-1).bool()
    multiplicity = (same_prefix & valid_prefix.unsqueeze(1)).sum(2).clamp_min(1)
    competitor_terms = (
        float(margin) - prefix_margins - multiplicity.float().log()
    ).masked_fill(~valid_prefix, -torch.inf)
    per_decision = torch.logsumexp(
        torch.cat([torch.zeros_like(cumulative[:, :1]), competitor_terms], dim=1),
        dim=1,
    )
    active = valid_prefix.any(1)
    loss = (per_decision * active.float()).sum() / active.float().sum().clamp_min(1.0)
    return loss, prefix_margins, valid_prefix


def repair_objective(outputs, reference, candidates, valid_negatives, config):
    """Candidates are [B, 1+N, D], with the logged positive at index zero.

    Both policies teacher-force the same candidates in deterministic eval mode.
    First-divergence margins are read from the POSITIVE trajectory: the two
    tokens therefore share exactly the same history and decoder prefix. No
    counterfactual causal effect on real users is claimed by this comparison.
    """
    batch, count, depth_count = candidates.shape
    positive = candidates[:, 0]
    negative = candidates[:, 1:]
    logits = outputs.logits.float().view(batch, count, depth_count, -1)
    base_logits = outputs.base_logits.float().view_as(logits)
    ref_logits = reference.logits.float().view_as(logits)
    token_logp = token_log_probabilities(logits, candidates)
    ref_token_logp = token_log_probabilities(ref_logits, candidates).detach()
    logp = token_logp.sum(-1)
    ref_logp = ref_token_logp.sum(-1)
    beta = float(config.get("preference_beta", 0.2))
    relative = beta * (logp - ref_logp)
    seq_differences = (relative[:, 1:] - relative[:, :1]).masked_fill(~valid_negatives, -torch.inf)
    # Partial-ranking softmax: each logged positive outranks sampled negatives.
    sequence_loss = torch.logsumexp(
        torch.cat([torch.zeros_like(relative[:, :1]), seq_differences], dim=1), dim=1
    ).mean()

    mismatch = negative.ne(positive.unsqueeze(1))
    divergence = mismatch.long().argmax(-1)
    valid = valid_negatives & mismatch.any(-1)
    prefix_loss, prefix_margins, valid_prefix = cumulative_prefix_ranking(
        token_logp, candidates, valid, float(config.get("prefix_margin", 0.5))
    )
    ref_cumulative = ref_token_logp.cumsum(-1)
    ref_prefix_margins = ref_cumulative[:, :1] - ref_cumulative[:, 1:]
    # Select one comparison depth from the FROZEN reference, then compare both
    # policies at that same point. Invalid candidates receive explicit zeros.
    worst_prefix_index = ref_prefix_margins.masked_fill(~valid_prefix, torch.inf).argmin(-1)
    ref_worst_margin = ref_prefix_margins.gather(-1, worst_prefix_index.unsqueeze(-1)).squeeze(-1)
    fused_worst_margin = prefix_margins.gather(-1, worst_prefix_index.unsqueeze(-1)).squeeze(-1)
    ref_worst_margin = ref_worst_margin.masked_fill(~valid, 0.0)
    fused_worst_margin = fused_worst_margin.masked_fill(~valid, 0.0)
    row = torch.arange(batch, device=candidates.device).unsqueeze(1).expand_as(divergence)
    target_token = positive.gather(1, divergence)
    other_token = negative.gather(2, divergence.unsqueeze(-1)).squeeze(-1)
    fused_at_branch = logits[:, 0][row, divergence]
    base_at_branch = base_logits[:, 0][row, divergence]
    ref_at_branch = ref_logits[:, 0][row, divergence]

    def margin(values):
        return (values.gather(2, target_token.unsqueeze(-1))
                - values.gather(2, other_token.unsqueeze(-1))).squeeze(-1)

    fused_margin = margin(fused_at_branch)
    base_margin = margin(base_at_branch)
    ref_margin = margin(ref_at_branch).detach()
    # Multiple beams may cross the SAME sibling edge. Count that decision once
    # per user instead of accidentally multiplying its influence by leaf count.
    branch_key = divergence * logits.size(-1) + other_token
    repeats = (branch_key.unsqueeze(2).eq(branch_key.unsqueeze(1))
               & valid.unsqueeze(1)).sum(-1).clamp_min(1)
    weights = valid.float() / repeats.float()
    normalizer = weights.sum().clamp_min(1.0)
    router_harm = F.relu(base_margin.detach() - fused_margin)
    guard_loss = (weights * router_harm).sum() / normalizer
    auxiliary = (
        float(config.get("sequence_weight", 0.1)) * sequence_loss
        + float(config.get("prefix_weight", 0.15)) * prefix_loss
        + float(config.get("router_guard_weight", 0.05)) * guard_loss
    )
    diagnostics = {
        "valid": valid.detach(), "depth": (divergence + 1).detach(),
        "base_margin": base_margin.detach(), "fused_margin": fused_margin.detach(),
        "reference_margin": ref_margin.detach(),
        "router_contribution": (fused_margin - base_margin).detach(),
        "repair_pressure": torch.sigmoid(-beta * (fused_margin - ref_margin)).detach(),
        "router_harm": router_harm.detach(),
        "branch_weight": weights.detach(),
        "worst_prefix_depth": (worst_prefix_index + 1).masked_fill(~valid, 0).detach(),
        "ref_worst_prefix_margin": ref_worst_margin.detach(),
        "fused_worst_prefix_margin": fused_worst_margin.detach(),
    }
    for depth in range(depth_count):
        diagnostics[f"prefix_margin_{depth + 1}"] = prefix_margins[:, :, depth].detach()
        diagnostics[f"reference_prefix_margin_{depth + 1}"] = ref_prefix_margins[:, :, depth].detach()
    metrics = {"sequence_preference_loss": sequence_loss,
               "prefix_ranking_loss": prefix_loss, "router_guard_loss": guard_loss,
               "usable_negatives": valid.float().sum() / max(valid.numel(), 1)}
    return auxiliary, metrics, diagnostics
