"""Conservative full-vocabulary mixture with a catalog-supported expert.

Only valid branching prefixes call this helper. It is not hard-constrained
decoding: all generator tokens retain probability when gate < 1.
"""

import torch


def catalog_probability_mixture(base_logits, expert_logits, legal, gate):
    """Each row must have at least one legal child; gate has shape [rows]."""
    base = base_logits.float()
    log_base = base.log_softmax(-1)
    log_expert = expert_logits.float().masked_fill(~legal, -torch.inf).log_softmax(-1)
    column_gate = gate.float().unsqueeze(-1)
    # Clamp only inside log(g); explicitly bypass at g=0 to preserve identity
    # even when the base probability is extremely small. No log(0) gradient.
    log_mix = torch.logaddexp(
        torch.log1p(-column_gate) + log_base,
        column_gate.clamp_min(1e-30).log() + log_expert,
    )
    mixed = base + (log_mix - log_base)
    return torch.where(column_gate.eq(0), base, mixed).to(base_logits.dtype)
