"""Train-only comparative supervision for a target-free inference gate.

The detached soft target compares catalog expert versus full-vocabulary base, NOT an
observed user utility or a calibrated probability of recommendation success.
"""

import torch
import torch.nn.functional as F


def competence_gate_loss(base_logits, route_scores, gate_logits, labels,
                         child_mask, semantic_depth, temperature=1.0):
    depth = min(int(semantic_depth), labels.size(1))
    legal = child_mask[:, :depth]
    branching = legal.sum(-1).gt(1)
    # Never let the gate target/features train the trunk or make the expert
    # deliberately worse in order to simplify the auxiliary gate objective.
    base = base_logits[:, :depth].detach().float()
    expert = route_scores[:, :depth].detach().float().masked_fill(~legal, -1e9)
    target = labels[:, :depth].unsqueeze(-1)
    advantage = (expert.log_softmax(-1).gather(-1, target)
                 - base.log_softmax(-1).gather(-1, target)).squeeze(-1)
    soft_target = torch.sigmoid(advantage / max(float(temperature), 1e-6))
    terms = F.binary_cross_entropy_with_logits(
        gate_logits[:, :depth].float(), soft_target, reduction="none"
    )
    weights = branching.float()
    loss = (terms * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {"advantage": advantage, "target": soft_target,
                  "branching": branching}
