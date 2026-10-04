"""V4 prefix-repair contracts for the training host; not run locally."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sid_modules.branch_repair import cumulative_prefix_ranking, repair_objective


def test_shared_prefix_is_not_a_ranking_decision_or_a_gradient_source():
    candidates = torch.tensor([[[0, 0, 0], [0, 0, 1]]])
    # In deterministic teacher forcing, equal decoder prefixes share their
    # model parameters and selected-token probabilities. Use shared variables
    # here, not independent fake copies with inconsistent prefix gradients.
    shared = torch.tensor([-0.2, -0.3], requires_grad=True)
    suffix = torch.tensor([-0.5, -0.7], requires_grad=True)
    token_logp = torch.stack([
        torch.cat([shared, suffix[:1]]),
        torch.cat([shared, suffix[1:]]),
    ]).unsqueeze(0)
    loss, _margins, active = cumulative_prefix_ranking(
        token_logp, candidates, torch.tensor([[True]]), margin=0.5
    )
    assert active.tolist() == [[[False, False, True]]]
    torch.testing.assert_close(loss, F.softplus(torch.tensor(0.3)))
    loss.backward()
    torch.testing.assert_close(shared.grad, torch.zeros_like(shared), atol=0, rtol=0)
    assert suffix.grad[0] < 0 and suffix.grad[1] > 0


def test_duplicate_negative_prefixes_contribute_one_unit_of_mass():
    # The negatives share prefixes at depths 1/2 and differ at depth 3.
    candidates = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 0, 1]]])
    token_logp = torch.full((1, 3, 3), -1.0, requires_grad=True)
    loss, _margins, _active = cumulative_prefix_ranking(
        token_logp, candidates, torch.tensor([[True, True]]), margin=0.5
    )
    one_competitor = F.softplus(torch.tensor(0.5))
    two_competitors = torch.log1p(2.0 * torch.exp(torch.tensor(0.5)))
    torch.testing.assert_close(loss, (2.0 * one_competitor + two_competitors) / 3.0)


def test_duplicate_whole_leaf_does_not_change_prefix_loss():
    unique = torch.tensor([[[0, 0], [1, 0]]])
    repeated = torch.tensor([[[0, 0], [1, 0], [1, 0]]])
    unique_logp = torch.tensor([[[-0.4, -0.2], [-0.6, -0.3]]])
    repeated_logp = unique_logp[:, [0, 1, 1]]
    first = cumulative_prefix_ranking(unique_logp, unique, torch.tensor([[True]]))[0]
    second = cumulative_prefix_ranking(
        repeated_logp, repeated, torch.tensor([[True, True]])
    )[0]
    torch.testing.assert_close(first, second)


def test_no_usable_negatives_gives_exact_zero_with_finite_gradients():
    candidates = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 0, 1]]])
    logits = torch.zeros(3, 3, 2, requires_grad=True)
    outputs = SimpleNamespace(logits=logits, base_logits=logits)
    reference = SimpleNamespace(logits=logits.detach().clone())
    loss, metrics, diagnostics = repair_objective(
        outputs, reference, candidates, torch.tensor([[False, False]]), {}
    )
    assert loss.item() == 0.0
    assert metrics["prefix_ranking_loss"].item() == 0.0
    assert diagnostics["worst_prefix_depth"].eq(0).all()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)


def test_prefix_margins_use_cumulative_sums_without_length_normalization():
    candidates = torch.tensor([[[0, 0, 0], [1, 0, 0]]])
    token_logp = torch.tensor([[[-0.3, -0.2, -0.7], [-0.8, -0.1, -0.9]]])
    loss, margins, active = cumulative_prefix_ranking(
        token_logp, candidates, torch.tensor([[True]]), margin=0.5
    )
    expected = torch.tensor([[[0.5, 0.4, 0.6]]])
    torch.testing.assert_close(margins, expected)
    assert active.all()
    torch.testing.assert_close(loss, F.softplus(0.5 - expected).mean())


def test_reference_selects_one_worst_depth_for_both_policies():
    candidates = torch.tensor([[[0, 0], [1, 1]]])
    # Reference margin is 0 at depth 1, then positive at depth 2. The current
    # model has a worse depth 2, which must not change the reference selector.
    ref_logits = torch.tensor([[[0., 0.], [0., 0.]], [[0., 0.], [2., 0.]]])
    logits = torch.tensor([[[4., 0.], [0., 8.]], [[4., 0.], [0., 8.]]],
                          requires_grad=True)
    outputs = SimpleNamespace(logits=logits, base_logits=logits)
    _loss, _metrics, diagnostics = repair_objective(
        outputs, SimpleNamespace(logits=ref_logits), candidates,
        torch.tensor([[True]]), {}
    )
    assert diagnostics["worst_prefix_depth"].eq(1).all()
    torch.testing.assert_close(diagnostics["ref_worst_prefix_margin"],
                               diagnostics["reference_prefix_margin_1"])
    torch.testing.assert_close(diagnostics["fused_worst_prefix_margin"],
                               diagnostics["prefix_margin_1"])
    assert (diagnostics["prefix_margin_2"] < diagnostics["prefix_margin_1"]).all()
