"""Fusion contracts specified for the training host; not run locally."""

import torch
from sid_modules.catalog_fusion import catalog_probability_mixture


def test_zero_gate_is_exact_identity_even_with_extreme_logits():
    base = torch.tensor([[120., -120., -70., 3.]], requires_grad=True)
    expert = torch.tensor([[-2., 7., 0., 0.]], requires_grad=True)
    legal = torch.tensor([[True, True, False, False]])
    result = catalog_probability_mixture(base, expert, legal, torch.zeros(1))
    torch.testing.assert_close(result, base, atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(expert.grad).all()


def test_legal_mass_recovers_and_illegal_order_is_unchanged():
    base = torch.tensor([[0., 1., 2., 3.]], requires_grad=True)
    expert = torch.tensor([[2., 0., 0., 0.]], requires_grad=True)
    legal = torch.tensor([[True, True, False, False]])
    gate = torch.tensor([0.2], requires_grad=True)
    result = catalog_probability_mixture(base, expert, legal, gate)
    initial = base.softmax(-1)
    actual = result.softmax(-1)
    memory = expert.masked_fill(~legal, -torch.inf).softmax(-1)
    expected = (1 - gate.unsqueeze(1)) * initial + gate.unsqueeze(1) * memory
    torch.testing.assert_close(actual, expected)
    mass = initial[:, :2].sum(-1)
    torch.testing.assert_close(actual[:, :2].sum(-1), mass + gate * (1 - mass))
    torch.testing.assert_close(actual[:, 2:] / initial[:, 2:], torch.full((1, 2), 0.8))
    (-actual[:, 0].log()).sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in (base, expert, gate))


def test_full_catalog_reduces_to_ordinary_posterior_mixture():
    base = torch.randn(2, 5)
    expert = torch.randn(2, 5)
    gate = torch.tensor([0., 0.25])
    result = catalog_probability_mixture(base, expert, torch.ones_like(base, dtype=torch.bool), gate)
    expected = (1 - gate[:, None]) * base.softmax(-1) + gate[:, None] * expert.softmax(-1)
    torch.testing.assert_close(result.softmax(-1), expected)
