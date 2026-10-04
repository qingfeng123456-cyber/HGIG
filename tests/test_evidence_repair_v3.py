"""Regression specifications for AutoDL; intentionally not executed locally."""

from types import SimpleNamespace
import torch
from hierarchical_credit import SIDTrie
from metrics import hierarchy_metric_sums, ranking_metric_sums
from sid_modules import PrototypeIndex, UncertaintyAwareTopologyRouter
from sid_modules.gate_supervision import competence_gate_loss
from sid_modules.negative_selection import select_repair_negatives
from sid_modules.branch_repair import repair_objective


def test_gate_supervision_is_train_only_and_detaches_both_experts():
    base = torch.tensor([[[2., 0.]], [[0., 2.]]], requires_grad=True)
    expert = torch.tensor([[[0., 2.]], [[2., 0.]]], requires_grad=True)
    gate = torch.zeros(2, 1, requires_grad=True)
    labels = torch.zeros(2, 1, dtype=torch.long)
    loss, detail = competence_gate_loss(base, expert, gate, labels,
                                        torch.ones_like(base, dtype=torch.bool), 1)
    assert detail["target"][0, 0] < 0.5 < detail["target"][1, 0]
    loss.backward()
    assert base.grad is None and expert.grad is None
    assert gate.grad[0, 0] > 0 and gate.grad[1, 0] < 0


def test_gate_auxiliary_ignores_singleton_decisions():
    scores = torch.zeros(1, 1, 3)
    gate = torch.zeros(1, 1, requires_grad=True)
    loss, detail = competence_gate_loss(scores, scores, gate, torch.tensor([[1]]),
                                        torch.tensor([[[False, True, False]]]), 1)
    loss.backward()
    assert loss == 0 and gate.grad.eq(0).all() and not detail["branching"].any()


def test_negative_mining_filters_history_and_covers_distinct_depths():
    sids = torch.tensor([[[0, 0, 0], [1, 1, 1], [1, 0, 0],
                          [1, 0, 1], [0, 1, 0], [0, 0, 1]]])
    decoded = SimpleNamespace(sids=sids, scores=torch.tensor([[6., 5., 4., 3., 2., 1.]]),
                              item_ids=torch.arange(1, 7).unsqueeze(0))
    labels = torch.tensor([[0, 0, 0]])
    history = torch.tensor([[1, 1, 1, -1, -1, -1]])
    negatives, valid, metrics, detail = select_repair_negatives(
        decoded, labels, history, history.ge(0), 3, 2)
    torch.testing.assert_close(negatives, sids[:, [2, 4, 5]])
    assert valid.all() and metrics["negative_selected_depths"] == 3
    torch.testing.assert_close(detail["negative_beam_rank"], torch.tensor([[3., 5., 6.]]))


def test_empty_negative_pool_keeps_finite_positive_placeholders():
    labels = torch.tensor([[0, 0]])
    decoded = SimpleNamespace(sids=torch.tensor([[[0, 0], [1, 1]]]),
                              scores=torch.tensor([[0., -1.]]), item_ids=torch.tensor([[1, 2]]))
    negatives, valid, metrics, _ = select_repair_negatives(
        decoded, labels, torch.tensor([[1, 1]]), torch.ones(1, 2, dtype=torch.bool), 2, 2)
    assert not valid.any() and negatives.eq(labels.unsqueeze(1)).all()
    assert all(torch.isfinite(value) for value in metrics.values())
    logits = torch.randn(3, 2, 2, requires_grad=True)
    outputs = SimpleNamespace(logits=logits, base_logits=logits)
    auxiliary, _, _ = repair_objective(outputs, SimpleNamespace(logits=logits.detach()),
                                       torch.cat([labels.unsqueeze(1), negatives], 1), valid, {})
    auxiliary.backward()
    assert auxiliary == 0 and torch.isfinite(logits.grad).all()


def test_absolute_repair_rewards_winning_not_only_reference_improvement():
    candidates = torch.tensor([[[0, 0], [1, 0]]])
    reference = SimpleNamespace(logits=torch.zeros(2, 2, 2))
    config = {"sequence_weight": 0., "router_guard_weight": 0.,
              "prefix_weight": 1., "prefix_margin": 0.5}
    losses = []
    for margin in (-2., 2.):
        logits = torch.zeros(2, 2, 2)
        logits[0, 0, 0] = margin
        outputs = SimpleNamespace(logits=logits, base_logits=logits)
        loss, _, _ = repair_objective(outputs, reference, candidates,
                                      torch.ones(1, 1, dtype=torch.bool), config)
        losses.append(loss)
    assert losses[1] < losses[0]


def test_last_depth_routing_matches_full_routing_at_every_used_position():
    codes = [[-1, -1, -1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    trie = SIDTrie(codes, semantic_depth=3)
    index = PrototypeIndex(trie, torch.randn(len(codes), 5))
    router = UncertaintyAwareTopologyRouter(8, 5, 3, 3, 4)
    router.bind_prototype_index(index)
    router.set_fusion_progress(1.0)
    labels = torch.tensor(codes[1:])
    logits = torch.randn(5, 3, 4)
    hidden = torch.randn(5, 3, 8)
    pooled = torch.randn(5, 8)
    full = router(logits, pooled, hidden, labels[:, :-1], index)
    for depth in range(3):
        partial = router(logits[:, :depth + 1], pooled, hidden[:, :depth + 1],
                         labels[:, :depth], index, active_depth=depth)
        torch.testing.assert_close(partial.logits[:, depth], full.logits[:, depth])


def test_vectorized_metric_api_preserves_invalid_ranks_and_prefix_counts():
    predictions = torch.tensor([[-1, 2], [3, 1]])
    targets = torch.tensor([2, 3])
    ordinary = ranking_metric_sums(predictions, targets, ["recall@1", "ndcg@2"])
    tensors = ranking_metric_sums(predictions, targets, ["recall@1", "ndcg@2"], return_tensors=True)
    assert ordinary == {key: float(value) for key, value in tensors.items()}
    predicted = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 0, 0], [-1, -1, -1]])
    gold = torch.tensor([[0, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0]])
    hierarchy = hierarchy_metric_sums(predicted, gold, 2, return_tensors=True)
    assert hierarchy["prefix_acc@1"] == 3 and hierarchy["prefix_acc@2"] == 2
    assert hierarchy["no_divergence"] == 1
    assert all(hierarchy[f"first_divergence@{depth}"] == 1 for depth in (1, 2, 3))
