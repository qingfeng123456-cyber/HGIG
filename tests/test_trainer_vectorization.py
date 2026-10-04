from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hierarchical_credit import SIDTrie
from telemetry import MechanismTrace
from trainer import Trainer, _build_catalog_child_mask, _catalog_sibling_ce


def _reference_catalog_ce(logits, labels, trie):
    per_depth = [[] for _ in range(labels.size(1))]
    for row in range(labels.size(0)):
        prefix = ()
        for depth in range(labels.size(1)):
            children = trie.get_valid_children(prefix)
            target = int(labels[row, depth])
            child_index = torch.tensor(children, device=logits.device)
            local = logits[row, depth].index_select(0, child_index).view(1, -1)
            position = torch.tensor([children.index(target)], device=logits.device)
            per_depth[depth].append(F.cross_entropy(local, position))
            prefix = prefix + (target,)
    depth_losses = [torch.stack(values).mean() for values in per_depth]
    return torch.stack(depth_losses).mean(), depth_losses


def test_vectorized_catalog_ce_matches_reference_value_and_gradient():
    codes = torch.tensor(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 2, 0],
        ]
    )
    trie = SIDTrie(codes.tolist(), semantic_depth=2)
    child_mask = _build_catalog_child_mask(codes, trie, code_number=5)
    target_items = torch.tensor([4, 2, 5, 1, 3])
    labels = codes.index_select(0, target_items)

    torch.manual_seed(23)
    initial = torch.randn(5, 3, 5, dtype=torch.float64)
    vector_logits = initial.clone().requires_grad_(True)
    reference_logits = initial.clone().requires_grad_(True)

    vector_loss, vector_depths = _catalog_sibling_ce(
        vector_logits, labels, target_items, child_mask
    )
    reference_loss, reference_depths = _reference_catalog_ce(
        reference_logits, labels, trie
    )

    torch.testing.assert_close(vector_loss, reference_loss, rtol=1e-12, atol=1e-12)
    for vector_depth, reference_depth in zip(vector_depths, reference_depths):
        torch.testing.assert_close(vector_depth, reference_depth, rtol=1e-12, atol=1e-12)

    vector_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        vector_logits.grad, reference_logits.grad, rtol=1e-12, atol=1e-12
    )


def test_catalog_mask_is_item_and_prefix_specific():
    codes = torch.tensor([[-1, -1], [0, 0], [0, 1], [1, 2]])
    trie = SIDTrie(codes.tolist(), semantic_depth=1)
    mask = _build_catalog_child_mask(codes, trie, code_number=4)

    assert not mask[0].any()
    assert mask[1, 0].nonzero().flatten().tolist() == [0, 1]
    assert mask[1, 1].nonzero().flatten().tolist() == [0, 1]
    assert mask[2, 1].nonzero().flatten().tolist() == [0, 1]
    assert mask[3, 1].nonzero().flatten().tolist() == [2]


def test_module_1_capture_returns_before_prototype_work_when_trace_is_full():
    class ExplodingPrototypeIndex:
        def get(self, *_args, **_kwargs):
            raise AssertionError("prototype lookup must not run for a full trace")

    trainer = Trainer.__new__(Trainer)
    trainer.trace = MechanismTrace({"dataset": "tiny"}, sample_limit=1)
    trainer.trace.extend_module_1([{"depth": 1, "gate": 0.25}])
    trainer.prototype_index = ExplodingPrototypeIndex()
    trainer.semantic_depth = 2
    trainer.config = {"telemetry": {"samples_per_batch": 8, "embedding_dimensions": 4}}
    outputs = SimpleNamespace(route_gates=torch.ones(1, 2))

    trainer._capture_module_1(outputs, torch.tensor([[0, 0]]))

    assert trainer.trace.payload["module_1_samples"] == [{"depth": 1, "gate": 0.25}]
