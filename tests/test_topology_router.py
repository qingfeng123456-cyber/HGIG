import torch
import torch.nn.functional as F

from hierarchical_credit import SIDTrie
from sid_modules import PrototypeIndex, UncertaintyAwareTopologyRouter


def catalog():
    codes = [[-1, -1], [0, 0], [0, 1], [1, 0]]
    embeddings = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    trie = SIDTrie(codes, semantic_depth=2)
    return trie, PrototypeIndex(trie, embeddings)


def router():
    result = UncertaintyAwareTopologyRouter(
        hidden_size=8,
        semantic_size=4,
        semantic_depth=2,
        code_length=2,
        code_number=4,
        strength=0.7,
        logit_scale_init=3.0,
        size_penalty=0.02,
        min_gate=0.0,
    )
    torch.manual_seed(19)
    for parameter in result.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    return result


def reference_route_loss(route_scores, labels, prefixes, prototype_index, semantic_depth):
    losses = []
    for row in range(labels.size(0)):
        for depth in range(min(labels.size(1), semantic_depth)):
            prefix = tuple(int(value) for value in prefixes[row, :depth].tolist())
            entry = prototype_index.get(prefix, route_scores.device)
            if entry is None:
                continue
            children, _prototypes, _sizes = entry
            if len(children) <= 1:
                continue
            target = int(labels[row, depth].item())
            if target not in children:
                raise RuntimeError
            child_index = torch.as_tensor(
                children, dtype=torch.long, device=route_scores.device
            )
            local = route_scores[row, depth].index_select(0, child_index).view(1, -1)
            position = torch.as_tensor(
                [children.index(target)], dtype=torch.long, device=route_scores.device
            )
            losses.append(F.cross_entropy(local, position))
    return route_scores.sum() * 0.0 if not losses else torch.stack(losses).mean()


def test_prototype_index_caches_one_copy_per_device_and_prefix():
    _trie, index = catalog()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    first = index._get_cached((), device)
    second = index._get_cached((), device)
    assert first is not None and second is not None
    assert first[1] is second[1]
    assert first[2] is second[2]
    assert first[3] is second[3]
    assert index.stats()["cached_devices"] == 1
    assert index.stats()["cached_prefixes"] == 1

    index._get_cached((0,), device)
    assert index.stats()["cached_prefixes"] == 2
    index.clear_device_cache()
    assert index.stats()["cached_devices"] == 0
    assert index.stats()["cached_prefixes"] == 0


def test_grouped_forward_matches_independent_rows_and_preserves_gradients():
    torch.manual_seed(23)
    _trie, index = catalog()
    topology_router = router()
    topology_router.bind_prototype_index(index)
    topology_router.set_fusion_progress(1.0)
    labels = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.long)
    prefixes = labels[:, :-1]
    base_logits = torch.randn(3, 2, 4, requires_grad=True)
    encoder_state = torch.randn(3, 8, requires_grad=True)
    decoder_states = torch.randn(3, 2, 8, requires_grad=True)

    grouped = topology_router(
        base_logits, encoder_state, decoder_states, prefixes, index, target_tokens=labels
    )
    independent = [
        topology_router(
            base_logits[row : row + 1],
            encoder_state[row : row + 1],
            decoder_states[row : row + 1],
            prefixes[row : row + 1],
            index,
            target_tokens=labels[row : row + 1],
        )
        for row in range(labels.size(0))
    ]
    for field in (
        "logits",
        "intent_vectors",
        "route_scores",
        "gates",
        "entropy",
        "margins",
    ):
        expected = torch.cat([getattr(output, field) for output in independent], dim=0)
        torch.testing.assert_close(getattr(grouped, field), expected, rtol=1e-5, atol=1e-6)

    grouped_loss = topology_router.route_loss(
        grouped.route_scores, labels, prefixes, index
    )
    reference_loss = reference_route_loss(
        grouped.route_scores, labels, prefixes, index, semantic_depth=2
    )
    torch.testing.assert_close(grouped_loss, reference_loss, rtol=1e-6, atol=1e-7)

    objective = grouped.logits.square().mean() + grouped_loss
    objective.backward()
    assert base_logits.grad is not None and torch.isfinite(base_logits.grad).all()
    # Router auxiliary CE deliberately detaches T5 states; the trunk remains
    # supervised by its primary objective instead of chasing a semantic proxy.
    assert encoder_state.grad is None
    assert decoder_states.grad is None
    projection_grad = topology_router.intent_projection.weight.grad
    assert projection_grad is not None and torch.isfinite(projection_grad).all()
    assert float(projection_grad.abs().sum()) > 0.0
