import json

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
T5Config = transformers.T5Config
T5ForConditionalGeneration = transformers.T5ForConditionalGeneration

from hierarchical_credit import SIDTrie, constrained_beam_search
from model import GenerativeSIDModel
from sid_catalog import build_fixed_sids
from sid_modules import PrototypeIndex, UncertaintyAwareTopologyRouter
from telemetry import MechanismTrace


def tiny_model(code_length=2, code_number=8):
    config = T5Config(
        num_layers=1,
        num_decoder_layers=1,
        d_model=16,
        d_ff=32,
        num_heads=2,
        d_kv=8,
        vocab_size=code_number,
        pad_token_id=0,
        decoder_start_token_id=0,
        eos_token_id=None,
    )
    return GenerativeSIDModel(
        {"num_beams": 4},
        T5ForConditionalGeneration(config),
        code_length=code_length,
        code_number=code_number,
    )


def tiny_catalog():
    codes = [[-1, -1], [0, 0], [0, 1], [1, 0], [1, 1]]
    embeddings = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ]
    )
    return codes, embeddings


def test_plain_and_routed_forward_are_catalog_decodable():
    torch.manual_seed(7)
    codes, embeddings = tiny_catalog()
    trie = SIDTrie(codes, semantic_depth=2)
    model = tiny_model()
    history = torch.tensor([[0, 0, 0, 1], [1, 0, 1, 1]])
    mask = torch.ones_like(history, dtype=torch.bool)
    labels = torch.tensor([[0, 0], [1, 1]])

    plain = model(input_ids=history, attention_mask=mask, labels=labels)
    assert plain.logits.shape == (2, 2, 8)
    assert plain.route_gates is None

    index = PrototypeIndex(trie, embeddings)
    router = UncertaintyAwareTopologyRouter(16, 4, 2, 2, 8, strength=0.5)
    model.configure_topology(router, index)
    routed = model(input_ids=history, attention_mask=mask, labels=labels)
    assert routed.route_gates.shape == (2, 2)
    assert torch.isfinite(routed.logits).all()
    loss = router.route_loss(routed.route_scores, labels, labels[:, :-1], index)
    assert torch.isfinite(loss)

    decoded = constrained_beam_search(model, history, mask, trie, num_beams=4, topk=2)
    for sid in decoded.sids.view(-1, 2).tolist():
        assert trie.is_valid_sid(sid)


class DummyTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_indices(self, values):
        return torch.tensor([[0], [0], [1]], device=values.device)


def test_fixed_sid_builder_resolves_collisions_without_training():
    embeddings = torch.randn(4, 3)
    codes = build_fixed_sids(DummyTokenizer(), embeddings, 1, 2, 4)
    assert codes == [[-1, -1], [0, 0], [0, 1], [1, 0]]
    assert len({tuple(row) for row in codes[1:]}) == 3


def test_mechanism_trace_round_trip(tmp_path):
    trace = MechanismTrace({"dataset": "tiny"}, sample_limit=2)
    trace.extend_module_1([{"depth": 1, "gate": 0.2}, {"depth": 2, "gate": 0.3}])
    trace.extend_module_2([{"depth": 1, "assigned_credit": -0.5}])
    path = tmp_path / "trace.json"
    trace.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "4.0"
    assert len(payload["module_1_samples"]) == 2
