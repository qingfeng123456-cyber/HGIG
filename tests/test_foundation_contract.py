"""Critical regression tests for execution on the training host (not run locally)."""

from types import SimpleNamespace
import torch
import torch.nn.functional as F
from hierarchical_credit import SIDTrie
from hierarchical_credit.decoding import beam_search
from metrics import ranking_metric_sums
from sid_modules import PrototypeIndex, UncertaintyAwareTopologyRouter
from sid_modules.branch_repair import repair_objective
from trainer import _full_vocabulary_ce


def test_full_vocabulary_ce_matches_released_formula():
    values = torch.randn(3, 4, 8, requires_grad=True)
    targets = torch.randint(8, (3, 4))
    actual, _ = _full_vocabulary_ce(values, targets)
    expected = F.cross_entropy(values.reshape(-1, 8), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)


def test_zero_gate_preserves_every_logit_and_parameters_exist_before_optimizer():
    trie = SIDTrie([[-1, -1], [0, 0], [0, 1], [1, 0]], semantic_depth=1)
    index = PrototypeIndex(trie, torch.randn(4, 5))
    router = UncertaintyAwareTopologyRouter(8, 5, 1, 2, 4)
    router.bind_prototype_index(index)
    optimizer = torch.optim.AdamW(router.parameters())
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    logits = torch.randn(3, 2, 4, requires_grad=True)
    result = router(logits, torch.randn(3, 8), torch.randn(3, 2, 8),
                    torch.tensor([[0], [0], [1]]), index)
    torch.testing.assert_close(result.logits, logits, atol=0, rtol=0)
    assert id(router.edge_codes) in parameter_ids
    assert torch.count_nonzero(result.gates) == 0


def test_invalid_sid_keeps_its_original_rank():
    result = ranking_metric_sums(torch.tensor([[-1, 7, 9]]), torch.tensor([7]),
                                 ["recall@1", "recall@5", "ndcg@5"])
    assert result["recall@1"] == 0
    assert result["recall@5"] == 1
    assert abs(result["ndcg@5"] - 1 / torch.log2(torch.tensor(3.)).item()) < 1e-6


class CountingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encoder_calls = 0

    def encode(self, history, mask):
        self.encoder_calls += 1
        return SimpleNamespace(last_hidden_state=history.float().unsqueeze(-1))

    def forward(self, attention_mask, decoder_input_ids, encoder_outputs=None):
        batch, positions = decoder_input_ids.shape
        values = torch.tensor([0.1, 1.0, 2.0], device=decoder_input_ids.device)
        return SimpleNamespace(logits=values.expand(batch, positions, 3))


def test_beam_encodes_once_and_does_not_hide_invalid_top_prediction():
    model = CountingPolicy()
    trie = SIDTrie([[-1, -1], [0, 0], [1, 1]], semantic_depth=1)
    inputs = torch.ones(2, 4, dtype=torch.long)
    result = beam_search(model, inputs, inputs.bool(), trie, num_beams=3, topk=3)
    assert model.encoder_calls == 1
    assert result.sids[:, 0].eq(torch.tensor([2, 2])).all()
    assert result.item_ids[:, 0].eq(-1).all()


def test_repair_uses_shared_first_divergence_and_finite_gradients():
    candidates = torch.tensor([[[0, 0], [1, 0], [1, 1]]])
    logits = torch.randn(3, 2, 3, requires_grad=True)
    outputs = SimpleNamespace(logits=logits, base_logits=logits)
    reference = SimpleNamespace(logits=logits.detach().clone())
    loss, _metrics, diagnostic = repair_objective(
        outputs, reference, candidates, torch.ones(1, 2, dtype=torch.bool), {})
    assert diagnostic["depth"].eq(1).all()
    torch.testing.assert_close(diagnostic["branch_weight"], torch.tensor([[0.5, 0.5]]))
    assert diagnostic["router_harm"].eq(0).all()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
