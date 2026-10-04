"""Module 1: evidence-adaptive prefix memory (EAPM).

Every legal branching edge in a fixed SID trie owns a semantic prototype and a
small collaborative residual. A prefix-conditioned decoder query scores local
siblings. The resulting edge posterior is mixed with the generator posterior
through a bounded gate, so the module starts close to the unmodified model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hierarchical_credit.sid_trie import SID, SIDTrie
from .catalog_fusion import catalog_probability_mixture


@dataclass
class RoutingOutput:
    logits: torch.Tensor
    intent_vectors: torch.Tensor
    route_scores: torch.Tensor
    gates: torch.Tensor
    entropy: torch.Tensor
    margins: torch.Tensor
    gate_logits: torch.Tensor


@dataclass
class PackedPrototypeIndex:
    """CSR-like branching-prefix graph stored contiguously on one device."""

    node_keys: torch.Tensor
    node_ptr: torch.Tensor
    child_index: torch.Tensor
    prototypes: torch.Tensor
    sizes: torch.Tensor
    reliabilities: torch.Tensor
    depth_offsets: Tuple[int, ...]
    code_number: int
    semantic_depth: int


class PrototypeIndex:
    """Frozen subtree prototypes with train-support reliability estimates."""

    def __init__(
        self,
        trie: SIDTrie,
        item_embeddings: torch.Tensor,
        item_frequencies: Optional[torch.Tensor] = None,
        shrinkage_kappa: float = 64.0,
    ) -> None:
        embeddings = item_embeddings.detach().float().cpu()
        if embeddings.ndim != 2:
            raise ValueError("item_embeddings must have [num_items, dim] shape")
        if item_frequencies is None:
            frequencies = torch.ones(embeddings.size(0), dtype=torch.float32)
            frequencies[0] = 0.0
        else:
            frequencies = torch.as_tensor(item_frequencies, dtype=torch.float32).cpu()
            if frequencies.ndim != 1 or frequencies.numel() != embeddings.size(0):
                raise ValueError("item_frequencies must contain one value per item")
            frequencies = frequencies.clamp_min(0.0)

        self.embedding_dim = int(embeddings.size(1))
        self.shrinkage_kappa = max(float(shrinkage_kappa), 1e-6)
        self._entries: Dict[
            SID, Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._device_cache: Dict[
            torch.device,
            Dict[SID, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {}
        self._packed_host_cache: Dict[Tuple[int, int], Tuple[object, ...]] = {}
        self._packed_device_cache: Dict[
            Tuple[torch.device, int, int], PackedPrototypeIndex
        ] = {}

        for parent in trie.prefix_to_items:
            if len(parent) >= trie.code_length:
                continue
            children = tuple(trie.get_valid_children(parent))
            child_prototypes = []
            child_sizes = []
            child_reliabilities = []
            for token in children:
                child_items = trie.get_items_under_prefix(parent + (int(token),))
                index = torch.as_tensor(child_items, dtype=torch.long)
                child_prototypes.append(embeddings.index_select(0, index).mean(dim=0))
                child_sizes.append(len(child_items))
                evidence = frequencies.index_select(0, index).sum()
                child_reliabilities.append(
                    torch.sqrt(evidence / (evidence + self.shrinkage_kappa))
                )
            self._entries[parent] = (
                children,
                F.normalize(torch.stack(child_prototypes), dim=-1),
                torch.as_tensor(child_sizes, dtype=torch.float32),
                torch.stack(child_reliabilities).float(),
            )

    def child_count(self, prefix: Iterable[int]) -> int:
        entry = self._entries.get(tuple(int(x) for x in prefix))
        return 0 if entry is None else len(entry[0])

    def get_host(
        self, prefix: Iterable[int]
    ) -> Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]]:
        entry = self._entries.get(tuple(int(x) for x in prefix))
        if entry is None:
            return None
        children, prototypes, sizes, _reliabilities = entry
        return children, prototypes, sizes

    @staticmethod
    def _depth_offsets(code_number: int, semantic_depth: int) -> Tuple[int, ...]:
        if int(code_number) <= 0 or int(semantic_depth) <= 0:
            raise ValueError("code_number and semantic_depth must be positive")
        offsets = []
        offset = 0
        width = 1
        int64_max = torch.iinfo(torch.int64).max
        for _depth in range(int(semantic_depth)):
            offsets.append(offset)
            if offset > int64_max - width:
                raise ValueError("SID prefix key exceeds int64 range")
            offset += width
            if width <= int64_max // int(code_number):
                width *= int(code_number)
        return tuple(offsets)

    @staticmethod
    def _prefix_key(prefix: SID, code_number: int, offsets: Tuple[int, ...]) -> int:
        depth = len(prefix)
        if depth >= len(offsets):
            raise ValueError("Prefix is outside the packed semantic depth")
        local = 0
        for token in prefix:
            token = int(token)
            if token < 0 or token >= int(code_number):
                raise ValueError("SID token is outside the configured vocabulary")
            local = local * int(code_number) + token
        return int(offsets[depth]) + local

    def _pack_host(self, code_number: int, semantic_depth: int) -> Tuple[object, ...]:
        cache_key = (int(code_number), int(semantic_depth))
        cached = self._packed_host_cache.get(cache_key)
        if cached is not None:
            return cached
        offsets = self._depth_offsets(*cache_key)
        branching = []
        for prefix, (children, prototypes, sizes, reliabilities) in self._entries.items():
            if len(prefix) >= int(semantic_depth) or len(children) <= 1:
                continue
            branching.append(
                (
                    self._prefix_key(prefix, int(code_number), offsets),
                    children,
                    prototypes,
                    sizes,
                    reliabilities,
                )
            )
        branching.sort(key=lambda value: value[0])
        node_keys = torch.as_tensor([value[0] for value in branching], dtype=torch.long)
        widths = [len(value[1]) for value in branching]
        node_ptr = torch.zeros(len(widths) + 1, dtype=torch.long)
        if widths:
            node_ptr[1:] = torch.as_tensor(widths, dtype=torch.long).cumsum(dim=0)
            child_index = torch.cat(
                [torch.as_tensor(value[1], dtype=torch.long) for value in branching]
            )
            prototypes = torch.cat([value[2] for value in branching], dim=0)
            sizes = torch.cat([value[3] for value in branching], dim=0)
            reliabilities = torch.cat([value[4] for value in branching], dim=0)
        else:
            child_index = torch.empty(0, dtype=torch.long)
            prototypes = torch.empty(0, self.embedding_dim, dtype=torch.float32)
            sizes = torch.empty(0, dtype=torch.float32)
            reliabilities = torch.empty(0, dtype=torch.float32)
        packed = (
            node_keys,
            node_ptr,
            child_index,
            prototypes,
            sizes,
            reliabilities,
            offsets,
        )
        self._packed_host_cache[cache_key] = packed
        return packed

    @staticmethod
    def _canonical_device(device: torch.device) -> torch.device:
        result = torch.device(device)
        if result.type == "cuda" and result.index is None:
            result = torch.device("cuda", torch.cuda.current_device())
        return result

    def prepare_packed(
        self, device: torch.device, code_number: int, semantic_depth: int
    ) -> PackedPrototypeIndex:
        target_device = self._canonical_device(device)
        cache_key = (target_device, int(code_number), int(semantic_depth))
        cached = self._packed_device_cache.get(cache_key)
        if cached is not None:
            return cached
        host = self._pack_host(int(code_number), int(semantic_depth))
        node_keys, node_ptr, child_index, prototypes, sizes, reliabilities, offsets = host
        packed = PackedPrototypeIndex(
            node_keys=node_keys.to(target_device),
            node_ptr=node_ptr.to(target_device),
            child_index=child_index.to(target_device),
            prototypes=prototypes.to(target_device),
            sizes=sizes.to(target_device),
            reliabilities=reliabilities.to(target_device),
            depth_offsets=offsets,
            code_number=int(code_number),
            semantic_depth=int(semantic_depth),
        )
        self._packed_device_cache[cache_key] = packed
        return packed

    def edge_count(self, code_number: int, semantic_depth: int) -> int:
        return int(self._pack_host(code_number, semantic_depth)[2].numel())

    def _get_cached(
        self, prefix: Iterable[int], device: torch.device
    ) -> Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        key = tuple(int(x) for x in prefix)
        entry = self._entries.get(key)
        if entry is None:
            return None
        children, prototypes, sizes, reliabilities = entry
        target_device = self._canonical_device(device)
        device_entries = self._device_cache.setdefault(target_device, {})
        cached = device_entries.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(children, dtype=torch.long, device=target_device),
                prototypes.to(target_device),
                sizes.to(target_device),
                reliabilities.to(target_device),
            )
            device_entries[key] = cached
        child_index, cached_prototypes, cached_sizes, cached_reliabilities = cached
        return children, child_index, cached_prototypes, cached_sizes, cached_reliabilities

    def get(
        self, prefix: Iterable[int], device: torch.device
    ) -> Optional[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]]:
        entry = self._get_cached(prefix, device)
        if entry is None:
            return None
        children, _child_index, prototypes, sizes, _reliabilities = entry
        return children, prototypes, sizes

    def clear_device_cache(self) -> None:
        self._device_cache.clear()
        self._packed_device_cache.clear()

    def stats(self) -> Mapping[str, object]:
        widths = [len(children) for children, _p, _s, _r in self._entries.values()]
        reliability = torch.cat([entry[3] for entry in self._entries.values()])
        return {
            "prefixes": len(self._entries),
            "embedding_dim": self.embedding_dim,
            "mean_siblings": sum(widths) / max(len(widths), 1),
            "max_siblings": max(widths, default=0),
            "mean_edge_reliability": float(reliability.mean()) if reliability.numel() else 0.0,
            "shrinkage_kappa": self.shrinkage_kappa,
            "cached_devices": len(self._device_cache),
            "cached_prefixes": sum(len(values) for values in self._device_cache.values()),
            "packed_devices": len(self._packed_device_cache),
        }


class UncertaintyAwareTopologyRouter(nn.Module):
    """Prefix-conditioned local edge expert with bounded posterior fusion."""

    def __init__(
        self,
        hidden_size: int,
        semantic_size: int,
        semantic_depth: int,
        code_length: int,
        code_number: int,
        strength: float = 1.0,
        size_penalty: float = 0.0,
        min_gate: float = 0.0,
        max_gate: float = 0.30,
        gate_bias: float = -4.0,
        memory_rank: int = 16,
        logit_scale_init: float = 8.0,
        logit_scale_max: float = 32.0,
    ) -> None:
        super().__init__()
        if float(min_gate) != 0.0:
            raise ValueError("EAPM requires min_gate=0 for an identity-preserving start")
        self.hidden_size = int(hidden_size)
        self.semantic_size = int(semantic_size)
        self.semantic_depth = int(semantic_depth)
        self.code_length = int(code_length)
        self.code_number = int(code_number)
        self.size_penalty = float(size_penalty)
        self.max_gate = float(max_gate) * float(strength)
        if not 0.0 <= self.max_gate < 1.0:
            raise ValueError("max_gate * strength must be in [0, 1) for a bounded catalog mixture")
        self.memory_rank = max(1, int(memory_rank))
        if not 1.0 <= logit_scale_init <= logit_scale_max:
            raise ValueError("Require 1 <= logit_scale_init <= logit_scale_max")
        self.logit_scale_max = float(logit_scale_max)
        self.logit_scale = nn.Parameter(torch.full(
            (self.semantic_depth,), math.log(float(logit_scale_init))))

        self.intent_projection = nn.Linear(self.hidden_size, self.semantic_size)
        self.depth_offset = nn.Embedding(self.code_length, self.semantic_size)
        nn.init.zeros_(self.depth_offset.weight)
        middle = max(self.hidden_size // 2, 16)
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_size + 5, middle), nn.GELU(), nn.Linear(middle, 1)
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, float(gate_bias))
        self.edge_codes = nn.Parameter(torch.empty(0, self.memory_rank))
        self.edge_bias = nn.Parameter(torch.empty(0))
        self._catalog_bound = False
        self.edge_projection = nn.Linear(self.memory_rank, self.semantic_size, bias=False)
        nn.init.zeros_(self.edge_projection.weight)
        self.register_buffer("fusion_progress", torch.zeros(()), persistent=True)

    def bind_prototype_index(self, prototype_index: PrototypeIndex) -> None:
        edge_count = prototype_index.edge_count(self.code_number, self.semantic_depth)
        if self._catalog_bound and self.edge_codes.size(0) == edge_count:
            return
        if self._catalog_bound:
            raise RuntimeError("The router is already bound to a different catalog")
        codes = self.intent_projection.weight.new_empty(edge_count, self.memory_rank)
        nn.init.normal_(codes, mean=0.0, std=0.02)
        self.edge_codes = nn.Parameter(codes)
        self.edge_bias = nn.Parameter(codes.new_zeros(edge_count))
        self._catalog_bound = True

    def set_fusion_progress(self, value: float) -> None:
        self.fusion_progress.fill_(min(max(float(value), 0.0), 1.0))

    def memory_regularization(self) -> torch.Tensor:
        if self.edge_codes.numel() == 0:
            return self.intent_projection.weight.sum() * 0.0
        return (self.edge_codes.square().mean() + self.edge_bias.square().mean()
                + self.edge_projection.weight.square().mean())

    def score_scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(0.0, math.log(self.logit_scale_max)).exp()

    @staticmethod
    def _segment_softmax(
        values: torch.Tensor, groups: torch.Tensor, group_count: int
    ) -> torch.Tensor:
        maximum = torch.full(
            (group_count,), -torch.inf, dtype=values.dtype, device=values.device
        )
        maximum.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
        exponentials = torch.exp(values - maximum.index_select(0, groups))
        denominator = torch.zeros(group_count, dtype=values.dtype, device=values.device)
        denominator.scatter_add_(0, groups, exponentials)
        return exponentials / denominator.index_select(0, groups).clamp_min(1e-12)

    @staticmethod
    def _entropy_terms(probabilities: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        return (
            -probabilities * probabilities.clamp_min(1e-12).log()
            / widths.float().log().clamp_min(1e-6)
        )

    def _edge_vectors(
        self, packed: PackedPrototypeIndex, edge_ids: torch.Tensor
    ) -> torch.Tensor:
        prototypes = packed.prototypes.index_select(0, edge_ids)
        residual = self.edge_projection(self.edge_codes.index_select(0, edge_ids))
        reliability = packed.reliabilities.index_select(0, edge_ids).unsqueeze(1)
        return F.normalize(prototypes + reliability * residual, dim=-1)

    def _gate(
        self,
        hidden: torch.Tensor,
        base_prob: torch.Tensor,
        semantic_prob: torch.Tensor,
        groups: torch.Tensor,
        widths: torch.Tensor,
        depths: torch.Tensor,
        legal_mass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group_count = int(widths.numel())
        edge_widths = widths.index_select(0, groups)
        base_entropy = torch.zeros(group_count, device=base_prob.device)
        semantic_entropy = torch.zeros_like(base_entropy)
        base_entropy.scatter_add_(0, groups, self._entropy_terms(base_prob, edge_widths))
        semantic_entropy.scatter_add_(
            0, groups, self._entropy_terms(semantic_prob, edge_widths)
        )
        midpoint = 0.5 * (base_prob + semantic_prob)
        js_terms = 0.5 * (
            base_prob * (base_prob.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
            + semantic_prob
            * (semantic_prob.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())
        )
        disagreement = torch.zeros_like(base_entropy)
        disagreement.scatter_add_(0, groups, js_terms)
        disagreement = (disagreement / math.log(2.0)).clamp(0.0, 1.0)
        depth_fraction = ((depths.float() + 1.0) / max(self.code_length, 1)).unsqueeze(1)
        features = torch.cat(
            [
                hidden.detach(),
                base_entropy.unsqueeze(1),
                semantic_entropy.unsqueeze(1),
                disagreement.unsqueeze(1),
                depth_fraction,
                legal_mass.unsqueeze(1),
            ],
            dim=1,
        ).detach()
        gate_logits = self.gate_network(features).squeeze(1)
        gate = self.max_gate * self.fusion_progress * torch.sigmoid(gate_logits)
        return gate, base_entropy, gate_logits

    def forward(
        self,
        base_logits: torch.Tensor,
        encoder_state: torch.Tensor,
        decoder_states: torch.Tensor,
        prefixes: torch.Tensor,
        prototype_index: PrototypeIndex,
        target_tokens: Optional[torch.Tensor] = None,
        active_depth: Optional[int] = None,
    ) -> RoutingOutput:
        del encoder_state
        batch_size, positions, vocab_size = base_logits.shape
        if vocab_size != self.code_number or decoder_states.shape[:2] != base_logits.shape[:2]:
            raise ValueError("Router inputs do not match the configured generator")
        if not self._catalog_bound:
            raise RuntimeError("Bind the prototype index before constructing the optimizer")

        route_scores = torch.zeros_like(base_logits, dtype=torch.float32)
        gates = torch.zeros(batch_size, positions, device=base_logits.device)
        entropies = torch.zeros_like(gates)
        margins = torch.zeros_like(gates)
        raw_gates = torch.zeros_like(gates)
        calibrated = base_logits.float().clone()
        depth_ids = torch.arange(positions, device=base_logits.device).clamp_max(
            self.code_length - 1
        )
        intents = F.normalize(
            self.intent_projection(decoder_states.detach())
            + self.depth_offset(depth_ids).unsqueeze(0),
            dim=-1,
        )
        route_intents = intents
        depth_count = min(positions, self.semantic_depth)
        packed = prototype_index.prepare_packed(
            base_logits.device, self.code_number, self.semantic_depth
        )

        scales = self.score_scale()
        if depth_count > 0 and active_depth in (None, 0) and prototype_index.child_count(()) > 1:
            width = prototype_index.child_count(())
            root_edges = torch.arange(width, dtype=torch.long, device=base_logits.device)
            root_children = packed.child_index.index_select(0, root_edges)
            semantic_logits = torch.matmul(
                route_intents[:, 0], self._edge_vectors(packed, root_edges).t()
            ) * scales[0]
            if self.size_penalty:
                semantic_logits = semantic_logits - self.size_penalty * torch.log1p(
                    packed.sizes.index_select(0, root_edges)
                ).unsqueeze(0)
            semantic_logits = semantic_logits + self.edge_bias.index_select(
                0, root_edges
            ).unsqueeze(0) * packed.reliabilities.index_select(0, root_edges).unsqueeze(0)
            route_scores[:, 0, root_children] = semantic_logits
            local_base = base_logits[:, 0].index_select(1, root_children).float()
            base_prob = torch.softmax(local_base, dim=-1)
            semantic_prob = torch.softmax(semantic_logits, dim=-1)
            group_ids = torch.arange(batch_size, device=base_logits.device).repeat_interleave(width)
            widths = torch.full(
                (batch_size,), width, dtype=torch.long, device=base_logits.device
            )
            depths = torch.zeros(batch_size, dtype=torch.long, device=base_logits.device)
            legal = torch.zeros_like(base_logits[:, 0], dtype=torch.bool)
            legal[:, root_children] = True
            full_base_prob = base_logits[:, 0].float().softmax(-1)
            legal_mass = (full_base_prob * legal).sum(-1)
            gate, entropy, gate_logit = self._gate(
                decoder_states[:, 0],
                base_prob.reshape(-1),
                semantic_prob.reshape(-1),
                group_ids,
                widths,
                depths,
                legal_mass,
            )
            # Unlike the earlier local mixture, this can recover probability
            # mass allocated to illegal tokens while retaining full-vocab beam.
            calibrated[:, 0] = catalog_probability_mixture(
                base_logits[:, 0], route_scores[:, 0], legal, gate)
            gates[:, 0] = gate
            raw_gates[:, 0] = gate_logit
            entropies[:, 0] = entropy
            if target_tokens is not None:
                lookup = torch.full(
                    (self.code_number,), -1, dtype=torch.long, device=base_logits.device
                )
                lookup[root_children] = torch.arange(width, device=base_logits.device)
                target_position = lookup.index_select(0, target_tokens[:, 0].long())
                safe_target = target_position.clamp_min(0)
                target_score = semantic_logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
                competitors = semantic_logits.detach().clone()
                competitors.scatter_(1, safe_target.unsqueeze(1), -torch.inf)
                margins[:, 0] = torch.where(
                    target_position.ge(0), target_score.detach() - competitors.max(dim=1).values,
                    torch.zeros_like(target_score),
                )

        selected_depths = [depth for depth in range(1, depth_count)
                           if active_depth is None or depth == active_depth]
        ragged_depth_count = len(selected_depths)
        if ragged_depth_count > 0 and packed.node_keys.numel() > 0:
            prefix_code = prefixes[:, 0].long()
            query_columns = []
            for depth in range(1, depth_count):
                if depth in selected_depths:
                    query_columns.append(prefix_code + packed.depth_offsets[depth])
                if depth + 1 < depth_count:
                    prefix_code = prefix_code * self.code_number + prefixes[:, depth].long()
            query_keys = torch.stack(query_columns, dim=1).reshape(-1)
            insertion = torch.searchsorted(packed.node_keys, query_keys)
            candidates = insertion.clamp_max(packed.node_keys.numel() - 1)
            valid = insertion.lt(packed.node_keys.numel()) & packed.node_keys.index_select(
                0, candidates
            ).eq(query_keys)
            flat_slots = torch.nonzero(valid, as_tuple=False).flatten()
            if flat_slots.numel() > 0:
                node_ids = insertion.index_select(0, flat_slots)
                starts = packed.node_ptr.index_select(0, node_ids)
                widths = packed.node_ptr.index_select(0, node_ids + 1) - starts
                edge_count = int(widths.sum().item())
                group_ids = torch.repeat_interleave(
                    torch.arange(flat_slots.numel(), device=base_logits.device),
                    widths,
                    output_size=edge_count,
                )
                repeated_starts = torch.repeat_interleave(starts, widths, output_size=edge_count)
                output_starts = torch.repeat_interleave(
                    widths.cumsum(0) - widths, widths, output_size=edge_count
                )
                edge_ids = repeated_starts + (
                    torch.arange(edge_count, device=base_logits.device) - output_starts
                )
                occurrence_rows = torch.div(
                    flat_slots, ragged_depth_count, rounding_mode="floor"
                )
                selected_tensor = torch.as_tensor(selected_depths, device=base_logits.device)
                occurrence_depths = selected_tensor.index_select(
                    0, flat_slots.remainder(ragged_depth_count))
                edge_rows = occurrence_rows.index_select(0, group_ids)
                edge_depths = occurrence_depths.index_select(0, group_ids)
                edge_children = packed.child_index.index_select(0, edge_ids)
                edge_queries = route_intents[
                    occurrence_rows, occurrence_depths
                ].index_select(0, group_ids)
                semantic_logits = (
                    edge_queries * self._edge_vectors(packed, edge_ids)
                ).sum(dim=-1) * scales.index_select(0, edge_depths)
                if self.size_penalty:
                    semantic_logits = semantic_logits - self.size_penalty * torch.log1p(
                        packed.sizes.index_select(0, edge_ids)
                    )
                semantic_logits = semantic_logits + (
                    self.edge_bias.index_select(0, edge_ids)
                    * packed.reliabilities.index_select(0, edge_ids)
                )
                route_scores[edge_rows, edge_depths, edge_children] = semantic_logits

                local_base = base_logits[edge_rows, edge_depths, edge_children].float()
                base_prob = self._segment_softmax(
                    local_base, group_ids, int(flat_slots.numel())
                )
                semantic_prob = self._segment_softmax(
                    semantic_logits, group_ids, int(flat_slots.numel())
                )
                occurrence_base = base_logits[occurrence_rows, occurrence_depths]
                legal = torch.zeros_like(occurrence_base, dtype=torch.bool)
                legal[group_ids, edge_children] = True
                legal_mass = (occurrence_base.float().softmax(-1) * legal).sum(-1)
                gate, entropy, gate_logit = self._gate(
                    decoder_states[occurrence_rows, occurrence_depths],
                    base_prob,
                    semantic_prob,
                    group_ids,
                    widths,
                    occurrence_depths,
                    legal_mass,
                )
                calibrated[occurrence_rows, occurrence_depths] = catalog_probability_mixture(
                    occurrence_base, route_scores[occurrence_rows, occurrence_depths], legal, gate)
                gates[occurrence_rows, occurrence_depths] = gate
                raw_gates[occurrence_rows, occurrence_depths] = gate_logit
                entropies[occurrence_rows, occurrence_depths] = entropy

                if target_tokens is not None:
                    occurrence_targets = target_tokens[
                        occurrence_rows, occurrence_depths
                    ]
                    matches = edge_children.eq(
                        occurrence_targets.index_select(0, group_ids)
                    )
                    diagnostic = semantic_logits.detach()
                    target_scores = torch.zeros_like(gate)
                    target_scores.scatter_add_(
                        0,
                        group_ids,
                        torch.where(matches, diagnostic, torch.zeros_like(diagnostic)),
                    )
                    competitors = torch.where(
                        matches, torch.full_like(diagnostic, -torch.inf), diagnostic
                    )
                    competitor_max = torch.full_like(gate, -torch.inf)
                    competitor_max.scatter_reduce_(
                        0, group_ids, competitors, reduce="amax", include_self=True
                    )
                    margins[occurrence_rows, occurrence_depths] = target_scores - competitor_max

        return RoutingOutput(
            logits=calibrated.to(base_logits.dtype),
            intent_vectors=intents,
            route_scores=route_scores,
            gates=gates,
            entropy=entropies,
            margins=margins,
            gate_logits=raw_gates,
        )

    def route_loss(
        self,
        route_scores: torch.Tensor,
        labels: torch.Tensor,
        prefixes: torch.Tensor,
        prototype_index: PrototypeIndex,
    ) -> torch.Tensor:
        """Reference local-edge CE retained for external users and tests."""

        prefix_rows: List[List[int]] = prefixes.detach().cpu().tolist()
        label_rows: List[List[int]] = labels.detach().cpu().tolist()
        losses = []
        for depth in range(min(labels.size(1), self.semantic_depth)):
            groups: Dict[SID, List[int]] = {}
            for row, values in enumerate(prefix_rows):
                groups.setdefault(tuple(int(x) for x in values[:depth]), []).append(row)
            for prefix, rows in groups.items():
                if prototype_index.child_count(prefix) <= 1:
                    continue
                entry = prototype_index._get_cached(prefix, route_scores.device)
                if entry is None:
                    continue
                children, child_index, _prototypes, _sizes, _reliabilities = entry
                positions = {int(token): index for index, token in enumerate(children)}
                row_index = torch.as_tensor(rows, dtype=torch.long, device=route_scores.device)
                targets = torch.as_tensor(
                    [positions[int(label_rows[row][depth])] for row in rows],
                    dtype=torch.long,
                    device=route_scores.device,
                )
                local = route_scores.index_select(0, row_index)[:, depth].index_select(
                    1, child_index
                )
                losses.append(F.cross_entropy(local, targets, reduction="none"))
        return route_scores.sum() * 0.0 if not losses else torch.cat(losses).mean()
