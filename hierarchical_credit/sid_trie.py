"""Catalog-constrained utilities for hierarchical discrete item identifiers.

The implementation deliberately does not assume that every identifier token has
the same role.  ``semantic_depth`` identifies the residual-quantization levels;
any remaining positions are catalog grounding / instance-resolution positions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


SID = Tuple[int, ...]


@dataclass(frozen=True)
class SIDMetadata:
    """Structural metadata exposed to generic hierarchy-aware modules."""

    code_length: int
    semantic_depth: int
    token_roles: Tuple[str, ...]

    @property
    def instance_depths(self) -> Tuple[int, ...]:
        return tuple(range(self.semantic_depth, self.code_length))


@dataclass(frozen=True)
class SIDTrieTensorTables:
    """Dense, device-resident transitions used by batched decoding.

    ``transitions[d][p, t]`` is the local node id at depth ``d + 1`` reached
    from prefix-node ``p`` by token ``t``; ``-1`` denotes an illegal edge.
    Keeping node ids local to each depth avoids storing full prefix tensors and
    lets beam search validate every candidate with one indexed GPU operation.
    """

    transitions: Tuple[torch.Tensor, ...]
    terminal_item_ids: torch.Tensor


class SIDTrie:
    """A deterministic trie over all valid catalog identifiers.

    Parameters
    ----------
    item_codes:
        Either a sequence indexed by item id (normally including the padding row
        at index 0) or a mapping from item id to its identifier.
    semantic_depth:
        Number of genuine semantic quantizer levels.  This must be derived from
        the tokenizer (for RQ-VAE, ``len(num_emb_list)``), not from code length.
    pad_item_id:
        Optional padding item.  Rows containing a negative token are ignored as
        well, so the original ``[-1, ...]`` padding convention is supported.
    token_roles:
        Optional explicit role names.  This keeps the component reusable for
        identifier schemes that do not use a single collision suffix.
    """

    def __init__(
        self,
        item_codes: Sequence[Sequence[int]] | Mapping[int, Sequence[int]],
        semantic_depth: int,
        pad_item_id: Optional[int] = 0,
        token_roles: Optional[Sequence[str]] = None,
    ) -> None:
        if isinstance(item_codes, Mapping):
            pairs = sorted((int(k), tuple(int(x) for x in v)) for k, v in item_codes.items())
        else:
            pairs = [(idx, tuple(int(x) for x in sid)) for idx, sid in enumerate(item_codes)]

        valid_pairs = [
            (item_id, sid)
            for item_id, sid in pairs
            if item_id != pad_item_id and sid and all(token >= 0 for token in sid)
        ]
        if not valid_pairs:
            raise ValueError("Cannot build SIDTrie from an empty catalog")

        code_lengths = {len(sid) for _, sid in valid_pairs}
        if len(code_lengths) != 1:
            raise ValueError(f"All catalog SIDs must have one length, got {sorted(code_lengths)}")
        self.code_length = code_lengths.pop()
        self.semantic_depth = int(semantic_depth)
        if not 0 < self.semantic_depth <= self.code_length:
            raise ValueError(
                f"semantic_depth must be in [1, {self.code_length}], got {self.semantic_depth}"
            )

        if token_roles is None:
            roles = ["semantic"] * self.semantic_depth
            roles.extend(["instance"] * (self.code_length - self.semantic_depth))
        else:
            roles = [str(role) for role in token_roles]
            if len(roles) != self.code_length:
                raise ValueError(
                    f"token_roles has length {len(roles)} but SID length is {self.code_length}"
                )
        self.metadata = SIDMetadata(
            code_length=self.code_length,
            semantic_depth=self.semantic_depth,
            token_roles=tuple(roles),
        )

        self._children: Dict[SID, set[int]] = defaultdict(set)
        self._prefix_to_items: Dict[SID, List[int]] = defaultdict(list)
        self._sid_to_item: Dict[SID, int] = {}
        self._item_to_sid: Dict[int, SID] = {}

        for item_id, sid in valid_pairs:
            if sid in self._sid_to_item:
                other = self._sid_to_item[sid]
                raise ValueError(
                    f"Full SID {sid} maps to both item {other} and item {item_id}; "
                    "append a deterministic instance-resolution token first"
                )
            self._sid_to_item[sid] = item_id
            self._item_to_sid[item_id] = sid
            for depth in range(self.code_length + 1):
                prefix = sid[:depth]
                self._prefix_to_items[prefix].append(item_id)
                if depth < self.code_length:
                    self._children[prefix].add(sid[depth])

        # Sorting once makes decoding, tests, and seeded sampling reproducible.
        self._children_sorted: Dict[SID, Tuple[int, ...]] = {
            prefix: tuple(sorted(children)) for prefix, children in self._children.items()
        }
        self._prefix_to_items_sorted: Dict[SID, Tuple[int, ...]] = {
            prefix: tuple(sorted(items)) for prefix, items in self._prefix_to_items.items()
        }

        # Prefixes are assigned deterministic, depth-local ids.  Tensor tables
        # are created lazily because the policy vocabulary and device are only
        # known when decoding starts.  The same tables are then reused by all
        # validation batches and epochs.
        self._prefixes_by_depth: Tuple[Tuple[SID, ...], ...] = tuple(
            tuple(sorted(prefix for prefix in self._prefix_to_items_sorted if len(prefix) == depth))
            for depth in range(self.code_length + 1)
        )
        self._prefix_ids_by_depth: Tuple[Dict[SID, int], ...] = tuple(
            {prefix: node_id for node_id, prefix in enumerate(prefixes)}
            for prefixes in self._prefixes_by_depth
        )
        self._minimum_vocab_size = 1 + max(token for _, sid in valid_pairs for token in sid)
        self._tensor_table_cache: Dict[Tuple[int, str], SIDTrieTensorTables] = {}

    def __len__(self) -> int:
        return len(self._sid_to_item)

    @property
    def item_to_sid(self) -> Mapping[int, SID]:
        return self._item_to_sid

    @property
    def sid_to_item(self) -> Mapping[SID, int]:
        return self._sid_to_item

    @property
    def prefix_to_items(self) -> Mapping[SID, Tuple[int, ...]]:
        return self._prefix_to_items_sorted

    def get_valid_children(self, prefix: Iterable[int] = ()) -> List[int]:
        key = tuple(int(x) for x in prefix)
        return list(self._children_sorted.get(key, ()))

    def is_valid_prefix(self, prefix: Iterable[int]) -> bool:
        key = tuple(int(x) for x in prefix)
        return len(key) <= self.code_length and key in self._prefix_to_items_sorted

    def is_valid_sid(self, sid: Iterable[int]) -> bool:
        return tuple(int(x) for x in sid) in self._sid_to_item

    def decode_sid(self, sid: Iterable[int]) -> Optional[int]:
        return self._sid_to_item.get(tuple(int(x) for x in sid))

    def encode_item(self, item_id: int) -> SID:
        try:
            return self._item_to_sid[int(item_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown catalog item id: {item_id}") from exc

    def get_items_under_prefix(self, prefix: Iterable[int]) -> List[int]:
        key = tuple(int(x) for x in prefix)
        return list(self._prefix_to_items_sorted.get(key, ()))

    def make_prefix_allowed_tokens_fn(
        self,
        decoder_start_token_id: Optional[int] = 0,
    ) -> Callable[[int, Sequence[int]], List[int]]:
        """Return a HuggingFace-compatible, target-independent constraint.

        ``batch_id`` is intentionally unused: catalog validity is shared by all
        users and no label/target is captured by the closure.
        """

        def allowed(_batch_id: int, decoder_tokens: Sequence[int]) -> List[int]:
            if hasattr(decoder_tokens, "tolist"):
                tokens = decoder_tokens.tolist()
            else:
                tokens = list(decoder_tokens)
            tokens = [int(x) for x in tokens]
            if (
                decoder_start_token_id is not None
                and tokens
                and tokens[0] == int(decoder_start_token_id)
            ):
                tokens = tokens[1:]
            return self.get_valid_children(tokens)

        return allowed

    @property
    def minimum_vocab_size(self) -> int:
        """Smallest decoder vocabulary that can represent the catalog."""

        return self._minimum_vocab_size

    def tensor_tables(
        self,
        vocab_size: int,
        device: torch.device | str,
    ) -> SIDTrieTensorTables:
        """Return cached dense trie transitions on ``device``.

        The tables contain only integer transitions; an edge is legal exactly
        when its entry is non-negative.  Consequently constrained decoding does
        not need Python prefix lookups, per-beam tensor construction, or GPU to
        CPU ``tolist`` synchronization inside the autoregressive loop.
        """

        vocab_size = int(vocab_size)
        if vocab_size < self._minimum_vocab_size:
            raise ValueError(
                f"Policy vocabulary ({vocab_size}) cannot represent catalog token "
                f"{self._minimum_vocab_size - 1}"
            )
        resolved_device = torch.device(device)
        cache_key = (vocab_size, str(resolved_device))
        cached = self._tensor_table_cache.get(cache_key)
        if cached is not None:
            return cached

        cpu_transitions: List[torch.Tensor] = []
        for depth in range(self.code_length):
            parents = self._prefixes_by_depth[depth]
            child_ids = self._prefix_ids_by_depth[depth + 1]
            table = torch.full(
                (len(parents), vocab_size),
                -1,
                dtype=torch.int32,
            )
            row_ids: List[int] = []
            token_ids: List[int] = []
            next_ids: List[int] = []
            for row_id, prefix in enumerate(parents):
                for token in self._children_sorted.get(prefix, ()):
                    row_ids.append(row_id)
                    token_ids.append(token)
                    next_ids.append(child_ids[prefix + (token,)])
            if row_ids:
                table[
                    torch.as_tensor(row_ids, dtype=torch.long),
                    torch.as_tensor(token_ids, dtype=torch.long),
                ] = torch.as_tensor(next_ids, dtype=torch.int32)
            cpu_transitions.append(table)

        leaf_ids = self._prefix_ids_by_depth[self.code_length]
        terminal_item_ids = torch.full(
            (len(leaf_ids),),
            -1,
            dtype=torch.long,
        )
        for sid, item_id in self._sid_to_item.items():
            terminal_item_ids[leaf_ids[sid]] = int(item_id)
        if bool(terminal_item_ids.lt(0).any()):
            raise RuntimeError("SID trie terminal table contains an unmapped leaf")

        result = SIDTrieTensorTables(
            transitions=tuple(table.to(resolved_device) for table in cpu_transitions),
            terminal_item_ids=terminal_item_ids.to(resolved_device),
        )
        self._tensor_table_cache[cache_key] = result
        return result

    def stats(self) -> Dict[str, object]:
        branching_by_depth: Dict[int, List[int]] = defaultdict(list)
        for prefix, children in self._children_sorted.items():
            branching_by_depth[len(prefix)].append(len(children))
        return {
            "catalog_size": len(self),
            "code_length": self.code_length,
            "semantic_depth": self.semantic_depth,
            "token_roles": list(self.metadata.token_roles),
            "prefix_count": len(self._prefix_to_items_sorted),
            "branching": {
                str(depth + 1): {
                    "parents": len(values),
                    "mean": sum(values) / max(len(values), 1),
                    "max": max(values, default=0),
                }
                for depth, values in sorted(branching_by_depth.items())
            },
        }
