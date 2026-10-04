"""Training and evaluation for a generic fixed-SID generative recommender."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from transformers.optimization import get_scheduler

from hierarchical_credit import SIDTrie
from hierarchical_credit.decoding import beam_search
from sid_modules.branch_repair import repair_objective
from sid_modules.gate_supervision import competence_gate_loss
from sid_modules.negative_selection import select_repair_negatives
from utils import init_seed
from metrics import hierarchy_metric_sums, ranking_metric_sums
from sid_catalog import catalog_hash
from sid_modules import PrototypeIndex, UncertaintyAwareTopologyRouter
from telemetry import MechanismTrace


def _target_item_frequencies(dataset, num_items: int) -> torch.Tensor:
    """Count train-target evidence without touching validation/test labels."""

    frequencies = torch.zeros(int(num_items), dtype=torch.float32)
    records = getattr(dataset, "data", None)
    if records is not None:
        for record in records:
            target = int(record["target"][0])
            if 0 <= target < frequencies.numel():
                frequencies[target] += 1.0
        return frequencies
    for _history, target in dataset:
        item_id = int(target[0])
        if 0 <= item_id < frequencies.numel():
            frequencies[item_id] += 1.0
    return frequencies


def _build_catalog_child_mask(
    item_codes: torch.Tensor,
    trie: SIDTrie,
    code_number: int,
) -> torch.Tensor:
    """Precompute valid sibling tokens for every catalog item and SID depth.

    Row zero is the padding item and intentionally remains all-false.  For a
    real item, the mask at a depth contains exactly the children of the prefix
    preceding that item's target token.  Building this once removes Trie and
    Python work from every supervised micro-batch.
    """

    if item_codes.ndim != 2:
        raise ValueError("item_codes must have [num_items, code_length] shape")
    code_number = int(code_number)
    if code_number <= 0:
        raise ValueError("code_number must be positive")

    codes = item_codes.detach().cpu().long()
    mask = torch.zeros(
        codes.size(0), codes.size(1), code_number, dtype=torch.bool
    )
    for item_id, sid in trie.item_to_sid.items():
        if item_id < 0 or item_id >= codes.size(0):
            raise ValueError(f"Trie item id {item_id} is outside item_codes")
        if tuple(codes[item_id].tolist()) != tuple(sid):
            raise ValueError(f"Trie SID disagrees with item_codes for item {item_id}")
        prefix: Tuple[int, ...] = ()
        for depth, target in enumerate(sid):
            children = trie.get_valid_children(prefix)
            if target not in children:
                raise ValueError(f"Catalog target {target} is invalid below prefix {prefix}")
            if children and (min(children) < 0 or max(children) >= code_number):
                raise ValueError("Catalog token is outside the configured vocabulary")
            mask[item_id, depth, children] = True
            prefix = prefix + (target,)
    return mask


def _catalog_sibling_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_items: torch.Tensor,
    catalog_child_mask: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Vectorized CE over the valid siblings selected by each target item."""

    if logits.ndim != 3:
        raise ValueError("logits must have [batch, code_length, vocabulary] shape")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match the batch and code-length dimensions")
    target_items = target_items.reshape(-1).long()
    if target_items.size(0) != logits.size(0):
        raise ValueError("target_items must contain one item id per batch row")
    if catalog_child_mask.ndim != 3 or catalog_child_mask.shape[1:] != logits.shape[1:]:
        raise ValueError("catalog_child_mask shape does not match logits")

    mask_items = target_items.to(device=catalog_child_mask.device)
    valid_children = catalog_child_mask.index_select(0, mask_items)
    if valid_children.device != logits.device:
        valid_children = valid_children.to(logits.device)
    masked_logits = logits.masked_fill(~valid_children, float("-inf"))
    token_losses = F.cross_entropy(
        masked_logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    depth_losses = list(token_losses.mean(dim=0).unbind(0))
    return torch.stack(depth_losses).mean(), depth_losses


def _full_vocabulary_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Cross-entropy over the complete SID token vocabulary."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must have [batch, code_length, vocabulary] shape")
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    depth_losses = list(token_losses.mean(dim=0).unbind(0))
    return token_losses.mean(), depth_losses


def _catalog_symmetric_kl(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    target_items: torch.Tensor,
    catalog_child_mask: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional dropout consistency over genuine catalog branches.

    Both views see the same history and target.  Their independent dropout
    masks produce two stochastic sub-models, while the target-prefix catalog
    mask ensures that regularization never assigns mass to an illegal SID
    child.  Singleton prefixes are excluded because they contain no decision.
    The calculation is kept in float32 with a finite mask sentinel so mixed
    precision cannot create ``0 * inf`` NaNs at invalid vocabulary entries.
    """

    if first_logits.shape != second_logits.shape or first_logits.ndim != 3:
        raise ValueError("paired logits must share [batch, code_length, vocabulary] shape")
    if float(temperature) <= 0.0:
        raise ValueError("consistency temperature must be positive")
    target_items = target_items.reshape(-1).long()
    if target_items.size(0) != first_logits.size(0):
        raise ValueError("target_items must contain one item id per paired row")
    if (
        catalog_child_mask.ndim != 3
        or catalog_child_mask.shape[1:] != first_logits.shape[1:]
    ):
        raise ValueError("catalog_child_mask shape does not match paired logits")

    mask_items = target_items.to(device=catalog_child_mask.device)
    valid_children = catalog_child_mask.index_select(0, mask_items)
    if valid_children.device != first_logits.device:
        valid_children = valid_children.to(first_logits.device)
    branching = valid_children.sum(dim=-1).gt(1)

    scale = float(temperature)
    sentinel = -1.0e4
    first = (first_logits.float() / scale).masked_fill(~valid_children, sentinel)
    second = (second_logits.float() / scale).masked_fill(~valid_children, sentinel)
    first_log_prob = F.log_softmax(first, dim=-1)
    second_log_prob = F.log_softmax(second, dim=-1)
    first_prob = first_log_prob.exp()
    second_prob = second_log_prob.exp()

    forward_terms = first_prob * (first_log_prob - second_log_prob)
    reverse_terms = second_prob * (second_log_prob - first_log_prob)
    symmetric = 0.5 * (
        torch.where(valid_children, forward_terms, torch.zeros_like(forward_terms)).sum(dim=-1)
        + torch.where(valid_children, reverse_terms, torch.zeros_like(reverse_terms)).sum(dim=-1)
    )
    weights = branching.to(symmetric.dtype)
    loss = (symmetric * weights).sum() / weights.sum().clamp_min(1.0)
    return loss * (scale * scale), branching


def _branching_route_ce(
    route_scores: torch.Tensor,
    labels: torch.Tensor,
    target_items: torch.Tensor,
    catalog_child_mask: torch.Tensor,
    semantic_depth: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Module-1 CE over genuine branching decisions only.

    This is algebraically identical to computing a separate cross entropy for
    every target prefix with more than one valid child and averaging all those
    terms.  Reusing the precomputed catalog mask avoids a second Python prefix
    grouping pass after the router forward pass.
    """

    if route_scores.ndim != 3:
        raise ValueError("route_scores must have [batch, code_length, vocabulary] shape")
    depth = min(
        int(semantic_depth),
        route_scores.size(1),
        labels.size(1),
        catalog_child_mask.size(1),
    )
    if depth <= 0:
        empty = torch.zeros(
            labels.size(0), 0, dtype=torch.bool, device=route_scores.device
        )
        return route_scores.sum() * 0.0, empty

    mask_items = target_items.reshape(-1).long().to(catalog_child_mask.device)
    valid_children = catalog_child_mask.index_select(0, mask_items)[:, :depth]
    if valid_children.device != route_scores.device:
        valid_children = valid_children.to(route_scores.device)
    local_scores = route_scores[:, :depth].masked_fill(~valid_children, float("-inf"))
    branching = valid_children.sum(dim=-1).gt(1)
    token_losses = F.cross_entropy(
        local_scores.reshape(-1, route_scores.size(-1)),
        labels[:, :depth].reshape(-1),
        reduction="none",
    ).view(labels.size(0), depth)
    weights = branching.to(token_losses.dtype)
    route_loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
    return route_loss, branching


def _accumulate_detached(
    totals: Dict[str, object], key: str, value: object
) -> None:
    """Accumulate scalar metrics without synchronizing CUDA on every batch."""

    if torch.is_tensor(value):
        scalar = value.detach().to(dtype=torch.float64)
        if scalar.numel() != 1:
            raise ValueError(f"Metric {key} must be scalar")
        totals[key] = scalar if key not in totals else totals[key] + scalar
    else:
        scalar = float(value)
        totals[key] = scalar if key not in totals else float(totals[key]) + scalar


def _metric_means(totals: Mapping[str, object], count: int) -> Dict[str, float]:
    """Materialize epoch means, synchronizing tensor totals only once each."""

    result: Dict[str, float] = {}
    for key, total in totals.items():
        if torch.is_tensor(total):
            result[key] = float((total / count).item())
        else:
            result[key] = float(total) / count
    return result


class Trainer:
    """Train the complete fixed-SID model with EAPM and RCBR.

    EAPM fuses catalog evidence into autoregressive generation. RCBR refines
    the resulting model with beam-mined candidates and a frozen reference.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        model: torch.nn.Module,
        accelerator,
        item_codes: torch.Tensor,
        item_embeddings: torch.Tensor,
        train_data,
        valid_data,
        test_data,
        policy_data,
    ) -> None:
        self.config = dict(config)
        self.accelerator = accelerator
        self.device = accelerator.device
        self.code_number = int(config["code_num"])
        self.code_length = int(config["code_length"])
        self.semantic_depth = int(config.get("semantic_depth", self.code_length))
        self.decoding_protocol = str(config.get("decoding_protocol", "full_vocabulary"))
        self._common_pretraining = False
        self.primary_ce_mode = str(
            config.get("primary_ce_mode", "full_vocabulary")
        ).strip().lower()
        if self.primary_ce_mode not in {"full_vocabulary", "catalog_siblings"}:
            raise ValueError(
                "primary_ce_mode must be 'full_vocabulary' or 'catalog_siblings'"
            )
        self.supervised_gradient_accumulation_steps = int(
            config.get("gradient_accumulation_steps", 1)
        )
        consistency = config.get("stage1_consistency", {})
        self.stage1_consistency_enabled = False
        if consistency.get("enabled", False):
            raise ValueError("Two-view consistency is unsupported; set enabled=false")
        self.stage1_consistency_weight = float(consistency.get("weight", 0.0))
        self.stage1_consistency_temperature = float(
            consistency.get("temperature", 1.0)
        )
        # A paired forward contains two stochastic views.  Splitting an outer
        # batch in half keeps its largest T5 activation batch no larger than the
        # configured physical batch (512 -> 256 + 256).
        self.stage1_pair_chunk_size = max(1, int(config["batch_size"]) // 2)
        self.item_codes = item_codes.long().to(self.device)
        self.item_embeddings = item_embeddings.detach().float().cpu()
        self.trie = SIDTrie(
            self.item_codes.cpu().tolist(),
            semantic_depth=self.semantic_depth,
            token_roles=config.get("token_roles"),
        )
        self.catalog_child_mask = _build_catalog_child_mask(
            self.item_codes, self.trie, self.code_number
        ).to(self.device)
        self.catalog_version = catalog_hash(self.item_codes.cpu().tolist())
        modules = config.get("modules", {})
        # The public runner contains the complete model, with both modules active.
        self.module_1_enabled = True
        self.module_2_enabled = True
        module_1 = modules.get("module_1", {})
        train_frequencies = (
            _target_item_frequencies(train_data.dataset, self.item_embeddings.size(0))
            if self.module_1_enabled
            else None
        )
        # Build the catalog prototype store once before training.
        self.prototype_index: Optional[PrototypeIndex] = (
            PrototypeIndex(
                self.trie,
                self.item_embeddings,
                item_frequencies=train_frequencies,
                shrinkage_kappa=float(module_1.get("support_shrinkage", 64.0)),
            )
            if self.module_1_enabled
            else None
        )
        if self.module_1_enabled:
            if self.prototype_index is None:
                raise RuntimeError("Module 1 requires a prototype index")
            router = UncertaintyAwareTopologyRouter(
                hidden_size=model.hidden_size,
                semantic_size=self.prototype_index.embedding_dim,
                semantic_depth=self.semantic_depth,
                code_length=self.code_length,
                code_number=self.code_number,
                strength=float(module_1.get("strength", 1.0)),
                size_penalty=float(module_1.get("size_penalty", 0.0)),
                min_gate=float(module_1.get("min_gate", 0.0)),
                max_gate=float(module_1.get("max_gate", 0.25)),
                gate_bias=float(module_1.get("gate_bias", -4.0)),
                memory_rank=int(module_1.get("memory_rank", 16)),
                logit_scale_init=float(module_1.get("logit_scale_init", 8.0)),
                logit_scale_max=float(module_1.get("logit_scale_max", 32.0)),
            )
            model.configure_topology(router, self.prototype_index)
        else:
            model.configure_topology(None, None)

        # Separate sample order from worker startup RNG. Persistent worker
        # reuse must not change sample order when a warmstart is cached.
        self.train_order_generator = getattr(train_data.sampler, "generator", None)
        self.policy_order_generator = getattr(policy_data.sampler, "generator", None)
        self.optimizer = self._optimizer(model, float(config["learning_rate"]))
        self._assert_optimizer_coverage(model, self.optimizer)
        self.training_contract = self._build_training_contract(
            model=model,
            train_data=train_data,
            gradient_accumulation_steps=int(
                config.get("gradient_accumulation_steps", 1)
            ),
        )
        total_steps = max(
            1,
            int(self.training_contract["optimizer_updates_per_epoch"])
            * int(config["epochs"]),
        )
        self.scheduler = get_scheduler(
            name=str(config.get("lr_scheduler_type", "cosine")),
            optimizer=self.optimizer,
            num_warmup_steps=int(config.get("warmup_steps", 0)),
            num_training_steps=total_steps,
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_data,
            self.valid_data,
            self.test_data,
            self.policy_data,
        ) = accelerator.prepare(
            model,
            self.optimizer,
            self.scheduler,
            train_data,
            valid_data,
            test_data,
            policy_data,
        )
        if self.accelerator.is_main_process:
            print(
                "[stage1-contract] "
                + json.dumps(self.training_contract, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        if self.module_1_enabled:
            if self.prototype_index is None:
                raise RuntimeError("Module 1 requires a prototype index")
            # Upload one packed prototype graph before the progress bar starts.
            # This replaces thousands of lazy per-prefix host-to-device copies
            # during the first epoch with one contiguous transfer.
            packed = self.prototype_index.prepare_packed(
                self.device,
                code_number=self.code_number,
                semantic_depth=self.semantic_depth,
            )
            if self.accelerator.is_main_process:
                print(
                    "[module1] routing=hybrid(root_gemm+packed_deep) | "
                    f"branch_nodes={packed.node_keys.numel()} | "
                    f"prototype_edges={packed.child_index.numel()} | "
                    "singleton_prefixes=skipped",
                    flush=True,
                )

        self.metric_names = [x.strip().lower() for x in str(config["metrics"]).split(",")]
        self.valid_metric = str(config["valid_metric"]).lower()
        self.max_topk = max(int(name.split("@", 1)[1]) for name in self.metric_names)
        self.output_dir = Path(str(config["results_path"]))
        self.checkpoint_dir = Path(str(config["save_path"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint: Optional[str] = None
        self.best_score = float("-inf")
        self.best_validation: Optional[Dict[str, float]] = None
        self.trace = MechanismTrace(
            {
                "dataset": config["dataset"],
                "run_name": config["run_name"],
                "seed": int(config["seed"]),
                "module_1": self.module_1_enabled,
                "module_2": self.module_2_enabled,
                "stage1_consistency": {
                    "enabled": self.stage1_consistency_enabled,
                    "weight": self.stage1_consistency_weight,
                    "temperature": self.stage1_consistency_temperature,
                },
                "catalog_version": self.catalog_version,
                "training_contract": dict(self.training_contract),
                "catalog": self.trie.stats(),
                "module_names": {"module_1": "EAPM", "module_2": "RCBR"},
                "revision": "hgig-public-v1",
                "decoding_protocol": self.decoding_protocol,
                "prototypes": (
                    self.prototype_index.stats()
                    if self.prototype_index is not None
                    else {"enabled": False}
                ),
            },
            sample_limit=int(config.get("telemetry", {}).get("sample_limit", 4096)),
        )

    def _optimizer(self, model: torch.nn.Module, learning_rate: float):
        return optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
        )

    @staticmethod
    def _assert_optimizer_coverage(model: torch.nn.Module, optimizer) -> None:
        """Fail fast if any trainable model tensor is absent from the optimizer."""

        named_trainable = {
            id(parameter): name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
        trainable_ids = set(named_trainable)
        missing = sorted(
            named_trainable[parameter_id]
            for parameter_id in trainable_ids - optimizer_ids
        )
        extra = optimizer_ids - trainable_ids
        if missing or extra or len(optimizer_parameters) != len(optimizer_ids):
            details = []
            if missing:
                details.append(f"missing={missing[:8]}")
            if extra:
                details.append(f"non_trainable_or_foreign={len(extra)}")
            if len(optimizer_parameters) != len(optimizer_ids):
                details.append("duplicate_parameter_references=true")
            raise AssertionError(
                "Optimizer does not exactly cover every requires_grad model parameter: "
                + ", ".join(details)
            )

    def _build_training_contract(
        self,
        model: torch.nn.Module,
        train_data,
        gradient_accumulation_steps: int,
    ) -> Dict[str, object]:
        """Return a machine-readable audit of Stage-1 architecture and budget."""

        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        backbone = getattr(model, "backbone", None)
        if (
            backbone is None
            or not hasattr(backbone, "encoder")
            or not hasattr(backbone, "decoder")
        ):
            raise AssertionError(
                "Stage-1 model must expose a complete T5 encoder-decoder backbone"
            )

        def parameter_counts(module: torch.nn.Module) -> Tuple[int, int]:
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            return int(trainable), int(total)

        backbone_trainable, backbone_total = parameter_counts(backbone)
        model_trainable, model_total = parameter_counts(model)
        configured_batches = len(train_data)
        batch_limit = self.config.get("max_batches_per_epoch")
        train_batches = (
            min(configured_batches, int(batch_limit))
            if batch_limit not in (None, 0, "null")
            else configured_batches
        )
        max_history = int(self.config["max_his_len"])
        stage1_views = 2 if self.stage1_consistency_enabled else 1
        paired_chunk = min(self.stage1_pair_chunk_size, int(self.config["batch_size"]))
        return {
            "stage": "stage1-supervised",
            "encoder_blocks": len(backbone.encoder.block),
            "decoder_blocks": len(backbone.decoder.block),
            "d_model": int(backbone.config.d_model),
            "d_ff": int(backbone.config.d_ff),
            "num_heads": int(backbone.config.num_heads),
            "d_kv": int(backbone.config.d_kv),
            "logit_head_scale": 1.0,
            "sid_initialization": model.sid_initialization,
            "decoding_protocol": self.decoding_protocol,
            "recommender_pretraining": dict(self.config.get("recommender_pretraining", {})),
            "backbone_trainable_params": backbone_trainable,
            "backbone_total_params": backbone_total,
            "model_trainable_params": model_trainable,
            "model_total_params": model_total,
            "optimizer_coverage": "exact",
            "max_history": max_history,
            "max_encoder_tokens": max_history * self.code_length,
            "train_examples": len(train_data.dataset),
            "max_examples_per_epoch": min(
                len(train_data.dataset),
                int(train_batches) * int(self.config["batch_size"]),
            ),
            "train_batches": int(train_batches),
            "micro_batch_size": int(self.config["batch_size"]),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer_updates_per_epoch": math.ceil(
                train_batches / gradient_accumulation_steps
            ),
            "training_views_per_example": stage1_views,
            "catalog_consistency": self.stage1_consistency_enabled,
            "catalog_consistency_weight": self.stage1_consistency_weight,
            "catalog_consistency_temperature": self.stage1_consistency_temperature,
            "primary_ce_mode": self.primary_ce_mode,
            "paired_view_chunk_size": (
                paired_chunk if self.stage1_consistency_enabled else None
            ),
            "max_examples_per_t5_forward": (
                min(int(self.config["batch_size"]), 2 * paired_chunk)
                if self.stage1_consistency_enabled
                else int(self.config["batch_size"])
            ),
        }

    def _batch_limit(self, stage: str) -> Optional[int]:
        if stage == "policy":
            value = self.config.get("modules", {}).get("module_2", {}).get("max_batches_per_epoch")
        else:
            value = self.config.get("max_batches_per_epoch")
        return None if value in (None, 0, "null") else int(value)

    def _set_module_1_progress(self, epoch: int) -> float:
        """Warm the expert first, then expose a bounded fusion gradually."""

        if not self.module_1_enabled:
            return 0.0
        module = self.config.get("modules", {}).get("module_1", {})
        warmup = max(0, int(module.get("fusion_warmup_epochs", 5)))
        ramp = max(1, int(module.get("fusion_ramp_epochs", 10)))
        progress = 0.0 if epoch < warmup else min((epoch - warmup + 1) / ramp, 1.0)
        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.topology_router.set_fusion_progress(progress)
        return float(progress)

    def _map_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_history = batch["input_ids"].to(self.device)
        target_items = batch["targets"].to(self.device).view(-1)
        batch_size = item_history.size(0)
        history = self.item_codes[item_history].contiguous().view(batch_size, -1)
        history_mask = history.ne(-1)
        labels = self.item_codes[target_items].contiguous()
        return history, history_mask, labels, target_items

    def _catalog_ce(
        self, logits: torch.Tensor, labels: torch.Tensor, target_items: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Primary CE; full-vocabulary mode matches the released recommender."""
        if self.primary_ce_mode == "full_vocabulary":
            return _full_vocabulary_ce(logits, labels)
        return _catalog_sibling_ce(
            logits, labels, target_items, self.catalog_child_mask
        )

    def _capture_module_1(self, outputs, labels: torch.Tensor) -> None:
        if outputs.route_gates is None:
            return
        if self.prototype_index is None:
            raise RuntimeError("Module-1 telemetry requires a prototype index")
        captured_samples = self.trace.payload["module_1_samples"]
        remaining = max(self.trace.sample_limit - len(captured_samples), 0)
        if remaining == 0:
            return
        maximum = int(self.config.get("telemetry", {}).get("samples_per_batch", 8))
        sample_count = min(labels.size(0), maximum)
        if sample_count <= 0:
            return
        projection_dim = min(
            int(self.config.get("telemetry", {}).get("embedding_dimensions", 32)),
            outputs.intent_vectors.size(-1),
        )
        vocabulary = outputs.route_scores.size(-1)
        diagnostics = torch.stack(
            [outputs.route_gates, outputs.route_entropy, outputs.route_margins,
             torch.sigmoid(outputs.route_gate_logits)],
            dim=-1,
        )
        # Transfer one compact telemetry tensor per sampled batch. The previous
        # record-by-record .item()/.cpu() path forced over a hundred GPU
        # synchronizations whenever telemetry was captured.
        packed_rows = torch.cat(
            [
                labels[:sample_count, : self.semantic_depth].to(
                    outputs.route_scores.dtype
                ),
                outputs.route_scores[:sample_count, : self.semantic_depth].reshape(
                    sample_count, -1
                ),
                outputs.base_logits[:sample_count, : self.semantic_depth].reshape(sample_count, -1),
                outputs.logits[:sample_count, : self.semantic_depth].reshape(sample_count, -1),
                outputs.intent_vectors[
                    :sample_count, : self.semantic_depth, :projection_dim
                ].reshape(sample_count, -1),
                diagnostics[:sample_count, : self.semantic_depth].reshape(
                    sample_count, -1
                ),
            ],
            dim=1,
        ).detach().cpu()
        cursor = 0
        label_values = packed_rows[:, cursor : cursor + self.semantic_depth].long()
        cursor += self.semantic_depth
        score_width = self.semantic_depth * vocabulary
        score_values = packed_rows[:, cursor : cursor + score_width].view(
            sample_count, self.semantic_depth, vocabulary
        )
        cursor += score_width
        base_values = packed_rows[:, cursor : cursor + score_width].view(
            sample_count, self.semantic_depth, vocabulary)
        cursor += score_width
        intent_width = self.semantic_depth * projection_dim
        fused_values = packed_rows[:, cursor : cursor + score_width].view(
            sample_count, self.semantic_depth, vocabulary)
        cursor += score_width
        intent_values = packed_rows[:, cursor : cursor + intent_width].view(
            sample_count, self.semantic_depth, projection_dim
        )
        cursor += intent_width
        diagnostic_values = packed_rows[:, cursor:].view(
            sample_count, self.semantic_depth, 4
        )
        label_rows = [[int(value) for value in row] for row in label_values.tolist()]
        records = []
        for row, label_row in enumerate(label_rows):
            for depth in range(self.semantic_depth):
                if len(records) >= remaining:
                    break
                prefix = tuple(label_row[:depth])
                entry = self.prototype_index.get_host(prefix)
                # A sole valid child is not a routing decision.  Excluding it
                # also keeps mechanism plots focused on cases Module 1 can alter.
                if entry is None or len(entry[0]) <= 1:
                    continue
                children, prototypes, _sizes = entry
                snapshot = getattr(self, "_diagnostic_memory", {}).get(prefix)
                if snapshot is not None:
                    prototypes = snapshot
                target = label_row[depth]
                target_position = children.index(target)
                scores = score_values[row, depth].index_select(
                    0, torch.as_tensor(children, dtype=torch.long)
                )
                masked = scores.clone()
                masked[target_position] = -torch.inf
                alternative_position = int(masked.argmax().item())
                gate, entropy, margin, gate_probability = diagnostic_values[row, depth].tolist()
                base_full_logp = base_values[row, depth].log_softmax(-1)
                fused_full_logp = fused_values[row, depth].log_softmax(-1)
                advantage = float((scores.log_softmax(-1)[target_position]
                                   - base_full_logp[target]).item())
                gate_temperature = float(self.config["modules"]["module_1"].get("gate_target_temperature", 1.0))
                soft_target = float(torch.sigmoid(torch.tensor(advantage / gate_temperature)).item())
                records.append(
                    {
                        "depth": depth + 1,
                        "source": getattr(self, "_telemetry_source", "training_attempt"),
                        "prefix": list(prefix),
                        "target_token": int(target),
                        "alternative_token": int(children[alternative_position]),
                        "support_reliability": float(self.prototype_index._entries[prefix][3][target_position]),
                        "gate": float(gate),
                        "gate_probability": float(gate_probability),
                        "expert_log_likelihood_advantage": advantage,
                        "gate_soft_target": soft_target,
                        "legal_mass_before": float(base_full_logp[list(children)].exp().sum()),
                        "legal_mass_after": float(fused_full_logp[list(children)].exp().sum()),
                        "base_entropy": float(entropy),
                        "prototype_margin": float(margin),
                        "intent": intent_values[row, depth],
                        "target_prototype": prototypes[target_position][:projection_dim],
                        "alternative_prototype": prototypes[alternative_position][
                            :projection_dim
                        ],
                    }
                )
            if len(records) >= remaining:
                break
        self.trace.extend_module_1(records)

    def _capture_module_2(self, diagnostics, epoch: int, source: str) -> None:
        remaining = self.trace.sample_limit - len(self.trace.payload["module_2_edges"])
        if remaining <= 0:
            return
        fields = [key for key in diagnostics if key != "valid"]
        valid = diagnostics["valid"]
        packed = torch.stack([diagnostics[key].float() for key in fields], dim=-1)
        rows = packed[valid][:min(remaining, 32)].detach().cpu().tolist()
        self.trace.extend_module_2([
            {**dict(zip(fields, values)), "epoch": int(epoch), "source": source}
            for values in rows
        ])

    def _supervised_objective(self, outputs, labels, target_items, module_active):
        """Positive-data cross-entropy anchor shared by both training stages."""
        ce, depths = self._catalog_ce(outputs.logits, labels, target_items)
        zero = ce * 0.0
        base_ce, route_loss, gate_loss, memory_reg = ce, zero, zero, zero
        branches = None
        loss = ce
        if module_active:
            module = self.config["modules"]["module_1"]
            base_ce, _ = self._catalog_ce(outputs.base_logits, labels, target_items)
            route_loss, branches = _branching_route_ce(
                outputs.route_scores, labels, target_items,
                self.catalog_child_mask, self.semantic_depth)
            child_mask = self.catalog_child_mask.index_select(0, target_items.reshape(-1).long())
            gate_loss, _ = competence_gate_loss(
                outputs.base_logits, outputs.route_scores, outputs.route_gate_logits,
                labels, child_mask, self.semantic_depth,
                float(module.get("gate_target_temperature", 1.0)))
            router = self.accelerator.unwrap_model(self.model).topology_router
            memory_reg = router.memory_regularization()
            anchor = float(module.get("base_anchor_fraction", 0.25))
            loss = ((1.0 - anchor) * ce + anchor * base_ce
                    + float(module.get("route_loss_weight", 0.05)) * route_loss
                    + float(module.get("gate_loss_weight", 0.02)) * gate_loss
                    + float(module.get("memory_l2_weight", 1e-4)) * memory_reg)
        metrics = {"recommendation_ce": ce, "base_ce": base_ce, "route_loss": route_loss,
                   "gate_competence_loss": gate_loss, "memory_regularization": memory_reg}
        return loss, ce, depths, branches, metrics

    def _train_supervised_epoch(self, epoch: int) -> Dict[str, float]:
        started_at = time.perf_counter()
        self.model.train()
        module_active = self.module_1_enabled and not self._common_pretraining
        fusion_progress = self._set_module_1_progress(epoch) if module_active else 0.0
        self.optimizer.zero_grad()
        totals: Dict[str, object] = {}
        count = 0
        limit = self._batch_limit("supervised")
        module = self.config.get("modules", {}).get("module_1", {})
        stage = "warmup-rec" if self._common_pretraining else "stage1-supervised"
        capture_every = max(1, int(self.config.get("telemetry", {}).get("capture_every_batches", 20)))
        iterator = tqdm(
            self.train_data, total=min(len(self.train_data), limit) if limit else len(self.train_data),
            desc=f"{stage}[{self.config['run_name']},M1={'on' if module_active else 'off'}] e{epoch}",
            disable=not self.accelerator.is_main_process,
            mininterval=float(self.config.get("progress_mininterval_seconds", 5.0)),
        )
        for batch_index, batch in enumerate(iterator):
            if limit is not None and batch_index >= limit:
                break
            with self.accelerator.accumulate(self.model):
                history, mask, labels, target_items = self._map_batch(batch)
                outputs = self.model(input_ids=history, attention_mask=mask, labels=labels)
                loss, ce, depths, branches, objective_metrics = self._supervised_objective(
                    outputs, labels, target_items, module_active)
                if batch_index % 50 == 0 and not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError("Non-finite supervised objective")
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            count += 1
            for key, value in {"loss": loss, **objective_metrics}.items():
                _accumulate_detached(totals, key, value)
            for depth, value in enumerate(depths):
                _accumulate_detached(totals, f"depth_ce_{depth + 1}", value)
            if module_active:
                for depth in range(self.semantic_depth):
                    active = branches[:, depth].float()
                    denominator = active.sum().clamp_min(1.0)
                    for key, values in (("gate", outputs.route_gates),
                                        ("entropy", outputs.route_entropy),
                                        ("margin", outputs.route_margins)):
                        _accumulate_detached(totals, f"{key}_{depth + 1}",
                                             (values[:, depth] * active).sum() / denominator)
                if batch_index % capture_every == 0:
                    self._capture_module_1(outputs, labels)
            if batch_index % max(1, int(self.config.get("progress_loss_every_batches", 10))) == 0:
                iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if count == 0:
            raise RuntimeError("Supervised stage received no batches")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = _metric_means(totals, count)
        result.update(epoch_seconds=time.perf_counter() - started_at,
                      module_1_fusion_progress=fusion_progress,
                      optimizer_steps=math.ceil(count / self.supervised_gradient_accumulation_steps))
        result["seconds_per_batch"] = result["epoch_seconds"] / count
        if module_active:
            router = self.accelerator.unwrap_model(self.model).topology_router
            result["expert_logit_scales"] = router.score_scale().detach().cpu().tolist()
        return result

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        started_at = time.perf_counter()
        totals = defaultdict(float)
        examples = 0
        for batch in tqdm(
            data_loader,
            desc=f"evaluate[{self.config['run_name']}]",
            disable=not self.accelerator.is_main_process,
            mininterval=float(self.config.get("progress_mininterval_seconds", 5.0)),
        ):
            history, history_mask, labels, target_items = self._map_batch(batch)
            decoded = beam_search(
                self.accelerator.unwrap_model(self.model),
                history,
                history_mask,
                self.trie,
                num_beams=int(self.config["num_beams"]),
                topk=self.max_topk,
                decoder_start_token_id=0,
                protocol=self.decoding_protocol,
            )
            gathered_items, gathered_targets, gathered_sids, gathered_labels = self.accelerator.gather_for_metrics(
                (decoded.item_ids, target_items, decoded.sids, labels)
            )
            batch_metrics = ranking_metric_sums(gathered_items, gathered_targets, self.metric_names,
                                                return_tensors=True)
            batch_metrics["invalid_sid_fraction_at_reported_topk"] = gathered_items.lt(0).float().mean(-1).sum()
            hierarchy = hierarchy_metric_sums(
                gathered_sids[:, 0], gathered_labels, self.semantic_depth, return_tensors=True
            )
            for key, value in {**batch_metrics, **hierarchy}.items():
                _accumulate_detached(totals, key, value)
            examples += int(gathered_targets.numel())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_evaluation_seconds = time.perf_counter() - started_at
        print(f"[evaluation] protocol={self.decoding_protocol} | examples={examples} | "
              f"seconds={self.last_evaluation_seconds:.2f}", flush=True)
        return {key: round(value, 6) for key, value in _metric_means(totals, max(examples, 1)).items()}

    def _save(self, stage: str, epoch: int, metrics: Mapping[str, object]) -> str:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{stage}_epoch_{epoch:04d}.pt"
        if self.accelerator.is_main_process:
            temporary = path.with_suffix(".pt.tmp")
            torch.save(self.accelerator.unwrap_model(self.model).state_dict(), temporary)
            os.replace(temporary, path)
            code_path = path.with_suffix(".sid.json")
            with open(code_path, "w", encoding="utf-8") as output:
                json.dump(self.item_codes.cpu().tolist(), output)
            meta = {
                "stage": stage,
                "epoch": int(epoch),
                "metrics": dict(metrics),
                "catalog_version": self.catalog_version,
                "module_1": self.module_1_enabled,
                "module_2": self.module_2_enabled,
            }
            with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as output:
                json.dump(meta, output, indent=2)
        self.accelerator.wait_for_everyone()
        return str(path)

    def _load(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(state)

    def _reset_optimizer(self, learning_rate: float, epochs: int, warmup_steps: int,
                         horizon_epochs: Optional[int] = None) -> None:
        raw_model = self.accelerator.unwrap_model(self.model)
        self.optimizer = self._optimizer(raw_model, learning_rate)
        self._assert_optimizer_coverage(raw_model, self.optimizer)
        horizon = int(horizon_epochs or epochs)
        steps = max(1, int(self.training_contract["optimizer_updates_per_epoch"]) * horizon)
        self.scheduler = get_scheduler(
            name=str(self.config.get("lr_scheduler_type", "cosine")), optimizer=self.optimizer,
            num_warmup_steps=int(warmup_steps), num_training_steps=steps,
        )
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

    def _reset_phase_stream(self, phase: int) -> None:
        seed = int(self.config["seed"])
        init_seed(seed + phase, bool(self.config.get("reproducibility", True)))
        for loader, order_generator, offset in (
            (self.train_data, self.train_order_generator, 101),
            (self.policy_data, self.policy_order_generator, 202),
        ):
            if order_generator is not None:
                order_generator.manual_seed(seed + offset + phase)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generator.manual_seed(seed + 400 + offset + phase)
            if callable(getattr(loader, "set_epoch", None)):
                loader.set_epoch(0)

    def pretrain_recommender(self) -> Dict[str, object]:
        """Warm up the fixed-SID backbone before EAPM and RCBR training."""
        settings = dict(self.config.get("recommender_pretraining", {}))
        epochs = int(settings.get("epochs", 50))
        print("[warmup-rec] Training the fixed-SID backbone before the EAPM/RCBR stages.",
              flush=True)
        if epochs <= 0:
            return {"enabled": False, "epochs": 0}
        keys = ("seed", "batch_size", "gradient_accumulation_steps", "max_his_len",
                "encoder_layers", "decoder_layers", "d_model", "d_ff", "num_heads", "d_kv",
                "dropout_rate", "weight_decay", "sid_embedding_initialization", "primary_ce_mode",
                "decoding_protocol", "train_data_ratio", "valid_data_ratio", "max_batches_per_epoch",
                "num_beams", "eval_batch_size", "valid_metric")
        contract = {key: self.config.get(key) for key in keys}
        contract.update(revision="hgig-public-v1", catalog=self.catalog_version,
                        pretraining=settings, train_examples=self.training_contract["train_examples"])
        fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        cache_path = Path(str(self.config.get("warmstart_path") or
                              self.output_dir / "common_warmstart.pt"))
        raw = self.accelerator.unwrap_model(self.model)

        def load_core(state):
            incompatible = raw.load_state_dict(state, strict=False)
            unexpected = incompatible.unexpected_keys
            missing = [key for key in incompatible.missing_keys if not key.startswith("topology_router.")]
            if unexpected or missing:
                raise ValueError(f"Incompatible common warmstart: missing={missing}, unexpected={unexpected}")

        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            if payload.get("fingerprint") != fingerprint:
                raise ValueError("Warmstart configuration changed; use a different warmstart_path")
            load_core(payload["state_dict"])
            result = dict(payload["summary"])
            result.update(reused=True, cache_path=str(cache_path))
            print(f"[warmup-rec] reused shared checkpoint: {cache_path}", flush=True)
        else:
            self._common_pretraining = True
            raw.topology_enabled = False
            self._reset_phase_stream(0)
            self._reset_optimizer(float(settings.get("learning_rate", 0.005)), epochs,
                                  int(settings.get("warmup_steps", 8000)),
                                  int(settings.get("scheduler_horizon_epochs", 100)))
            best_score = float("-inf")
            best_state = best_metrics = None
            best_epoch = -1
            patience = 0
            history = []
            try:
                for epoch in range(epochs):
                    training = self._train_supervised_epoch(epoch)
                    validation = self.evaluate(self.valid_data)
                    record = {"stage": "common_pretraining", "epoch": epoch,
                              "train": training, "validation": validation,
                              "evaluation_seconds": self.last_evaluation_seconds}
                    history.append(record)
                    self.trace.epoch(record)
                    self.trace.save(self.output_dir / "mechanism_trace.json")
                    score = float(validation[self.valid_metric])
                    if score > best_score:
                        best_score, best_metrics, best_epoch = score, validation, epoch
                        best_state = {key: value.detach().cpu().clone()
                                      for key, value in raw.state_dict().items()
                                      if not key.startswith("topology_router.")}
                        patience = 0
                    else:
                        patience += 1
                    print(f"[warmup-rec] epoch={epoch} | validation={validation} | "
                          f"best={best_score:.6f}", flush=True)
                    if patience >= int(settings.get("early_stop", 15)):
                        break
            finally:
                self._common_pretraining = False
                raw.topology_enabled = True
            if best_state is None:
                raise RuntimeError("Recommender pretraining produced no checkpoint")
            load_core(best_state)
            result = {"enabled": True, "reused": False, "epochs": len(history),
                      "best_epoch": best_epoch, "best_validation": best_metrics,
                      "history": history, "cache_path": str(cache_path),
                      "objective": "fixed_SID_full_vocabulary_CE", "fingerprint": fingerprint}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save({"fingerprint": fingerprint, "state_dict": best_state,
                        "summary": result}, temporary)
            os.replace(temporary, cache_path)
        self.trace.payload["metadata"]["common_pretraining"] = {
            key: value for key, value in result.items() if key != "history"
        }
        # Include the shared pretrained model in subsequent validation selection.
        # At fusion_progress=0 the attached module has exactly zero correction.
        self.best_validation = result["best_validation"]
        self.best_score = float(self.best_validation[self.valid_metric])
        self.best_checkpoint = self._save("stage1_entry", -1, self.best_validation)
        return result

    def train_supervised(self) -> Dict[str, object]:
        stage_started_at = time.perf_counter()
        patience = 0
        history = []
        epochs = int(self.config["epochs"])
        self._reset_phase_stream(10000)
        self._reset_optimizer(float(self.config["learning_rate"]), epochs,
                              int(self.config.get("warmup_steps", 0)))
        eval_step = max(1, min(int(self.config.get("eval_step", 1)), epochs))
        if self.accelerator.is_main_process:
            print(
                "[stage1-supervised:start] "
                f"run={self.config['run_name']} | "
                f"M1={'on' if self.module_1_enabled else 'off'} | "
                "views=1 | primary_objective=full_vocabulary_CE | "
                f"epochs={epochs} | eval_step={eval_step} | "
                f"optimizer_updates_per_epoch="
                f"{self.training_contract['optimizer_updates_per_epoch']}",
                flush=True,
            )
        for epoch in range(epochs):
            train_metrics = self._train_supervised_epoch(epoch)
            record = {"stage": "supervised", "epoch": epoch, **train_metrics}
            self.trace.epoch(record)
            self.trace.save(self.output_dir / "mechanism_trace.json")
            should_evaluate = (epoch + 1) % eval_step == 0 or epoch + 1 == epochs
            if not should_evaluate:
                history.append(
                    {"epoch": epoch, "train": train_metrics, "validation": None}
                )
                continue

            validation = self.evaluate(self.valid_data)
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation,
                            "evaluation_seconds": self.last_evaluation_seconds})
            history[-1]["train_evaluation_seconds"] = train_metrics["epoch_seconds"] + self.last_evaluation_seconds
            print(f"[stage1 validation] epoch={epoch} | metrics={validation} | "
                  f"train_seconds={train_metrics['epoch_seconds']:.2f} | "
                  f"eval_seconds={self.last_evaluation_seconds:.2f} | "
                  f"combined_seconds={history[-1]['train_evaluation_seconds']:.2f}", flush=True)
            score = float(validation[self.valid_metric])
            if score > self.best_score:
                self.best_score = score
                self.best_validation = validation
                self.best_checkpoint = self._save("supervised", epoch, validation)
                patience = 0
            else:
                patience += 1
            if patience >= int(self.config.get("early_stop", 10)):
                break
        if self.best_checkpoint is None:
            raise RuntimeError("No supervised checkpoint was selected")
        self._load(self.best_checkpoint)
        if self.accelerator.is_main_process:
            print(
                "[stage1-supervised:complete] "
                f"run={self.config['run_name']} | epochs_ran={len(history)} | "
                f"best_{self.valid_metric}={self.best_score:.8f} | "
                f"elapsed_seconds={time.perf_counter() - stage_started_at:.2f} | "
                f"checkpoint={Path(self.best_checkpoint).name}",
                flush=True,
            )
        return {
            "best_score": self.best_score,
            "best_validation": self.best_validation,
            "checkpoint": self.best_checkpoint,
            "history": history,
        }

    def _repair_batch(self, history, mask, labels, target_items, reference_policy):
        module = self.config.get("modules", {}).get("module_2", {})
        raw_model = self.accelerator.unwrap_model(self.model)
        count = int(module.get("hard_negatives", 3))
        decoded = beam_search(
            raw_model, history, mask, self.trie,
            num_beams=int(self.config["num_beams"]), topk=int(self.config["num_beams"]),
            protocol=self.decoding_protocol,
        )
        negatives, usable, mining_metrics, mining_diagnostics = select_repair_negatives(
            decoded, labels, history, mask, count, self.code_number)
        candidates = torch.cat([labels.unsqueeze(1), negatives], dim=1)
        repeats = count + 1
        pair_labels = candidates.flatten(0, 1)
        pair_mask = mask.repeat_interleave(repeats, dim=0)
        # Each history is encoded once for all positive/negative candidates.
        encoded = raw_model.encode(history, mask).last_hidden_state
        outputs = self.model(
            attention_mask=pair_mask, labels=pair_labels,
            encoder_outputs=(encoded.repeat_interleave(repeats, dim=0),),
        )
        with torch.no_grad():
            ref_encoded = reference_policy.encode(history, mask).last_hidden_state
            reference = reference_policy(
                attention_mask=pair_mask, labels=pair_labels,
                encoder_outputs=(ref_encoded.repeat_interleave(repeats, dim=0),),
            )
        auxiliary, metrics, diagnostics = repair_objective(
            outputs, reference, candidates, usable, module
        )
        fields = ("logits", "base_logits", "route_scores", "route_gate_logits")
        positive_outputs = SimpleNamespace(**{
            key: (getattr(outputs, key).reshape(history.size(0), repeats, *getattr(outputs, key).shape[1:])[:, 0]
                  if getattr(outputs, key, None) is not None else None)
            for key in fields})
        anchor, nll, _depths, _branches, anchor_metrics = self._supervised_objective(
            positive_outputs, labels, target_items, self.module_1_enabled)
        loss = float(module.get("nll_weight", 1.0)) * anchor + auxiliary
        diagnostics.update(mining_diagnostics)
        return loss, {"anchor_nll": nll, **anchor_metrics, **metrics, **mining_metrics}, diagnostics

    def _policy_epoch(self, epoch: int, reference_policy) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, object] = {}
        updates = 0
        started_at = time.perf_counter()
        limit = self._batch_limit("policy")
        iterator = tqdm(
            self.policy_data, total=min(len(self.policy_data), limit) if limit else len(self.policy_data),
            desc=f"stage2-branch-repair[{self.config['run_name']}] e{epoch}",
            disable=not self.accelerator.is_main_process,
            mininterval=float(self.config.get("progress_mininterval_seconds", 5.0)),
        )
        for batch_index, batch in enumerate(iterator):
            if limit is not None and batch_index >= limit:
                break
            history, mask, labels, targets = self._map_batch(batch)
            self.optimizer.zero_grad()
            loss, metrics, diagnostics = self._repair_batch(history, mask, labels, targets, reference_policy)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("Non-finite branch-repair objective")
            self.accelerator.backward(loss * self.supervised_gradient_accumulation_steps)
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            for key, value in {"loss": loss, **metrics}.items():
                _accumulate_detached(totals, key, value)
            updates += 1
            if batch_index % 20 == 0:
                self._capture_module_2(diagnostics, epoch, "training_attempt")
                iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if updates == 0:
            raise RuntimeError("Module 2 received no batches")
        result = _metric_means(totals, updates)
        result.update(policy_optimizer_steps=updates, epoch_seconds=time.perf_counter() - started_at)
        self.trace.epoch({"stage": "module_2", "epoch": epoch, **result})
        self.trace.save(self.output_dir / "mechanism_trace.json")
        return result

    def train_rcbr(self) -> Dict[str, object]:
        module = self.config.get("modules", {}).get("module_2", {})
        self._reset_phase_stream(20000)
        entry_checkpoint = self.best_checkpoint
        entry_score = self.best_score
        selection_policy = "best_rcbr_epoch"
        reference = copy.deepcopy(self.accelerator.unwrap_model(self.model)).to(self.device).eval()
        reference.requires_grad_(False)

        policy_limit = self._batch_limit("policy")
        policy_batches = (
            min(len(self.policy_data), policy_limit)
            if policy_limit is not None
            else len(self.policy_data)
        )
        refinement_epochs = int(module.get("epochs", 1))
        updates_per_batch = int(module.get("updates_per_batch", 1))
        mode = "router_aware_branch_repair"
        objective = "supervised_anchor + sequence_preference + cumulative_prefix_ranking + router_guard"
        if self.accelerator.is_main_process:
            print(
                "[stage2 start] "
                f"run={self.config['run_name']} | init=stage1_best | "
                f"mode={mode} | "
                "RCBR=on | reference=frozen_stage1_best | "
                f"policy_ratio={float(self.config.get('policy_data_ratio', 0.05)):g} | "
                f"policy_batches={policy_batches} | epochs={refinement_epochs} | "
                f"updates_per_batch={updates_per_batch} | "
                f"selection={selection_policy} | "
                "trainable_scope=whole_model | "
                f"objective={objective}",
                flush=True,
            )

        raw_model = self.accelerator.unwrap_model(self.model)
        self.optimizer = self._optimizer(raw_model, float(module.get("learning_rate", 2e-5)))
        self._assert_optimizer_coverage(raw_model, self.optimizer)
        steps = max(1, policy_batches * refinement_epochs)
        steps *= max(1, updates_per_batch)
        self.scheduler = get_scheduler(
            name="cosine", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=steps
        )
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

        selected_checkpoint = None
        selected_validation = None
        # Select the best RCBR epoch by validation. The Stage-1 entry is
        # recorded for comparison but is not a fallback candidate.
        selected_score = float("-inf")
        selected_epoch = None
        history = []
        for epoch in range(refinement_epochs):
            train_metrics = self._policy_epoch(epoch, reference)
            validation = self.evaluate(self.valid_data)
            history.append(
                {
                    "epoch": epoch,
                    "mode": mode,
                    "train": train_metrics,
                    "validation": validation,
                    "evaluation_seconds": self.last_evaluation_seconds,
                    "train_evaluation_seconds": train_metrics["epoch_seconds"] + self.last_evaluation_seconds,
                }
            )
            score = float(validation[self.valid_metric])
            if not math.isfinite(score):
                raise ValueError(
                    f"Non-finite Stage-2 {self.valid_metric} at epoch {epoch}: {score}"
                )
            is_new_best = score > selected_score
            if is_new_best:
                selected_score = score
                selected_validation = validation
                selected_checkpoint = self._save("module_2", epoch, validation)
                selected_epoch = epoch
            if self.accelerator.is_main_process:
                print(
                    "[stage2 validation] "
                    f"mode={mode} | epoch={epoch} | "
                    f"{self.valid_metric}={score:.6f} | "
                    f"delta_vs_stage1={score - entry_score:+.6f} | "
                    f"new_best={'yes' if is_new_best else 'no'} | "
                    f"train_seconds={train_metrics['epoch_seconds']:.2f} | "
                    f"eval_seconds={self.last_evaluation_seconds:.2f} | "
                    f"combined_seconds={history[-1]['train_evaluation_seconds']:.2f}",
                    flush=True,
                )

        if selected_checkpoint is None or selected_validation is None:
            raise RuntimeError("Stage 2 produced no selectable checkpoint")
        self._load(selected_checkpoint)
        self.best_checkpoint = selected_checkpoint
        self.best_score = selected_score
        self.best_validation = selected_validation
        improved_vs_stage1 = selected_score > entry_score
        if self.accelerator.is_main_process:
            print(
                "[stage2 selected] "
                f"mode={mode} | policy={selection_policy} | "
                f"epoch={selected_epoch} | "
                f"{self.valid_metric}={selected_score:.6f} | "
                f"delta_vs_stage1={selected_score - entry_score:+.6f} | "
                f"improved_vs_stage1={'yes' if improved_vs_stage1 else 'no'} | "
                f"checkpoint={selected_checkpoint}",
                flush=True,
            )
        self.trace.payload["metadata"]["stage2_selection"] = {
            "mode": mode,
            "module_2_enabled": True,
            "selection_policy": selection_policy,
            "accepted": True,
            "improved_vs_stage1": improved_vs_stage1,
            "entry_checkpoint": entry_checkpoint,
            "entry_score": float(entry_score),
            "best_score": float(self.best_score),
            "valid_metric": self.valid_metric,
            "selected_epoch": selected_epoch,
            "checkpoint": self.best_checkpoint,
        }
        self.capture_final_diagnostics(reference)
        self.trace.save(self.output_dir / "mechanism_trace.json")
        return {
            "enabled": True,
            "stage2_executed": True,
            "mode": mode,
            "policy_batches_per_epoch": int(policy_batches),
            "epochs": int(refinement_epochs),
            "updates_per_batch": int(updates_per_batch),
            "planned_policy_optimizer_steps": int(steps),
            "selection_policy": selection_policy,
            "accepted": True,
            "improved_vs_stage1": improved_vs_stage1,
            "entry_checkpoint": entry_checkpoint,
            "entry_score": entry_score,
            "best_score": self.best_score,
            "checkpoint": self.best_checkpoint,
            "selected_epoch": selected_epoch,
            "history": history,
        }

    @torch.no_grad()
    def capture_final_diagnostics(self, reference=None) -> None:
        """Capture the selected model on a fixed validation subset for replotting.

        Training attempts remain separately available, so an unselected Stage-2
        epoch or an early untrained router cannot be mistaken for the final mechanism.
        Test labels never enter model selection or these diagnostic objectives.
        """
        self.model.eval()
        self.trace.payload["module_1_training_samples"] = self.trace.payload["module_1_samples"]
        self.trace.payload["module_2_training_samples"] = self.trace.payload["module_2_edges"]
        self.trace.payload["module_1_samples"] = []
        self.trace.payload["module_2_edges"] = []
        self._telemetry_source = "selected_checkpoint_validation"
        self._diagnostic_memory = {}
        raw = self.accelerator.unwrap_model(self.model)
        if self.module_1_enabled:
            router = raw.topology_router
            self.trace.payload["metadata"]["selected_module_1"] = {
                "fusion_progress": float(router.fusion_progress.cpu()),
                "fusion_cap": router.max_gate,
                "expert_logit_scales": router.score_scale().cpu().tolist(),
                "note": "Zero fusion progress means validation selected a no-fusion checkpoint.",
            }
            packed = self.prototype_index.prepare_packed(self.device, self.code_number, self.semantic_depth)
            edges = torch.arange(packed.child_index.numel(), device=self.device)
            vectors = router._edge_vectors(packed, edges).cpu()
            keys = packed.node_keys.cpu().tolist()
            pointers = packed.node_ptr.cpu().tolist()
            lookup = {key: index for index, key in enumerate(keys)}
            for prefix in self.prototype_index._entries:
                if len(prefix) >= self.semantic_depth:
                    continue
                key = self.prototype_index._prefix_key(prefix, self.code_number, packed.depth_offsets)
                index = lookup.get(key)
                if index is not None:
                    self._diagnostic_memory[prefix] = vectors[pointers[index]:pointers[index + 1]]
        limit = int(self.config.get("telemetry", {}).get("final_diagnostic_batches", 16))
        count = 0
        if self.module_1_enabled or reference is not None:
            for index, batch in enumerate(self.valid_data):
                if index >= limit:
                    break
                history, mask, labels, targets = self._map_batch(batch)
                if self.module_1_enabled:
                    outputs = self.model(input_ids=history, attention_mask=mask, labels=labels)
                    self._capture_module_1(outputs, labels)
                if reference is not None:
                    _loss, _metrics, diagnostics = self._repair_batch(history, mask, labels, targets, reference)
                    self._capture_module_2(diagnostics, -1, self._telemetry_source)
                count += int(labels.size(0))
        self.trace.payload["metadata"]["final_diagnostics"] = {
            "checkpoint": self.best_checkpoint, "split": "validation", "users": count,
            "sampling": "first_fixed_validation_batches", "max_batches": limit,
            "training_attempts_stored_separately": True,
        }
        self._diagnostic_memory.clear()

    def test(self) -> Dict[str, float]:
        if self.best_checkpoint:
            self._load(self.best_checkpoint)
        result = self.evaluate(self.test_data)
        self.trace.payload["metadata"]["test_seconds"] = self.last_evaluation_seconds
        self.trace.save(self.output_dir / "mechanism_trace.json")
        return result
