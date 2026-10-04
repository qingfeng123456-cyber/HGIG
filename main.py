"""One-command runner for the generic fixed-SID recommendation model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Mapping

import accelerate
import numpy as np
import torch
import transformers
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from transformers import T5Config, T5ForConditionalGeneration

from data import Collator, SequentialSplitDataset, load_split_data
from model import GenerativeSIDModel
from sid_catalog import load_sids, save_sids
from trainer import Trainer
from utils import convert_config_dict, deep_update, init_seed, parse_command_line_args


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="./config/scientific.yaml")
    args, unknown = parser.parse_known_args()
    return args, unknown


def load_configuration(path: str | Path) -> Dict[str, object]:
    """Load YAML with an optional relative ``extends`` parent."""
    source_path = Path(path).resolve()
    with open(source_path, "r", encoding="utf-8") as source:
        current = yaml.safe_load(source) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    base = load_configuration(source_path.parent / str(parent))
    return deep_update(base, current)


def validate_config(config: Mapping[str, object]) -> None:
    required = {
        "dataset",
        "data_path",
        "semantic_emb_path",
        "code_num",
        "code_length",
        "semantic_depth",
        "epochs",
        "learning_rate",
        "metrics",
        "valid_metric",
        "num_beams",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    for key in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "code_num",
        "code_length",
        "semantic_depth",
        "num_beams",
    ):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["semantic_depth"]) > int(config["code_length"]):
        raise ValueError("semantic_depth cannot exceed code_length")
    if len(config.get("num_emb_list", [])) != int(config["semantic_depth"]):
        raise ValueError("num_emb_list must contain one codebook size per semantic level")
    if any(int(value) > int(config["code_num"]) for value in config["num_emb_list"]):
        raise ValueError("Every tokenizer codebook must fit the generator vocabulary")
    metric_names = [metric.strip().lower() for metric in str(config["metrics"]).split(",")]
    cutoffs = [int(metric.split("@", 1)[1]) for metric in metric_names]
    if int(config["num_beams"]) < max(cutoffs):
        raise ValueError("num_beams must cover the largest evaluation cutoff")
    if str(config["valid_metric"]).strip().lower() not in metric_names:
        raise ValueError("valid_metric must be included in metrics")
    stage1_consistency = config.get("stage1_consistency", {})
    if not isinstance(stage1_consistency, Mapping):
        raise ValueError("stage1_consistency must be a mapping")
    if float(stage1_consistency.get("weight", 0.0)) < 0.0:
        raise ValueError("stage1_consistency.weight must be non-negative")
    if float(stage1_consistency.get("temperature", 1.0)) <= 0.0:
        raise ValueError("stage1_consistency.temperature must be positive")
    if stage1_consistency.get("enabled", False) and float(
        config.get("dropout_rate", 0.0)
    ) <= 0.0:
        raise ValueError(
            "stage1_consistency requires dropout_rate > 0 to create stochastic views"
        )
    if stage1_consistency.get("enabled", False) and int(config["batch_size"]) < 2:
        raise ValueError(
            "stage1_consistency requires batch_size >= 2 for memory-bounded paired views"
        )
    modules = config.get("modules", {})
    module_2 = modules.get("module_2", {})
    module_1 = modules.get("module_1", {})
    if not module_1 or not module_2:
        raise ValueError("HGIG requires both EAPM and RCBR configuration blocks")
    if "enabled" in module_1 or "enabled" in module_2:
        raise ValueError("This release trains the complete HGIG model; module toggles are unsupported")
    if not config.get("sid_path"):
        raise ValueError("sid_path must point to a fixed catalog SID JSON file")
    if "temperature" in module_1:
        raise ValueError("Use module_1.logit_scale_init/logit_scale_max, not temperature")
    if any(key in module_2 for key in ("branch_weight", "absolute_weight", "absolute_margin", "difficulty_temperature")):
        raise ValueError("Use the current RCBR prefix_weight/prefix_margin configuration")
    for key in ("epochs", "batch_size", "updates_per_batch", "hard_negatives"):
        if int(module_2.get(key, 0)) <= 0:
            raise ValueError(f"modules.module_2.{key} must be positive")
    policy_ratio = float(config.get("policy_data_ratio", 0.0))
    if not 0.0 < policy_ratio <= 1.0:
        raise ValueError("policy_data_ratio must be in (0, 1]")
    for value, label in (
        (config.get("max_batches_per_epoch"), "max_batches_per_epoch"),
        (module_2.get("max_batches_per_epoch"), "modules.module_2.max_batches_per_epoch"),
    ):
        if value not in (None, "null") and int(value) < 0:
            raise ValueError(f"{label} must be null, zero, or positive")
    supervised_limit = config.get("max_batches_per_epoch")
    accumulation = int(config["gradient_accumulation_steps"])
    if (
        supervised_limit not in (None, 0, "null")
        and accumulation > 1
        and int(supervised_limit) % accumulation != 0
    ):
        raise ValueError(
            "A limited Stage-1 run must use max_batches_per_epoch divisible by "
            "gradient_accumulation_steps so the final accumulated update is not truncated"
        )
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "Module 1 packed routing requires a PyTorch build with Tensor.scatter_reduce_ "
            "support; install the pinned requirements or use a compatible PyTorch 2.x image."
        )
    if int(module_2.get("updates_per_batch", 1)) != 1:
        raise ValueError("RCBR currently requires updates_per_batch=1")
    if int(module_2.get("hard_negatives", 3)) + 1 > int(config["num_beams"]):
        raise ValueError("num_beams must cover hard_negatives+1")
    if str(config.get("primary_ce_mode", "full_vocabulary")) != "full_vocabulary":
        raise ValueError("HGIG requires full_vocabulary cross-entropy")
    if str(config.get("decoding_protocol", "full_vocabulary")) != "full_vocabulary":
        raise ValueError("HGIG requires full_vocabulary decoding")
    if float(module_1.get("min_gate", 0.0)) != 0.0:
        raise ValueError("Module 1 must start from min_gate=0")
    if not 0.0 <= float(module_1.get("max_gate", 0.30)) * float(module_1.get("strength", 1.0)) < 1.0:
        raise ValueError("module_1.max_gate * strength must be in [0, 1)")
    if not 0.0 <= float(module_1.get("base_anchor_fraction", 0.25)) <= 1.0:
        raise ValueError("module_1.base_anchor_fraction must be in [0, 1]")
    if not 1.0 <= float(module_1.get("logit_scale_init", 8.0)) <= float(module_1.get("logit_scale_max", 32.0)):
        raise ValueError("Require 1 <= logit_scale_init <= logit_scale_max")
    for name in ("gate_target_temperature",):
        if float(module_1.get(name, 1.0)) <= 0:
            raise ValueError(f"module_1.{name} must be positive")
    for module, keys in ((module_1, ("gate_loss_weight", "route_loss_weight", "memory_l2_weight")),
                         (module_2, ("prefix_weight", "prefix_margin", "sequence_weight",
                                     "router_guard_weight", "nll_weight"))):
        if any(float(module.get(key, 0.0)) < 0 for key in keys):
            raise ValueError("Loss weights and margins cannot be negative")
    for name in ("num_workers", "eval_num_workers", "policy_num_workers"):
        if int(config.get(name, 0)) < 0:
            raise ValueError(f"{name} cannot be negative")
    if int(config.get("prefetch_factor", 2)) < 1 or int(config.get("cpu_threads", 4)) < 1:
        raise ValueError("prefetch_factor and cpu_threads must be positive")
    pretraining = config.get("recommender_pretraining", {})
    if int(pretraining.get("epochs", 50)) < 0 or int(pretraining.get("warmup_steps", 8000)) < 0:
        raise ValueError("Pretraining epochs and warmup_steps cannot be negative")


def data_loaders(config, num_items, train, valid, test):
    collator = Collator(eos_token_id=-1, pad_token_id=0, max_length=int(config["max_his_len"]))
    def loader_options(worker_key):
        workers = min(int(config.get(worker_key, config.get("num_workers", 0))), os.cpu_count() or 1)
        options = {"num_workers": workers, "collate_fn": collator,
                   "pin_memory": torch.cuda.is_available()}
        if workers:
            options.update(persistent_workers=bool(config.get("persistent_workers", True)),
                           prefetch_factor=int(config.get("prefetch_factor", 2)))
        return options
    train_dataset = SequentialSplitDataset(
        config, num_items, train, float(config.get("train_data_ratio", 1.0))
    )
    valid_dataset = SequentialSplitDataset(
        config, num_items, valid, float(config.get("valid_data_ratio", 1.0))
    )
    test_dataset = SequentialSplitDataset(
        config, num_items, test, float(config.get("test_data_ratio", 1.0))
    )
    policy_dataset = SequentialSplitDataset(
        config, num_items, train, float(config.get("policy_data_ratio", 0.05))
    )
    module_2 = config.get("modules", {}).get("module_2", {})
    train_generator = torch.Generator().manual_seed(int(config["seed"]) + 101)
    policy_generator = torch.Generator().manual_seed(int(config["seed"]) + 202)
    return (
        DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            sampler=RandomSampler(train_dataset, generator=train_generator),
            generator=torch.Generator().manual_seed(int(config["seed"]) + 501),
            **loader_options("num_workers"),
        ),
        DataLoader(valid_dataset, batch_size=int(config["eval_batch_size"]), shuffle=False,
                   generator=torch.Generator().manual_seed(int(config["seed"]) + 503),
                   **loader_options("eval_num_workers")),
        DataLoader(test_dataset, batch_size=int(config["eval_batch_size"]), shuffle=False,
                   generator=torch.Generator().manual_seed(int(config["seed"]) + 504),
                   **loader_options("eval_num_workers")),
        DataLoader(
            policy_dataset,
            batch_size=int(module_2.get("batch_size", 2)),
            sampler=RandomSampler(policy_dataset, generator=policy_generator),
            generator=torch.Generator().manual_seed(int(config["seed"]) + 502),
            **loader_options("policy_num_workers"),
        ),
    )


def build_backbone(config, sid_centroids: torch.Tensor | None = None) -> GenerativeSIDModel:
    t5_config = T5Config(
        num_layers=int(config["encoder_layers"]),
        num_decoder_layers=int(config["decoder_layers"]),
        d_model=int(config["d_model"]),
        d_ff=int(config["d_ff"]),
        num_heads=int(config["num_heads"]),
        d_kv=int(config["d_kv"]),
        dropout_rate=float(config["dropout_rate"]),
        activation_function=str(config.get("activation_function", "relu")),
        feed_forward_proj=str(config.get("feed_forward_proj", "relu")),
        # The backbone LM head is not used: SID logits are tied to the four
        # depth-specific embeddings below. Only token 0 is needed for the T5
        # decoder start embedding, matching the efficient original trunk.
        vocab_size=int(config.get("backbone_vocab_size", 1)),
        pad_token_id=0,
        decoder_start_token_id=0,
        eos_token_id=None,
        n_positions=int(config.get("max_length", 256)),
    )
    return GenerativeSIDModel(
        config,
        T5ForConditionalGeneration(config=t5_config),
        code_length=int(config["code_length"]),
        code_number=int(config["code_num"]),
        sid_centroids=sid_centroids,
    )


def prepare_catalog(config, num_items: int):
    embedding_path = Path(str(config["data_path"])) / str(config["dataset"]) / str(
        config["semantic_emb_path"]
    )
    raw = np.load(embedding_path)
    if raw.shape[0] != num_items - 1:
        raise ValueError(
            f"Embedding rows ({raw.shape[0]}) do not match non-padding catalog items ({num_items - 1})"
        )
    padded = torch.zeros(num_items, raw.shape[1], dtype=torch.float32)
    padded[1:] = torch.as_tensor(raw, dtype=torch.float32)

    sid_path = config.get("sid_path")
    if not sid_path or not Path(str(sid_path)).is_file():
        raise FileNotFoundError(f"Fixed SID catalog not found: {sid_path}")
    codes = load_sids(str(sid_path))
    if len(codes) != num_items:
        raise ValueError("SID mapping does not cover the complete catalog including padding")
    width = int(config["code_length"])
    vocabulary = int(config["code_num"])
    if codes[0] != [-1] * width:
        raise ValueError("SID row zero must be the [-1] padding row")
    if any(len(row) != width or any(token < 0 or token >= vocabulary for token in row)
           for row in codes[1:]):
        raise ValueError("Each catalog SID must have code_length tokens in [0, code_num)")
    if len({tuple(row) for row in codes[1:]}) != len(codes) - 1:
        raise ValueError("Full catalog SIDs must be unique")
    return torch.as_tensor(codes, dtype=torch.long), padded, None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def manifest(config, trainer: Trainer) -> Dict[str, object]:
    return {
        "revision": "hgig-public-v1",
        "schema_version": "1.0",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "catalog_version": trainer.catalog_version,
        "stage2_selection_policy": "best_rcbr_epoch",
        "training_contract": dict(trainer.training_contract),
        "config": {k: v for k, v in config.items() if k != "accelerator"},
    }


def run(config: Dict[str, object]):
    accelerator = config["accelerator"]
    torch.set_num_threads(min(int(config.get("cpu_threads", 4)), os.cpu_count() or 1))
    init_seed(int(config["seed"]) + accelerator.process_index, bool(config["reproducibility"]))
    _item_map, num_items, train_split, valid_split, test_split = load_split_data(config)
    item_codes, item_embeddings, sid_centroids = prepare_catalog(config, num_items)
    loaders = data_loaders(config, num_items, train_split, valid_split, test_split)
    if accelerator.is_main_process:
        print(
            "[run] "
            f"dataset={config['dataset']} | run={config['run_name']} | "
            f"device={accelerator.device} | processes={accelerator.num_processes} | "
            "EAPM=on | RCBR=on | "
            f"eval_batch={config['eval_batch_size']} | cpu_threads={torch.get_num_threads()} | "
            f"loader_workers={[loader.num_workers for loader in loaders]}",
            flush=True,
        )
    # Reset before T5 construction to keep the backbone initialization reproducible.
    init_seed(int(config["seed"]) + accelerator.process_index, bool(config["reproducibility"]))
    use_centroids = str(config.get("sid_embedding_initialization", "gaussian")).lower()
    if use_centroids in {"random", "gaussian", "none"}:
        sid_centroids = None
    else:
        raise ValueError("The public fixed-SID runner supports Gaussian SID initialization only")
    stage1_model = build_backbone(config, sid_centroids=sid_centroids)
    trainer = Trainer(
        config,
        stage1_model,
        accelerator,
        item_codes,
        item_embeddings,
        *loaders,
    )
    # Reset the stochastic training stream after module construction.
    init_seed(int(config["seed"]) + accelerator.process_index, bool(config["reproducibility"]))
    if accelerator.is_main_process:
        results_path = Path(str(config["results_path"]))
        write_json(results_path / "run_manifest.json", manifest(config, trainer))
        save_sids(results_path / "catalog.sid.json", item_codes.tolist())
    pretraining = trainer.pretrain_recommender()
    supervised = trainer.train_supervised()
    module_2 = trainer.train_rcbr()
    final_test = trainer.test()
    results = {
        "revision": "hgig-public-v1",
        "recommender_pretraining": pretraining,
        "decoding_protocol": trainer.decoding_protocol,
        "supervised": supervised,
        "module_2": module_2,
        "final_test": final_test,
        "best_checkpoint": trainer.best_checkpoint,
    }
    if accelerator.is_main_process:
        results_path = Path(str(config["results_path"]))
        save_sids(results_path / "catalog.sid.json", item_codes.tolist())
        write_json(results_path / "final_results.json", results)
        write_json(results_path / "run_manifest.json", manifest(config, trainer))
        write_json(
            results_path / "training_summary.json",
            {
                "dataset": config["dataset"],
                "run_name": config["run_name"],
                "modules": config.get("modules", {}),
                "training_contract": dict(trainer.training_contract),
                "supervised": supervised,
                "recommender_pretraining": pretraining,
                "module_2": module_2,
                "best_checkpoint": trainer.best_checkpoint,
                "final_test": final_test,
                "mechanism_trace": str(results_path / "mechanism_trace.json"),
            },
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    accelerator.wait_for_everyone()
    return results


if __name__ == "__main__":
    args, unknown = parse_arguments()
    configuration = load_configuration(args.config)
    deep_update(configuration, parse_command_line_args(unknown))
    configuration = convert_config_dict(configuration)
    validate_config(configuration)

    run_name = str(configuration.get("run_name") or "hgig")
    output_root = Path(str(configuration.get("output_root", ".")))
    configuration["run_name"] = run_name
    configuration["results_path"] = str(
        output_root / "results" / str(configuration["dataset"]) / run_name
    )
    configuration["save_path"] = str(
        output_root / "checkpoints" / str(configuration["dataset"]) / run_name
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(configuration.get("gradient_accumulation_steps", 1))
    )
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "The HGIG release is single-GPU; launch it with ordinary "
            "Python and expose only one CUDA device."
        )
    configuration["accelerator"] = accelerator
    run(configuration)
