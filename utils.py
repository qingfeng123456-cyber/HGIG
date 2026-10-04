"""Small data, configuration, and reproducibility helpers for HGIG."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import torch


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def load_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def init_seed(seed: int, reproducibility: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not reproducibility
    torch.backends.cudnn.deterministic = reproducibility


def convert_config_dict(config: dict) -> dict:
    for key, value in list(config.items()):
        if isinstance(value, dict):
            config[key] = convert_config_dict(value)
        elif isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError, TypeError):
                parsed = value.lower() == "true" if value.lower() in {"true", "false"} else value
            config[key] = parsed
    return config


def parse_command_line_args(unparsed: list[str]) -> dict:
    overrides: dict = {}
    for argument in unparsed:
        if not argument.startswith("--") or "=" not in argument:
            raise ValueError(f"Expected --key=value, received {argument!r}")
        name, value = argument[2:].split("=", 1)
        if not name:
            raise ValueError("An override key cannot be empty")
        if value.lower() in {"none", "null"}:
            parsed = None
        elif value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
        cursor = overrides
        fields = name.split(".")
        for field in fields[:-1]:
            cursor = cursor.setdefault(field, {})
        cursor[fields[-1]] = parsed
    return overrides


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
