# HGIG: fixed-SID generative recommendation

This repository contains the complete HGIG recommendation model. It trains a
T5-style sequence-to-SID backbone, Evidence-Adaptive Prefix Memory (EAPM),
and Router-aware Contrastive Branch Repair (RCBR). It does not run ablation
studies. The public runner is single-GPU.

## Project layout

- `main.py`: configuration, training, validation, and test entry point.
- `model.py`, `trainer.py`: generative model and two-stage training.
- `sid_modules/`: EAPM catalog routing, gate supervision, RCBR candidate mining,
  and branch-repair losses.
- `hierarchical_credit/`: SID trie and batched beam decoding.
- `config/`: configurations for Scientific, Instruments, and Games.
- `tests/`: focused unit and invariant checks.

## Environment

Install a PyTorch build compatible with your CUDA driver (or CPU), then install
the remaining packages:

```bash
python -m pip install -r requirements.txt
```

The exact package versions are listed for reproducibility, but hardware-specific
PyTorch wheels are intentionally not bundled.

## Input data

The runner does **not** download or distribute datasets, item embeddings, or
tokenizer checkpoints. Prepare these files for each dataset named `<name>`:

```text
data/<name>/<name>.train.jsonl
data/<name>/<name>.valid.jsonl
data/<name>/<name>.test.jsonl
data/<name>/<name>.emb_map.json
data/<name>/<semantic_emb_path>
data/<name>/<name>.sid.json
```

Each JSONL row has `inter_history` (an ordered list of item tokens) and
`target_id` (the next-item token). The map JSON maps item tokens to contiguous
integer IDs. ID 0 is reserved for padding, and the remaining IDs must match
the row order in the `.npy` embedding matrix. The embedding matrix therefore
has one row per non-padding item. The SID JSON is a list indexed by those same
integer IDs; row 0 is `[-1, -1, -1, -1]`, and each remaining row is a unique
four-token SID with values in `[0, 256)`. The first three tokens are semantic
positions and the fourth resolves collisions. These dimensions are defaults;
change the YAML fields consistently if using a different SID scheme.

SIDs may be produced by any offline tokenizer that satisfies this format. The
tokenizer is not updated during recommendation training. Set `data_path`,
`semantic_emb_path`, and `sid_path` in a configuration file or override them on
the command line. The current runner does not create SIDs or load an RQ-VAE.

## Run

From the repository root, run one dataset at a time:

```bash
python main.py --config config/scientific.yaml
python main.py --config config/instrument.yaml
python main.py --config config/game.yaml
```

For an external data directory, pass overrides such as
`--data_path=/path/to/data --sid_path=/path/to/scientific.sid.json`.
The YAML files contain the hyperparameters used by this implementation; they
are not evidence of an achieved result on a new dataset. The model selects the
best RCBR epoch using validation `ndcg@10` and tests that Stage-2 checkpoint,
even if its validation score is below the Stage-1 checkpoint. It does not
guarantee an improvement in held-out metrics.

Outputs are saved under `results/<dataset>/hgig/` and
`checkpoints/<dataset>/hgig/`, including `final_results.json`,
`training_summary.json`, the mechanism trace, the fixed SID catalog, and model
checkpoints. The default random seed is 2020.

## Tests and provenance

Run `python -m pytest tests` on an environment with the development dependency
installed. This public folder was prepared by static inspection; no local
end-to-end training run is claimed. See [NOTICE.md](NOTICE.md) before publishing
or redistributing code and externally prepared data.
