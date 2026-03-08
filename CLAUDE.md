# CLAUDE.md — AI Assistant Guide for autoresearch

## Project Overview

Autonomous AI research system for LLM pretraining by @karpathy. An AI agent autonomously modifies training code, runs 5-minute experiments, evaluates results, and keeps or discards changes. Simplified single-GPU implementation based on [nanochat](https://github.com/karpathy/nanochat).

## Repository Structure

```
prepare.py      — Fixed constants, data prep, tokenizer, dataloader, evaluation (DO NOT MODIFY)
train.py        — GPT model, optimizer, training loop (AGENT MODIFIES THIS FILE ONLY)
program.md      — Agent instructions/workflow for autonomous research (human edits)
pyproject.toml  — Dependencies (do not modify)
analysis.ipynb  — Jupyter notebook for analyzing results.tsv
results.tsv     — Experiment log (created per-run, tab-separated)
run.log         — Training output log (created per-run)
```

## Critical Rules

1. **Only edit `train.py`** — this is the sole file the agent modifies
2. **Never modify `prepare.py`** — it contains the fixed evaluation harness, dataloader, and constants
3. **Never install new packages** — only use what's in `pyproject.toml`
4. **Never modify `evaluate_bpb()`** — it is the ground truth metric in `prepare.py`
5. **Fixed time budget** — training always runs for exactly 5 minutes (wall clock)
6. **Metric: val_bpb** — lower is better, vocabulary-size-independent

## Development Environment

- **Python**: 3.10+ (see `.python-version`)
- **Package manager**: `uv` (not pip)
- **GPU**: Single NVIDIA GPU required (tested on H100)
- **PyTorch**: 2.9.1 with CUDA 12.8
- **Data cache**: `~/.cache/autoresearch/`

## Common Commands

```bash
# Install dependencies
uv sync

# One-time data preparation (~2 min)
uv run prepare.py

# Run a training experiment (~5 min)
uv run train.py

# Run training and capture output (for autonomous loop)
uv run train.py > run.log 2>&1

# Extract key metrics from log
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## No Linting, Formatting, or Tests

This project intentionally has no linter, formatter, or test suite. The evaluation metric (`val_bpb`) IS the test. There is no CI/CD pipeline.

## Autonomous Experiment Workflow

See `program.md` for the full protocol. Summary:

1. **Setup**: Create branch `autoresearch/<tag>`, verify data, initialize `results.tsv`
2. **Loop**: Modify `train.py` -> commit -> run `uv run train.py > run.log 2>&1` -> extract metrics -> log to `results.tsv` -> keep (if improved) or `git reset` (if not)
3. **Never stop**: The agent runs indefinitely until manually interrupted

### results.tsv Format (tab-separated)

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
```

- `status`: `keep`, `discard`, or `crash`
- `memory_gb`: peak_vram_mb / 1024, rounded to .1f
- Use `0.000000` val_bpb and `0.0` memory for crashes

## Key Architecture Details (train.py)

- GPT model with configurable depth/dimension (ASPECT_RATIO=64, depth*64=model_dim)
- MuonAdamW optimizer: Muon for 2D matrix params, AdamW for the rest
- `torch.compile()` with `dynamic=False, fullgraph=True`
- RMSNorm, rotary embeddings, Flash Attention 3 (Hopper fallback)
- Value embeddings (ResFormer-style), sliding window pattern "SSSL"
- Learning rates scaled proportional to 1/sqrt(d_model)
- Seed: `torch.manual_seed(42)` for reproducibility

## Design Philosophy

- **Simplicity first**: prefer deleting code over adding it
- **Metric-driven**: success measured solely by val_bpb
- **Weigh complexity vs improvement**: a tiny gain with ugly complexity is not worth it; a simplification with equal results is always worth it
- **Self-contained**: one GPU, one file, one metric, no distributed training
