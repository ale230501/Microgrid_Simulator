# Project Structure Guide

This file defines the canonical folder layout and where to place new files.

## Canonical top-level directories
- `SIMULATOR/`: simulator engine and core microgrid logic.
- `RULE_BASED/`: rule-based controller code and params.
- `MODEL_PREDICTIVE/`: MPC controller code and params.
- `RL_AGENT/`: RL controller code, env glue, utilities.
- `BC_MPC/`: behavior cloning + PPO fine-tuning from MPC data.
- `configs/`: canonical YAML configs for controllers and sweeps.
- `data/`: input datasets (train/test/eval/warmup CSV files).
- `scripts/`: operational entry points for training/testing/sweeps.
- `tools/`: analysis and plotting utilities.
- `logs/`: local runtime logs (ignored except README).
- `outputs/`: generated artifacts (ignored in git).

## Scripts layout
- `scripts/train/`: long-running training jobs (`.sh`).
- `scripts/test/`: offline test/evaluation runs (`.sh`).
- `scripts/sweeps/`: batch/sweep orchestration (`.ps1`, helper prompts).

Each script under `scripts/` resolves repository root automatically, so it can be launched from any working directory.

## Legacy compatibility folders
- `Simulation_Scripts/`
- `test_sweep/`

These are compatibility wrappers only. New scripts should be added under `scripts/`, not in legacy folders.

Legacy config files are also kept in:
- `RL_AGENT/PARAMS/`
- `MODEL_PREDICTIVE/`
- `RULE_BASED/`
Use `configs/` as canonical source for new work.

## Naming and placement rules
- New orchestration scripts: put in `scripts/` and group by purpose (`train`, `test`, `sweeps`).
- New model/controller code: place in the corresponding controller folder (`RL_AGENT/`, `MODEL_PREDICTIVE/`, `RULE_BASED/`, `BC_MPC/`).
- New controller configs: place in `configs/controllers/<controller>/...`.
- New sweep specs/examples: place in `configs/sweeps/`.
- New datasets: place in `data/` with explicit suffixes (`_train`, `_test`, `_eval`, `_warmup`).
- New analysis utilities: place in `tools/`.
- Tool caches/bundles: keep local caches ignored (e.g. `tools/tectonic/cache/`) and write generated reports/logs to `outputs/` or `logs/`.
