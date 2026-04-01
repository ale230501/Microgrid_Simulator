# Microgrid Simulator

Offline microgrid simulator with three control paths: rule based control, MPC baseline, and an RL agent.

## Project layout
- `SIMULATOR/` microgrid core, models, and utilities
- `RULE_BASED/` offline RBC runner and YAML config
- `RL_AGENT/` offline RL training and evaluation
- `MODEL_PREDICTIVE/` offline MPC runner and config
- `BC_MPC/` behavior cloning from MPC trajectories
- `configs/` canonical YAML configuration files (controllers and sweeps)
- `data/` input datasets
- `scripts/train/` canonical training scripts
- `scripts/test/` canonical offline test scripts
- `scripts/sweeps/` canonical sweep and batch-eval scripts
- `configs/sweeps/` canonical sweep example YAML files
- `tools/` analysis utilities (plotting, sweep helpers)
- `logs/` local runtime logs (ignored except README)
- `outputs/` logs and plots

For compatibility, the old folders `Simulation_Scripts/` and `test_sweep/` still exist as wrappers that call `scripts/`.
Legacy config files under `RL_AGENT/PARAMS/`, `MODEL_PREDICTIVE/`, and `RULE_BASED/` are kept for compatibility, but canonical configs are under `configs/`.
See `PROJECT_STRUCTURE.md` for a full structure map and conventions.

## Requirements
- Python 3.10+
- CSV with columns `datetime`, `solar`, `load` (energy per timestep in kWh)
- Available datasets in `data/`:
  - `scenario_0_timeseries_hourly.csv` (derived from `pymgrid` scenario 0)
  - `warmup_MPC.csv` (warmup/BC generation)

Install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Conda setup for a remote server (VS Code Remote):
```bash
conda create -n microgrid-sim-server python=3.10 -y
conda activate microgrid-sim-server
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the server does not have a Gurobi license, set `mpc.solver: ECOS_BB` in:
- `configs/controllers/mpc/params_OPSD.yml`

## Code documentation workflow
- Agent instructions are in `AGENTS.md`.
- Per-file technical docs are in `docs/tex/files/`.
- Chapter-level code guide is in `docs/tex/chapters/` with root `docs/tex/main.tex`.

Before commit, run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/docs/check_code_docs_sync.ps1
```

## RBC offline run
```bash
python RULE_BASED/ems_offline.py --config configs/controllers/rbc/params_OPSD.yml --start-step 0 --end-step 6000
```
Config file: `configs/controllers/rbc/params_OPSD.yml`.

If `--end-step` is omitted, the run uses `start_step + steps` from the config.
`--inference-dataset-path` can be omitted if dataset path is set in config
(`ems.inference_dataset_path` or `scenario.inference_dataset_path`).
To use the same dataset bundle as a `pymgrid` scenario (separate `LoadModule`/`RenewableModule`/`GridModule` files),
set in config:
- `scenario.dataset_mode: pymgrid_bundle`
- `scenario.load_dataset_path`, `scenario.pv_dataset_path`, `scenario.grid_dataset_path`
For realistic battery chemistry models (`LFP`/`NMC`/`NCA`), you can disable SoH degradation by setting
`battery.disable_soh_degradation: true` in the YAML config (default is `false`).

## RL agent (offline)
Train:
```bash
python RL_AGENT/ems_offline_RL_agent.py --mode train
# or
bash scripts/train/training_rl_agent_ppo.sh
```

Evaluate:
```bash
python RL_AGENT/ems_offline_RL_agent.py --mode eval --model-path <path>
# or quick offline test
bash scripts/test/test_rl_agent.sh
```

Config file: `configs/controllers/rl/opsd/params_RL_agent_OPSD_1_DAY.yml`.

Useful RL energy-routing flags in the YAML config:
- `rl.allow_grid_charge_for_battery: false` limits battery charging to instantaneous PV surplus only.
- `rl.allow_battery_export_to_grid: false` limits battery discharge to local load only, avoiding export caused by the battery.

## MPC baseline
The MPC implementation lives under `SIMULATOR/src/pymgrid/algos/mpc/`.
Offline runner:
```bash
python MODEL_PREDICTIVE/ems_offline_mpc_v0.py --config configs/controllers/mpc/params_OPSD.yml --start-step 0 --end-step 2688
# or
bash scripts/test/test_mpc.sh
```
If `--end-step` is omitted, the run uses `start_step + steps` from the config.
`--MPC_inference_dataset_path` and `--MPC_forecast_dataset_path` can be omitted if dataset paths are set in config
(`ems.*` or `scenario.*`; forecast falls back to inference path).

## Sweep scripts
PowerShell sweep entry points are now under `scripts/sweeps/`, for example:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sweeps\run_rl_sweep.ps1
```

## Outputs
- RBC logs: `outputs/RBC/RBC_data_<timestamp>/microgrid_log.csv`
- RL logs and checkpoints: under the output directory defined in the RL config
- Plots and battery transition history under `outputs/` or the RL run folder
- Optional local/runtime logs can be kept in `logs/`
