# Scripts

Canonical script entry points live here.
Controller configs referenced by these scripts are under `configs/controllers/`.

## Train
- `scripts/train/training_rl_agent_ppo.sh`
- `scripts/train/training_rl_agent_ppo_bc.sh`
- `scripts/train/generate_mpc_bc_dataset.sh`
- `scripts/train/train_bc_from_mpc.sh`

## Test
- `scripts/test/test_rbc.sh`
- `scripts/test/test_mpc.sh`
- `scripts/test/test_rl_agent.sh`
- `scripts/test/test_rl_agent_bc.sh`

## Sweeps
- `scripts/sweeps/run_rl_sweep.ps1`
- `scripts/sweeps/run_mpc_sweep.ps1`
- `scripts/sweeps/run_rl_test_sweep.ps1`
- `scripts/sweeps/run_rl_test_reward_sweep_OPSD_1_week.ps1`
- `scripts/sweeps/run_rl_test_reward_sweep_OPSD_1_week_alt.ps1`
- `scripts/sweeps/run_rl_test_reward_sweep_OPSD_1_month.ps1`

## Docs
- `scripts/docs/check_code_docs_sync.ps1`
  - Verifica che, quando viene modificato un file Python core, sia aggiornato anche il relativo file `.tex` in `docs/tex/files/`.

Legacy paths under `Simulation_Scripts/` and `test_sweep/` are compatibility wrappers.
