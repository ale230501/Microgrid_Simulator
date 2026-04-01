# Tools

Utility scripts for analysis and sweep orchestration support.

## Plotting
- `tools/plot_pareto_soh_costs.py`
- `tools/plot_pareto_soh_costs_opsd.py`

## Sweep helpers
- `tools/rl_reward_sweep.py`
- `tools/rl_eval_sweep.py`

Canonical sweep specs are under `configs/sweeps/`.
The copies in `tools/rl_*_sweep_example_*.yml` are kept for compatibility.

## Local binaries
- `tools/tectonic/` contains a local Tectonic bundle used for report workflows.
- `tools/tectonic/cache/` is local cache and should not be versioned.
