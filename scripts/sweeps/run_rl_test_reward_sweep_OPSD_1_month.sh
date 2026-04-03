#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$repo_root/scripts/sweeps/run_rl_test_sweep.sh" \
  --repo-root "$repo_root" \
  --base-out "$repo_root/outputs/RL_REWARD_SWEEP_OPSD_1_MONTH" \
  --train-source primary \
  --train-root-primary "$repo_root/outputs/RL_REWARD_SWEEP_OPSD_1_MONTH/train" \
  --dataset-path "$repo_root/data/scenario_0_timeseries_hourly.csv" \
  --test-start-step 0 \
  --test-steps 5760 \
  --max-parallel 10 \
  --device cpu