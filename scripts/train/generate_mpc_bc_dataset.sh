#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python BC_MPC/generate_mpc_bc_dataset.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_WEEK.yml \
  --mpc-config configs/controllers/mpc/params_OPSD.yml \
  --dataset-path data/warmup_MPC.csv \
  --output-dir BC_MPC/outputs/ \
  --max-steps 2688
