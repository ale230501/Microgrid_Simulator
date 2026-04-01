#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python BC_MPC/train_bc_from_mpc.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_WEEK.yml \
  --dataset BC_MPC/outputs/mpc_bc_dataset.npz \
  --output-dir BC_MPC/outputs/policy/bc_model \
  --epochs 160 \
  --batch-size 32
