#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python BC_MPC/training_rl_agent_ppo.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_WEEK.yml \
  --bc-policy BC_MPC/outputs/policy/bc_model/bc_20260122_202158/bc_policy.pt \
  --output-dir BC_MPC/outputs/ppo_trained
