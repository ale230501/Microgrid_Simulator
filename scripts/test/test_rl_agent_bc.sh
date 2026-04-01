#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Simple runner for a trained RL agent.

MODEL_PATH="BC_MPC/outputs/ppo_trained/ppo_20260122_104935/ppo_trained.zip"
DATASET_PATH="data/scenario_0_timeseries_hourly.csv"
MODE="test"  # eval | test | rollout
CONFIG="configs/controllers/rl/opsd/params_RL_agent_OPSD_1_WEEK.yml"
START_STEP=0
TIMESTEPS=5760

python RL_AGENT/ems_offline_RL_agent.py \
  --mode "${MODE}" \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  --test-start-step "${START_STEP}" \
  --test-steps "${TIMESTEPS}" \
  --test-dataset-path "${DATASET_PATH}"

