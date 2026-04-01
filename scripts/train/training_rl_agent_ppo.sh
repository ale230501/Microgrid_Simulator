#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Standard PPO training (no behavioral cloning).

CONFIG="configs/controllers/rl/opsd/params_RL_agent_OPSD_1_WEEK.yml"
DATASET_PATH="data/scenario_0_timeseries_hourly.csv"

python RL_AGENT/ems_offline_RL_agent.py \
  --mode train \
  --config "${CONFIG}" \
  --train-dataset-path "${DATASET_PATH}"

