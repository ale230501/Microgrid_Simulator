#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Offline MPC test runner.

CONFIG="configs/controllers/mpc/params_OPSD.yml"
INFERENCE_DATASET="data/scenario_14_timeseries_hourly.csv"
FORECAST_DATASET="data/scenario_14_timeseries_hourly.csv"
START_STEP=0
END_STEP=2688

python MODEL_PREDICTIVE/ems_offline_mpc_v0.py \
  --config "${CONFIG}" \
  --MPC_inference_dataset_path "${INFERENCE_DATASET}" \
  --MPC_forecast_dataset_path "${FORECAST_DATASET}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}"

