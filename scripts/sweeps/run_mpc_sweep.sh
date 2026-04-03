#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mpc_root=""
inference_dataset=""
forecast_dataset=""
start_step=0
end_step=""
max_parallel=4

usage() {
  cat <<'USAGE'
Usage: run_mpc_sweep.sh [options]

Options:
  --repo-root PATH
  --mpc-root PATH
  --inference-dataset PATH
  --forecast-dataset PATH
  --start-step N
  --end-step N
  --max-parallel N
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="$2"; shift 2 ;;
    --mpc-root) mpc_root="$2"; shift 2 ;;
    --inference-dataset) inference_dataset="$2"; shift 2 ;;
    --forecast-dataset) forecast_dataset="$2"; shift 2 ;;
    --start-step) start_step="$2"; shift 2 ;;
    --end-step) end_step="$2"; shift 2 ;;
    --max-parallel) max_parallel="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$repo_root"

[[ -n "$mpc_root" ]] || mpc_root="$repo_root/outputs/MPC"
[[ -n "$inference_dataset" ]] || inference_dataset="$repo_root/data/scenario_14_timeseries_hourly.csv"
[[ -n "$forecast_dataset" ]] || forecast_dataset="$inference_dataset"

if [[ ! -d "$mpc_root" ]]; then
  echo "MPC output root not found: $mpc_root" >&2
  exit 1
fi
if [[ ! -f "$inference_dataset" ]]; then
  echo "Inference dataset not found: $inference_dataset" >&2
  exit 1
fi
if [[ ! -f "$forecast_dataset" ]]; then
  echo "Forecast dataset not found: $forecast_dataset" >&2
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_exe="$VIRTUAL_ENV/bin/python"
else
  python_exe="python"
fi

mpc_script="$repo_root/MODEL_PREDICTIVE/ems_offline_mpc_v0.py"

log_dir="$mpc_root/test_logs"
mkdir -p "$log_dir"

mapfile -t runs < <(find "$mpc_root" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#runs[@]} == 0 )); then
  echo "No MPC run folders found under: $mpc_root"
  exit 0
fi

pids=()
wait_one() {
  if wait -n 2>/dev/null; then
    return 0
  fi
  local first_pid="${pids[0]:-}"
  if [[ -n "$first_pid" ]]; then
    wait "$first_pid" || true
  fi
}
prune_pids() {
  local live=()
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live+=("$pid")
    fi
  done
  pids=("${live[@]:-}")
}

for run_dir in "${runs[@]}"; do
  cfg=""
  if [[ -f "$run_dir/params_OPSD.yml" ]]; then
    cfg="$run_dir/params_OPSD.yml"
  else
    cfg="$(find "$run_dir" -maxdepth 1 -type f -name '*.yml' | sort | head -n 1 || true)"
  fi

  if [[ -z "$cfg" ]]; then
    echo "[WARN] Skipping (no .yml found): $run_dir" >&2
    continue
  fi

  run_name="$(basename "$run_dir")"
  log_path="$log_dir/$run_name.log"
  echo "Running MPC for $run_name"

  (
    cd "$repo_root"
    cmd=(
      "$python_exe" "$mpc_script"
      --config "$cfg"
      --MPC_inference_dataset_path "$inference_dataset"
      --MPC_forecast_dataset_path "$forecast_dataset"
      --start-step "$start_step"
    )
    if [[ -n "$end_step" ]]; then
      cmd+=(--end-step "$end_step")
    fi
    "${cmd[@]}" >"$log_path" 2>&1
  ) &

  pids+=("$!")
  while (( ${#pids[@]} >= max_parallel )); do
    wait_one || true
    prune_pids
  done
done

for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done

echo "Done. Logs: $log_dir"