#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base_out=""
train_root_primary=""
train_root_secondary=""
train_source="primary"
dataset_path=""
test_start_step=0
test_steps=5760
max_parallel=15
device="cpu"
action_deadband_kwh="0.0"
soc_action_guard_band="0.0"

usage() {
  cat <<'USAGE'
Usage: run_rl_test_sweep.sh [options]

Options:
  --repo-root PATH
  --base-out PATH
  --train-root-primary PATH
  --train-root-secondary PATH
  --train-source primary|secondary|both
  --dataset-path PATH
  --test-start-step N
  --test-steps N
  --max-parallel N
  --device cpu|cuda
  --action-deadband-kwh FLOAT
  --soc-action-guard-band FLOAT
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="$2"; shift 2 ;;
    --base-out) base_out="$2"; shift 2 ;;
    --train-root-primary) train_root_primary="$2"; shift 2 ;;
    --train-root-secondary) train_root_secondary="$2"; shift 2 ;;
    --train-source) train_source="$2"; shift 2 ;;
    --dataset-path) dataset_path="$2"; shift 2 ;;
    --test-start-step) test_start_step="$2"; shift 2 ;;
    --test-steps) test_steps="$2"; shift 2 ;;
    --max-parallel) max_parallel="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --action-deadband-kwh) action_deadband_kwh="$2"; shift 2 ;;
    --soc-action-guard-band) soc_action_guard_band="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$repo_root"

[[ -n "$base_out" ]] || base_out="$repo_root/outputs/RL_TRAINING"
[[ -n "$train_root_primary" ]] || train_root_primary="$repo_root/outputs/RL_TRAINING/train"
[[ -n "$train_root_secondary" ]] || train_root_secondary="$repo_root/BC_MPC/outputs/ppo_trained"
[[ -n "$dataset_path" ]] || dataset_path="$repo_root/data/scenario_0_timeseries_hourly.csv"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_exe="$VIRTUAL_ENV/bin/python"
else
  python_exe="python"
fi

test_script="$repo_root/RL_AGENT/ems_offline_RL_agent.py"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

case "$train_source" in
  primary) train_roots=("$train_root_primary") ;;
  secondary) train_roots=("$train_root_secondary") ;;
  both) train_roots=("$train_root_primary" "$train_root_secondary") ;;
  *) echo "Invalid --train-source: $train_source" >&2; exit 1 ;;
esac

if [[ ! -f "$dataset_path" ]]; then
  echo "Dataset not found: $dataset_path" >&2
  exit 1
fi

log_dir="$base_out/test_logs"
mkdir -p "$log_dir" "$base_out/test_configs"

mapfile -t models < <(
  for root in "${train_roots[@]}"; do
    if [[ ! -d "$root" ]]; then
      echo "[WARN] Training output not found: $root" >&2
      continue
    fi
    find "$root" -type f \( -name 'model_final.zip' -o -name 'ppo_trained.zip' \)
  done | sort -u
)

if (( ${#models[@]} == 0 )); then
  echo "No trained models found under: ${train_roots[*]}"
  exit 0
fi

echo "Found ${#models[@]} trained models. Max parallel tests: $max_parallel (device=$device)"
echo "Train roots: ${train_roots[*]}"
echo "Dataset: $dataset_path"

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

for model in "${models[@]}"; do
  run_dir="$(dirname "$model")"
  cfg="$(find "$run_dir" -maxdepth 1 -type f -name '*.yml' | sort | head -n 1 || true)"
  if [[ -z "$cfg" ]]; then
    echo "[WARN] Skipping (no .yml found): $run_dir" >&2
    continue
  fi

  run_name="$(basename "$run_dir")_test"
  tmp_cfg="$base_out/test_configs/$run_name.yml"

  "$python_exe" - "$cfg" "$tmp_cfg" "$dataset_path" "$action_deadband_kwh" "$soc_action_guard_band" <<'PY'
import sys
import pathlib
import yaml

cfg_path, out_path, dataset_path, action_deadband, soc_guard = sys.argv[1:6]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rl = cfg.setdefault("rl", {})
rl["dataset_path_test"] = dataset_path
rl["randomize_initial_SoC"] = False
rl["action_deadband_kwh"] = float(action_deadband)
rl["soc_action_guard_band"] = float(soc_guard)
rl["tb_reasoning_plot_every"] = 0
rl["tb_enable_figures"] = False
rl["tb_enable_histograms"] = False
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  log_path="$log_dir/$run_name.log"
  echo "Testing $run_name"

  (
    cd "$repo_root"
    "$python_exe" "$test_script" \
      --mode test \
      --config "$tmp_cfg" \
      --model-path "$model" \
      --output-dir "$base_out" \
      --run-name "$run_name" \
      --device "$device" \
      --test-start-step "$test_start_step" \
      --test-steps "$test_steps" \
      --test-dataset-path "$dataset_path" \
      >"$log_path" 2>&1
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