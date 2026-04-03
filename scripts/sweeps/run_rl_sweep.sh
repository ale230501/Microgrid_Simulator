#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cfg_root=""
base_out=""
max_parallel=10
omp_threads=8
mkl_threads=8

usage() {
  cat <<'USAGE'
Usage: run_rl_sweep.sh [options]

Options:
  --repo-root PATH
  --cfg-root PATH
  --base-out PATH
  --max-parallel N
  --omp-threads N
  --mkl-threads N
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="$2"; shift 2 ;;
    --cfg-root) cfg_root="$2"; shift 2 ;;
    --base-out) base_out="$2"; shift 2 ;;
    --max-parallel) max_parallel="$2"; shift 2 ;;
    --omp-threads) omp_threads="$2"; shift 2 ;;
    --mkl-threads) mkl_threads="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$repo_root"
[[ -n "$cfg_root" ]] || cfg_root="$repo_root/configs/controllers/rl"
[[ -n "$base_out" ]] || base_out="$repo_root/outputs/RL_TRAINING"

if [[ ! -d "$cfg_root" ]]; then
  echo "Config root not found: $cfg_root" >&2
  exit 1
fi

mapfile -t cfgs < <(find "$cfg_root" -type f -name '*.yml' | sort)
if (( ${#cfgs[@]} == 0 )); then
  echo "No .yml configs found under $cfg_root" >&2
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_exe="$VIRTUAL_ENV/bin/python"
else
  python_exe="python"
fi

train_script="$repo_root/RL_AGENT/ems_offline_RL_agent.py"

export OMP_NUM_THREADS="$omp_threads"
export MKL_NUM_THREADS="$mkl_threads"

log_dir="$base_out/logs"
mkdir -p "$log_dir" "$base_out/configs"

echo "Found ${#cfgs[@]} config files. Max parallel jobs: $max_parallel"

norm_variants=(
  "n1|true|false|true|false"
  "n2|true|true|true|false"
  "n3|false|false|false|false"
  "n4|true|false|true|true"
  "n5|true|true|true|true"
)
reward_variants=(
  "r1|0.0"
  "r2|2.0"
)
soc_variants=(
  "s1|true"
  "s2|false"
)

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

for cfg in "${cfgs[@]}"; do
  base="$(basename "$cfg" .yml)"

  for n in "${norm_variants[@]}"; do
    IFS='|' read -r n_id obs_enabled norm_already act_enabled rew_enabled <<< "$n"
    for r in "${reward_variants[@]}"; do
      IFS='|' read -r r_id coeff_ssr <<< "$r"
      for s in "${soc_variants[@]}"; do
        IFS='|' read -r s_id rand_soc <<< "$s"

        run_name="${base}_${n_id}_${r_id}_${s_id}"
        tmp_cfg="$base_out/configs/$run_name.yml"

        "$python_exe" - "$cfg" "$tmp_cfg" "$obs_enabled" "$norm_already" "$act_enabled" "$rew_enabled" "$coeff_ssr" "$rand_soc" <<'PY'
import sys
import pathlib
import yaml

cfg_path, out_path = sys.argv[1], sys.argv[2]
obs_enabled = sys.argv[3].lower() == "true"
norm_already = sys.argv[4].lower() == "true"
act_enabled = sys.argv[5].lower() == "true"
rew_enabled = sys.argv[6].lower() == "true"
coeff_ssr = float(sys.argv[7])
rand_soc = sys.argv[8].lower() == "true"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rl = cfg.setdefault("rl", {})
norm = rl.setdefault("normalization", {})
obs = norm.setdefault("observations", {})
act = norm.setdefault("actions", {})
rew = norm.setdefault("reward", {})

obs["enabled"] = obs_enabled
obs["normalize_already_scaled"] = norm_already
act["enabled"] = act_enabled
rew["enabled"] = rew_enabled

reward = rl.setdefault("reward", {})
reward["coeff_SSR"] = coeff_ssr

rl["randomize_initial_SoC"] = rand_soc

pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

        log_path="$log_dir/$run_name.log"
        echo "Starting $run_name"

        (
          cd "$repo_root"
          "$python_exe" "$train_script" \
            --mode train \
            --config "$tmp_cfg" \
            --run-name "$run_name" \
            --output-dir "$base_out" \
            >"$log_path" 2>&1
        ) &

        pids+=("$!")
        while (( ${#pids[@]} >= max_parallel )); do
          wait_one || true
          prune_pids
        done
      done
    done
  done
done

for pid in "${pids[@]:-}"; do
  wait "$pid" || true
done

echo "Done. Logs: $log_dir"