import argparse
import csv
import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
SIMULATOR_ROOT = PROJECT_ROOT / "SIMULATOR"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from EMS_RL_agent import (
    EMS_RL_Agent,
    OfflineMicrogridRLEnv,
    load_rl_agent_config,
)
from tb_logger import (
    EpisodeBehaviouralBuffer,
    EpisodeSeriesBuffer,
    MovingAverage,
    TensorboardLogger,
    TensorboardTrainCallback,
)
from SIMULATOR.tools import print_final_report, plot_results


def _prepare_output_dir(base_dir: Path, mode: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / mode / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_output_base(base_dir: Path) -> Path:
    if base_dir.exists():
        return base_dir
    base_str = str(base_dir)
    if "__" in base_str:
        candidate = Path(base_str.replace("__", "_"))
        if candidate.exists():
            print(f"[warn] output_dir not found, using {candidate}")
            return candidate
    return base_dir


def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _start_tensorboard(log_dir: Path, rl_cfg: dict) -> Optional[subprocess.Popen]:
    auto_open = rl_cfg.get("tensorboard_auto_open", True)
    if not auto_open:
        return None

    host = str(rl_cfg.get("tensorboard_host", "localhost"))
    port = int(rl_cfg.get("tensorboard_port", 6006))
    url = f"http://{host}:{port}"

    if _is_port_open(host, port):
        print(f"[TensorBoard] Already running at {url}")
        webbrowser.open(url)
        return None

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[WARN] Unable to start TensorBoard: {exc}")
        print(f"[INFO] You can run it manually with: {' '.join(cmd)}")
        return None

    time.sleep(0.5)
    print(f"[TensorBoard] Running at {url}")
    webbrowser.open(url)
    return proc


def _require_cli_dataset_path(dataset_path: Optional[str], flag: str, mode: str) -> str:
    if not dataset_path:
        raise ValueError(f"{mode} requires a dataset path. Provide {flag}.")
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"{mode} dataset not found: {path}")
    return str(path)


def _is_pymgrid_bundle_mode(config: dict) -> bool:
    scenario_cfg = config.get("scenario") or {}
    mode_value = scenario_cfg.get("dataset_mode") or config.get("scenario_dataset_mode") or ""
    return str(mode_value).strip().lower() == "pymgrid_bundle"


def _select_dataset_for_mode(
    config: dict,
    mode: str,
    train_dataset_path: Optional[str] = None,
    eval_dataset_path: Optional[str] = None,
    test_dataset_path: Optional[str] = None,
) -> Optional[str]:
    rl_cfg = config.setdefault("rl", {})
    train_path = rl_cfg.get("dataset_path_train") or rl_cfg.get("dataset_path")
    eval_path = rl_cfg.get("dataset_path_eval") or train_path
    test_path = rl_cfg.get("dataset_path_test") or eval_path

    if mode == "train":
        selected = (
            _require_cli_dataset_path(train_dataset_path, "--train-dataset-path", "train")
            if train_dataset_path
            else train_path
        )
    elif mode == "eval":
        selected = (
            _require_cli_dataset_path(eval_dataset_path, "--eval-dataset-path", "eval")
            if eval_dataset_path
            else eval_path
        )
    else:
        selected = (
            _require_cli_dataset_path(test_dataset_path, "--test-dataset-path", mode)
            if test_dataset_path
            else test_path
        )

    if selected:
        rl_cfg["dataset_path"] = selected
        return selected

    if _is_pymgrid_bundle_mode(config):
        # In scenario-bundle mode, dataset CSV is optional.
        rl_cfg["dataset_path"] = None
        return None

    raise ValueError(
        f"{mode} requires a dataset path. Set rl.dataset_path_* in config or pass the CLI dataset flag."
    )


def _as_optional_int(value: object, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer or null, got {value!r}.") from None


def _apply_mode_step_split(config: dict, mode: str) -> None:
    rl_cfg = config.setdefault("rl", {})
    scenario_cfg = config.setdefault("scenario", {})
    ems_cfg = config.setdefault("ems", {})

    split_prefix = "train" if mode == "train" else "eval"
    split_start = _as_optional_int(
        rl_cfg.get(f"{split_prefix}_split_start_step"),
        f"rl.{split_prefix}_split_start_step",
    )
    split_end = _as_optional_int(
        rl_cfg.get(f"{split_prefix}_split_end_step"),
        f"rl.{split_prefix}_split_end_step",
    )
    if split_start is None and split_end is None:
        return

    if split_start is None:
        split_start = int(scenario_cfg.get("initial_step", 0) or 0)
    if split_start < 0:
        raise ValueError(f"rl.{split_prefix}_split_start_step must be >= 0, got {split_start}.")
    if split_end is not None and split_end < split_start:
        raise ValueError(
            f"rl.{split_prefix}_split_end_step ({split_end}) must be >= "
            f"rl.{split_prefix}_split_start_step ({split_start})."
        )

    scenario_cfg["initial_step"] = int(split_start)
    if split_end is not None:
        # Inclusive bound, consistent with scenario.final_step semantics in pymgrid YAML.
        scenario_cfg["final_step"] = int(split_end)

    # Keep ems slicing aligned with scenario slicing for bundle mode.
    ems_cfg["start_step"] = int(split_start)
    ems_cfg["end_step"] = None if split_end is None else int(split_end) + 1

    start_field = "train_start_step" if split_prefix == "train" else "eval_start_step"
    start_abs = int(rl_cfg.get(start_field, 0) or 0)
    start_rel = start_abs - split_start
    if start_rel < 0:
        print(
            f"[WARN] {start_field}={start_abs} is before {split_prefix}_split_start_step={split_start}; "
            "using 0 within split."
        )
        start_rel = 0
    rl_cfg[start_field] = int(start_rel)

    split_end_msg = "end" if split_end is None else str(split_end)
    print(
        f"[INFO] Applied {split_prefix} split: global[{split_start}:{split_end_msg}] "
        f"-> local start_step={rl_cfg[start_field]}."
    )

def _select_episode_steps(config: dict, mode: str) -> Optional[int]:
    rl_cfg = config.setdefault("rl", {})
    if mode == "train":
        steps = rl_cfg.get("train_episode_steps", rl_cfg.get("episode_steps"))
    else:
        steps = rl_cfg.get("eval_episode_steps", rl_cfg.get("episode_steps"))
    if steps is not None:
        rl_cfg["episode_steps"] = int(steps)
    return rl_cfg.get("episode_steps")


def _set_episode_steps(config: dict, mode: str, steps: int) -> None:
    rl_cfg = config.setdefault("rl", {})
    if mode == "train":
        rl_cfg["train_episode_steps"] = int(steps)
    else:
        rl_cfg["eval_episode_steps"] = int(steps)
    rl_cfg["episode_steps"] = int(steps)


def _get_max_steps(config_path: str, config: dict) -> int:
    env = OfflineMicrogridRLEnv(config_path=config_path, config=config, random_start=False)
    return int(env.max_steps)


def _resolve_episode_steps(mode: str, desired_steps: int, start_step: int, max_steps: int) -> int:
    if desired_steps <= 0:
        raise ValueError(f"{mode}: episode_steps must be > 0, got {desired_steps}.")
    if start_step < 0:
        raise ValueError(f"{mode}: start_step must be >= 0, got {start_step}.")
    if start_step >= max_steps:
        raise ValueError(f"{mode}: start_step {start_step} is beyond dataset max_steps {max_steps}.")

    end_step = start_step + desired_steps
    if end_step > max_steps:
        adjusted = max_steps - start_step
        print(
            f"[WARN] {mode}: start_step {start_step} + episode_steps {desired_steps} exceeds "
            f"max_steps {max_steps}. Using episode_steps={adjusted} instead."
        )
        return max(1, adjusted)
    return desired_steps


def _resolve_obs_bounds_path(model_path: str) -> Optional[Path]:
    if not model_path:
        return None
    path = Path(model_path)
    candidate = path / "obs_bounds.json" if path.is_dir() else path.parent / "obs_bounds.json"
    return candidate if candidate.exists() else None


def _resolve_normalization_stats_path(model_path: str) -> Optional[Path]:
    if not model_path:
        return None
    path = Path(model_path)
    candidate = (
        path / "normalization_state.json"
        if path.is_dir()
        else path.parent / "normalization_state.json"
    )
    return candidate if candidate.exists() else None


class EpisodePrintCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_overshoots = []
        self._episode_count = 0

    def _on_training_start(self) -> None:
        num_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_rewards = [0.0 for _ in range(num_envs)]
        self._episode_lengths = [0 for _ in range(num_envs)]
        self._episode_overshoots = [0 for _ in range(num_envs)]
        self._episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for idx, reward in enumerate(rewards):
            if idx >= len(self._episode_rewards):
                continue
            self._episode_rewards[idx] += float(reward)
            self._episode_lengths[idx] += 1
            if idx < len(infos) and isinstance(infos[idx], dict):
                try:
                    self._episode_overshoots[idx] = int(infos[idx].get("battery_overshoots") or 0)
                except (TypeError, ValueError):
                    self._episode_overshoots[idx] = 0
        for idx, done in enumerate(dones):
            if not done or idx >= len(self._episode_rewards):
                continue
            self._episode_count += 1
            print(
                f"[train episode {self._episode_count}] "
                f"steps={self._episode_lengths[idx]} reward={self._episode_rewards[idx]:.3f} "
                f"overshoots={self._episode_overshoots[idx]}"
            )
            self._episode_rewards[idx] = 0.0
            self._episode_lengths[idx] = 0
            self._episode_overshoots[idx] = 0
        return True


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _write_action_log_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "requested_kwh",
        "clipped_kwh",
        "actual_kwh",
        "was_clipped",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_battery_action_request_vs_actual(log_path: Path, out_path: Path, title: str) -> None:
    if not log_path.exists():
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = []
    requested = []
    actual = []
    clipped = []
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(_safe_float(row.get("step"), 0)))
            requested.append(_safe_float(row.get("requested_kwh"), 0.0))
            actual.append(_safe_float(row.get("actual_kwh"), 0.0))
            clipped.append(_safe_float(row.get("clipped_kwh"), 0.0))

    if not steps:
        return

    delta = [r - a for r, a in zip(requested, actual)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(actual, label="Actual (kWh)", color="#ff7f0e", linewidth=1.0)
    axes[0].plot(
        requested,
        label="Requested (kWh)",
        color="#000000",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        zorder=5,
    )
    axes[0].plot(clipped, label="Clipped (kWh)", color="#7f7f7f", linewidth=0.8, alpha=0.7)
    axes[0].axhline(0.0, color="#888888", linewidth=0.6)
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)

    axes[1].plot(delta, label="Requested - Actual (kWh)", color="#1f77b4", linewidth=0.8)
    axes[1].axhline(0.0, color="#888888", linewidth=0.6)
    axes[1].set_ylabel("Delta (kWh)")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _plot_rl_request_vs_max(log_df, config: dict, output_dir: Path, title: str) -> None:
    if log_df is None or log_df.empty:
        return

    battery_cfg = (config or {}).get("battery", {}) or {}
    capacity = _safe_float(battery_cfg.get("capacity"), 0.0)
    power_max = _safe_float(battery_cfg.get("power_max"), 0.0)
    min_soc = _safe_float(battery_cfg.get("min_soc"), 0.0)
    max_soc = _safe_float(battery_cfg.get("max_soc"), 0.0)
    sample_time = _safe_float(battery_cfg.get("sample_time"), 0.0)
    if capacity <= 0.0 or power_max <= 0.0 or sample_time <= 0.0:
        return

    try:
        current_charge = log_df["battery_0_current_charge"].to_numpy(dtype=float)
        charge_req = log_df["battery_0_charge_amount"].to_numpy(dtype=float)
        discharge_req = log_df["battery_0_discharge_amount"].to_numpy(dtype=float)
    except KeyError:
        return

    max_energy = power_max * sample_time
    min_charge = min_soc * capacity
    max_charge = max_soc * capacity

    if current_charge.size == 0:
        return

    base_charge = np.concatenate([current_charge[:1], current_charge[:-1]])
    max_charge_step = np.minimum(max_energy, max_charge - base_charge)
    max_discharge_step = np.minimum(max_energy, base_charge - min_charge)
    max_charge_step = np.clip(max_charge_step, 0.0, None)
    max_discharge_step = np.clip(max_discharge_step, 0.0, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rl_request_vs_max_kwh.png"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(charge_req, label="RL charge request (kWh)", color="#1f77b4", linewidth=0.8)
    axes[0].plot(max_charge_step, label="Max charge possible (kWh)", color="#ff7f0e", linewidth=0.8)
    axes[0].set_ylabel("Charge (kWh)")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(discharge_req, label="RL discharge request (kWh)", color="#2ca02c", linewidth=0.8)
    axes[1].plot(
        max_discharge_step,
        label="Max discharge possible (kWh)",
        color="#d62728",
        linewidth=0.8,
    )
    axes[1].set_ylabel("Discharge (kWh)")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _compute_stats(values) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def _new_reward_diagnostics_buffers() -> dict:
    return {
        "current_step": [],
        "reward": [],
        "base_reward": [],
        "reward_term_pymgrid": [],
        "reward_term_micro_throughput": [],
        "reward_term_soe_boundary": [],
        "cost_economic": [],
        "wear_cost": [],
        "soe_violation": [],
        "soe_boundary_penalty": [],
        "action_violation": [],
        "bad_logic_penalty": [],
        "cyclic_aging": [],
        "calendar_aging": [],
        "self_sufficiency_ratio": [],
        "micro_throughput": [],
        "grid_import": [],
        "grid_export": [],
        "net_load": [],
        "battery_action_requested": [],
        "battery_action_clipped": [],
        "battery_action_actual": [],
        "battery_overshoots": [],
        "price_buy": [],
        "price_sell": [],
        "soe": [],
        "soh": [],
    }


def _append_reward_diagnostics(
    log_path: Path,
    section: str,
    episode_idx: int,
    episode_length: int,
    reward_cfg: dict,
    buffers: dict,
) -> None:
    reward_mode_raw = reward_cfg.get("mode", "custom")
    reward_mode = str(reward_mode_raw).strip().lower()
    use_pymgrid_reward = bool(reward_cfg.get("use_pymgrid_reward", False)) or reward_mode in {
        "pymgrid",
        "base",
        "module",
    }

    coeffs = {}
    active_coeffs = {}

    if use_pymgrid_reward:
        keys = [
            "reward",
            "base_reward",
            "reward_term_pymgrid",
        ]
    else:
        coeffs = {
            "economic": _safe_float(reward_cfg.get("coeff_economic", 1.0), 1.0),
            "soe_violation": _safe_float(reward_cfg.get("coeff_soe_violation", reward_cfg.get("coeff_soc_violation", 1.0)), 1.0),
            "action_violation": _safe_float(reward_cfg.get("coeff_action_violation", 1.0), 1.0),
            "micro_throughput": _safe_float(reward_cfg.get("coeff_micro_throughput", 0.0), 0.0),
            "soe_boundary": _safe_float(reward_cfg.get("coeff_soe_boundary", reward_cfg.get("coeff_soc_boundary", 0.0)), 0.0),
            "bad_logic": _safe_float(reward_cfg.get("coeff_bad_logic", 1.0), 1.0),
            "cyclic_aging": _safe_float(reward_cfg.get("coeff_cyclic_aging", 1.0), 1.0),
            "calendar_aging": _safe_float(reward_cfg.get("coeff_soh_calendar", 1.0), 1.0),
            "ssr": _safe_float(reward_cfg.get("coeff_SSR", 0.0), 0.0),
            "wear_cost": _safe_float(reward_cfg.get("coeff_wear_cost", 0.0), 0.0),
        }
        active_coeffs = {k: float(v) for k, v in coeffs.items() if abs(float(v)) > 1e-12}

        coeff_to_metrics = {
            "economic": ["cost_economic"],
            "wear_cost": ["wear_cost"],
            "soe_violation": ["soe_violation"],
            "action_violation": ["action_violation"],
            "micro_throughput": ["micro_throughput", "reward_term_micro_throughput"],
            "soe_boundary": ["soe_boundary_penalty", "reward_term_soe_boundary"],
            "bad_logic": ["bad_logic_penalty"],
            "cyclic_aging": ["cyclic_aging"],
            "calendar_aging": ["calendar_aging"],
            "ssr": ["self_sufficiency_ratio"],
        }

        keys = ["reward", "base_reward"]
        for coeff_name in active_coeffs.keys():
            for metric_name in coeff_to_metrics.get(coeff_name, []):
                if metric_name not in keys:
                    keys.append(metric_name)

        if len(keys) == 2:
            keys.append("cost_economic")

    stats = {k: _compute_stats(buffers.get(k, [])) for k in keys}

    reward_values = buffers.get("reward", [])
    step_values = buffers.get("current_step", [])
    outliers = []
    if reward_values and step_values:
        pairs = list(zip(step_values, reward_values))
        pairs.sort(key=lambda x: x[1])
        outliers = pairs[:5]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(f"[{section}] episode={episode_idx} steps={episode_length}\n")
        handle.write(f"reward_mode={'pymgrid' if use_pymgrid_reward else 'custom'}\n")

        if not use_pymgrid_reward:
            coeffs_to_report = active_coeffs if active_coeffs else coeffs
            handle.write("reward_coeffs=" + json.dumps(coeffs_to_report, sort_keys=True) + "\n")

            def mean_abs(metric_name: str) -> float:
                return float(stats.get(metric_name, {}).get("mean_abs", 0.0))

            mean_abs_contrib = {
                "economic": mean_abs("cost_economic") * abs(coeffs_to_report.get("economic", 0.0)),
                "wear_cost": mean_abs("wear_cost") * abs(coeffs_to_report.get("wear_cost", 0.0)),
                "soe_violation": mean_abs("soe_violation") * abs(coeffs_to_report.get("soe_violation", 0.0)),
                "action_violation": mean_abs("action_violation") * abs(coeffs_to_report.get("action_violation", 0.0)),
                "micro_throughput": mean_abs("micro_throughput") * abs(coeffs_to_report.get("micro_throughput", 0.0)),
                "soe_boundary": mean_abs("soe_boundary_penalty") * abs(coeffs_to_report.get("soe_boundary", 0.0)),
                "bad_logic": mean_abs("bad_logic_penalty") * abs(coeffs_to_report.get("bad_logic", 0.0)),
                "cyclic_aging": mean_abs("cyclic_aging") * abs(coeffs_to_report.get("cyclic_aging", 0.0)),
                "calendar_aging": mean_abs("calendar_aging") * abs(coeffs_to_report.get("calendar_aging", 0.0)),
                "ssr": mean_abs("self_sufficiency_ratio") * abs(coeffs_to_report.get("ssr", 0.0)),
            }
            mean_abs_contrib = {k: float(v) for k, v in mean_abs_contrib.items() if v > 0.0}

            if mean_abs_contrib:
                if "economic" in mean_abs_contrib and mean_abs_contrib["economic"] > 1e-12:
                    base_scale = mean_abs_contrib["economic"]
                    ratios = {k: float(v / base_scale) for k, v in mean_abs_contrib.items()}
                    handle.write("mean_abs(|w*term|) ratios_vs_economic=" + json.dumps(ratios, sort_keys=True) + "\n")
                else:
                    first_term = next(iter(mean_abs_contrib.keys()))
                    base_scale = max(1e-12, mean_abs_contrib[first_term])
                    ratios = {k: float(v / base_scale) for k, v in mean_abs_contrib.items()}
                    handle.write("mean_abs(|w*term|) ratios_vs_first_active=" + json.dumps(ratios, sort_keys=True) + "\n")

        def write_line(name: str, s: dict) -> None:
            if s.get("count", 0) == 0:
                handle.write(f"{name}: empty\n")
                return
            handle.write(
                f"{name}: mean={s['mean']:.6g} std={s['std']:.6g} "
                f"p50={s['p50']:.6g} p90={s['p90']:.6g} p99={s['p99']:.6g} "
                f"min={s['min']:.6g} max={s['max']:.6g} mean_abs={s['mean_abs']:.6g}\n"
            )

        for k in keys:
            write_line(k, stats[k])

        overshoots = buffers.get("battery_overshoots", [])
        overshoots_end = int(overshoots[-1]) if overshoots else 0
        handle.write(f"battery_overshoots_end: {overshoots_end}\n")

        if outliers:
            handle.write("worst_reward_steps (current_step, reward): " + json.dumps(outliers) + "\n")


class RewardDiagnosticsCallback(BaseCallback):
    def __init__(self, log_path: Path, config: dict, section: str = "train"):
        super().__init__()
        self.log_path = Path(log_path)
        self.section = str(section)
        self.config = dict(config or {})
        self.reward_cfg = (self.config.get("rl", {}) or {}).get("reward", {}) or {}
        self._buffers = []
        self._episode_lengths = []
        self._episode_count = 0

    def _on_training_start(self) -> None:
        num_envs = int(getattr(self.training_env, "num_envs", 1))
        self._buffers = [_new_reward_diagnostics_buffers() for _ in range(num_envs)]
        self._episode_lengths = [0 for _ in range(num_envs)]
        self._episode_count = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "#" * 80 + "\n")
            handle.write(f"[{self.section}] start_time={datetime.now().isoformat(timespec='seconds')}\n")
            handle.write(f"dataset_path={self.config.get('rl', {}).get('dataset_path')}\n")
            handle.write("battery=" + json.dumps(self.config.get("battery", {}), sort_keys=True) + "\n")
            handle.write("ems=" + json.dumps(self.config.get("ems", {}), sort_keys=True) + "\n")
            handle.write("reward_cfg=" + json.dumps(self.reward_cfg, sort_keys=True) + "\n")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx in range(len(self._buffers)):
            reward = rewards[env_idx] if env_idx < len(rewards) else 0.0
            info = infos[env_idx] if env_idx < len(infos) else {}
            done = bool(dones[env_idx]) if env_idx < len(dones) else False

            self._episode_lengths[env_idx] += 1
            buf = self._buffers[env_idx]
            buf["reward"].append(_safe_float(reward))

            if isinstance(info, dict):
                buf["current_step"].append(int(info.get("current_step", -1)))
                for key in buf.keys():
                    if key in ("reward", "current_step"):
                        continue
                    if key in info:
                        buf[key].append(_safe_float(info.get(key)))

            if done:
                self._episode_count += 1
                _append_reward_diagnostics(
                    log_path=self.log_path,
                    section=self.section,
                    episode_idx=self._episode_count,
                    episode_length=self._episode_lengths[env_idx],
                    reward_cfg=self.reward_cfg,
                    buffers=buf,
                )
                self._buffers[env_idx] = _new_reward_diagnostics_buffers()
                self._episode_lengths[env_idx] = 0

        return True


def _run_episode(
    env: OfflineMicrogridRLEnv,
    model,
    deterministic: bool,
    random_start: bool,
    start_step: Optional[int],
    diagnostics_log_path: Optional[Path] = None,
    diagnostics_section: str = "eval",
    diagnostics_episode_idx: int = 0,
    action_log_path: Optional[Path] = None,
    reward_cfg: Optional[dict] = None,
    reasoning_buffer: Optional[EpisodeSeriesBuffer] = None,
    behavioural_buffer: Optional[EpisodeBehaviouralBuffer] = None,
) -> Tuple[float, int]:
    obs, _ = env.reset(random_start=random_start, start_step=start_step)
    total_reward = 0.0
    step_count = 0
    buffers = _new_reward_diagnostics_buffers()
    overshoots_end = 0
    action_log_rows = []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_raw = _safe_float(info.get("reward", reward)) if isinstance(info, dict) else _safe_float(reward)
        total_reward += reward_raw
        step_count += 1

        buffers["reward"].append(_safe_float(reward))
        if isinstance(info, dict):
            buffers["current_step"].append(int(info.get("current_step", -1)))
            for key in buffers.keys():
                if key in ("reward", "current_step"):
                    continue
                if key in info:
                    buffers[key].append(_safe_float(info.get(key)))
            requested = info.get("battery_action_requested")
            clipped = info.get("battery_action_clipped")
            actual = info.get("battery_action_actual")
            if requested is not None or clipped is not None or actual is not None:
                action_log_rows.append(
                    {
                        "step": int(info.get("current_step", step_count - 1)),
                        "requested_kwh": _safe_float(requested),
                        "clipped_kwh": _safe_float(clipped),
                        "actual_kwh": _safe_float(actual),
                        "was_clipped": int(bool(info.get("battery_action_was_clipped", False))),
                    }
                )
            if "battery_overshoots" in info:
                try:
                    overshoots_end = int(info.get("battery_overshoots") or 0)
                except (TypeError, ValueError):
                    overshoots_end = 0
            if reasoning_buffer is not None:
                reasoning_buffer.update(info)
            if behavioural_buffer is not None:
                behavioural_buffer.update(info)
        if terminated or truncated:
            break
    print(f"[episode] steps={step_count} reward={total_reward:.3f} overshoots={overshoots_end}")

    if diagnostics_log_path is not None and reward_cfg is not None:
        _append_reward_diagnostics(
            log_path=Path(diagnostics_log_path),
            section=diagnostics_section,
            episode_idx=int(diagnostics_episode_idx),
            episode_length=int(step_count),
            reward_cfg=dict(reward_cfg),
            buffers=buffers,
        )
    if action_log_path is not None and action_log_rows:
        _write_action_log_csv(Path(action_log_path), action_log_rows)
    return total_reward, overshoots_end

def _merge_running_stats_payloads(rms_payloads: list[dict], label: str) -> Optional[dict]:
    merged_mean = None
    merged_var = None
    merged_count = 0.0

    for idx, payload in enumerate(rms_payloads):
        if not isinstance(payload, dict):
            continue
        if payload.get("mean") is None or payload.get("var") is None:
            continue

        mean = np.asarray(payload.get("mean"), dtype=np.float64)
        var = np.asarray(payload.get("var"), dtype=np.float64)
        count = float(payload.get("count", 0.0))
        if count <= 0.0:
            continue

        if merged_mean is None:
            merged_mean = mean
            merged_var = var
            merged_count = count
            continue

        if mean.shape != merged_mean.shape or var.shape != merged_var.shape:
            raise ValueError(
                f"Inconsistent normalization shape for {label} at worker {idx}: "
                f"got mean={mean.shape}, var={var.shape}, expected mean={merged_mean.shape}, var={merged_var.shape}."
            )

        delta = mean - merged_mean
        tot_count = merged_count + count
        if tot_count <= 0.0:
            continue
        new_mean = merged_mean + delta * count / tot_count
        m_a = merged_var * merged_count
        m_b = var * count
        m_2 = m_a + m_b + np.square(delta) * merged_count * count / tot_count
        new_var = m_2 / tot_count

        merged_mean = new_mean
        merged_var = new_var
        merged_count = tot_count

    if merged_mean is None or merged_var is None or merged_count <= 0.0:
        return None

    return {
        "mean": merged_mean.tolist(),
        "var": merged_var.tolist(),
        "count": float(merged_count),
    }


def _merge_normalization_payloads(payloads: list[dict]) -> dict:
    merged = {"version": 1}

    obs_payloads = [
        payload.get("obs")
        for payload in payloads
        if isinstance(payload, dict) and payload.get("obs") is not None
    ]
    obs_dims = [
        int(payload.get("obs_dim"))
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("obs") is not None
        and payload.get("obs_dim") is not None
    ]
    if obs_dims and any(dim != obs_dims[0] for dim in obs_dims):
        raise ValueError(f"Inconsistent obs_dim across workers: {obs_dims}")
    merged_obs = _merge_running_stats_payloads(obs_payloads, label="obs")
    if merged_obs is not None:
        merged["obs"] = merged_obs
        if obs_dims:
            merged["obs_dim"] = int(obs_dims[0])

    reward_payloads = [
        payload.get("reward")
        for payload in payloads
        if isinstance(payload, dict) and payload.get("reward") is not None
    ]
    merged_reward = _merge_running_stats_payloads(reward_payloads, label="reward")
    if merged_reward is not None:
        merged["reward"] = merged_reward

    return merged


def _save_vec_env_normalization_state(vec_env, stats_path: Path) -> None:
    stats_path = Path(stats_path)
    payloads = []

    if hasattr(vec_env, "env_method"):
        try:
            payloads = vec_env.env_method("export_normalization_state")
        except Exception as exc:
            print(f"[WARN] Failed to collect normalization states via env_method: {exc}. Falling back to worker 0.")
            payloads = []

    if not payloads and hasattr(vec_env, "envs"):
        for env in vec_env.envs:
            if hasattr(env, "export_normalization_state"):
                payloads.append(env.export_normalization_state())

    if not payloads:
        if hasattr(vec_env, "env_method"):
            vec_env.env_method("save_normalization_state", str(stats_path), indices=0)
        else:
            vec_env.envs[0].save_normalization_state(stats_path)
        print("[WARN] Saved normalization statistics from worker 0 only (aggregation unavailable).")
        return

    merged_payload = _merge_normalization_payloads(payloads)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_payload, handle, indent=2)

    num_workers = len(payloads)
    if num_workers > 1:
        print(f"[INFO] Saved aggregated normalization statistics from {num_workers} workers.")

def train(config_path: str, config: dict, output_dir: Path, device: str, random_start: bool,
          start_step_default: Optional[int]):
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _start_tensorboard(tensorboard_dir, config.get("rl", {}) or {})

    agent = EMS_RL_Agent(config=config, device=device, tensorboard_log=str(tensorboard_dir))
    vec_env = agent.make_vec_env(
        config_path=config_path,
        random_start=random_start,
        start_step_default=start_step_default,
    )

    try:
        model = agent.build_model(vec_env)

        num_envs = int(getattr(vec_env, "num_envs", 1))
        print(f"[INFO] RL vectorized env: type={type(vec_env).__name__} num_envs={num_envs}")

        norm_cfg = (config.get("rl", {}) or {}).get("normalization", {}) or {}
        norm_obs_enabled = bool((norm_cfg.get("observations", {}) or {}).get("enabled", False))
        norm_reward_enabled = bool((norm_cfg.get("reward", {}) or {}).get("enabled", False))
        if num_envs > 1 and (norm_obs_enabled or norm_reward_enabled):
            print(
                "[INFO] Observation/reward normalization is tracked per worker and merged at save-time "
                "into a single normalization_state.json."
            )

        total_timesteps = int(config["rl"]["episode_steps"]) * int(config["rl"]["train_num_episodes"])
        checkpoint_freq = int(config["rl"].get("checkpoint_freq", 10000))
        checkpoint_save_freq = max(checkpoint_freq // max(num_envs, 1), 1)
        if num_envs > 1 and checkpoint_save_freq != checkpoint_freq:
            print(
                f"[INFO] Checkpoint callback adjusted for vec env: requested_timesteps={checkpoint_freq} "
                f"callback_calls={checkpoint_save_freq}"
            )
        checkpoint_cb = CheckpointCallback(save_freq=checkpoint_save_freq, save_path=str(checkpoint_dir), name_prefix="ppo")
        episode_cb = EpisodePrintCallback()
        diagnostics_cb = RewardDiagnosticsCallback(
            log_path=output_dir / "reward_diagnostics.txt",
            config=config,
            section="train",
        )
        tb_callback = TensorboardTrainCallback(
            log_dir=tensorboard_dir,
            step_downsample=int(config["rl"].get("tb_step_downsample", 1)),
            ma_window=int(config["rl"].get("tb_ma_window", 20)),
            include_price=bool(config["rl"].get("include_price", False)),
            reasoning_plot_every=int(config["rl"].get("tb_reasoning_plot_every", 10)),
            price_bands=(config.get("ems", {}) or {}).get("price_bands"),
            enable_figures=bool(config["rl"].get("tb_enable_figures", True)),
            enable_histograms=bool(config["rl"].get("tb_enable_histograms", True)),
        )
        callback = CallbackList([checkpoint_cb, episode_cb, diagnostics_cb, tb_callback])

        model.learn(total_timesteps=total_timesteps, callback=callback)

        model_path = output_dir / "model_final"
        model.save(str(model_path))
        save_stats = bool(norm_cfg.get("save_stats", False))
        stats_path = norm_cfg.get("stats_path")
        if save_stats and stats_path and (norm_obs_enabled or norm_reward_enabled):
            _save_vec_env_normalization_state(vec_env, Path(stats_path))
        return model_path
    finally:
        try:
            vec_env.close()
        except Exception as exc:
            print(f"[WARN] Failed to close vec_env cleanly: {exc}")

def evaluate(config_path: str, config: dict, output_dir: Path, model_path: str, device: str,
             deterministic: bool, random_start: bool, plot: bool, start_step_default: Optional[int]):
    env = OfflineMicrogridRLEnv(
        config_path=config_path,
        config=config,
        random_start=random_start,
        start_step_default=start_step_default,
    )
    env.set_normalization_update(False)
    agent = EMS_RL_Agent(config=config, device=device)
    model = agent.load(model_path, env=env)

    rewards = []
    diagnostics_path = output_dir / "reward_diagnostics.txt"
    reward_cfg = (config.get("rl", {}) or {}).get("reward", {}) or {}
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(
        log_dir=tensorboard_dir,
        step_downsample=int(config["rl"].get("tb_step_downsample", 1)),
        include_price=bool(config["rl"].get("include_price", False)),
        price_bands=(config.get("ems", {}) or {}).get("price_bands"),
        enable_figures=bool(config["rl"].get("tb_enable_figures", True)),
        enable_histograms=bool(config["rl"].get("tb_enable_histograms", True)),
    )
    return_ma = MovingAverage(int(config["rl"].get("tb_ma_window", 20)))
    reasoning_every = int(config["rl"].get("tb_reasoning_plot_every", 10))
    for _ in range(int(config["rl"]["eval_episodes"])):
        reasoning_buffer = EpisodeSeriesBuffer() if reasoning_every > 0 else None
        behavioural_buffer = EpisodeBehaviouralBuffer()
        episode_reward, episode_overshoots = _run_episode(
            env,
            model,
            deterministic,
            random_start,
            start_step=None,
            diagnostics_log_path=diagnostics_path,
            diagnostics_section="eval",
            diagnostics_episode_idx=len(rewards) + 1,
            reward_cfg=reward_cfg,
            reasoning_buffer=reasoning_buffer,
            behavioural_buffer=behavioural_buffer,
        )
        rewards.append(episode_reward)
        tb_logger.log_episode(
            {
                "episode_return": episode_reward,
                "episode_return_ma": return_ma.update(episode_reward),
                "battery_overshoots_end": int(episode_overshoots),
            },
            episode_idx=len(rewards),
            prefix="eval",
        )
        tb_logger.log_behavioural(
            behavioural_buffer,
            episode_idx=len(rewards),
            prefix="eval",
        )
        if reasoning_buffer is not None and len(rewards) % reasoning_every == 0:
            tb_logger.log_reasoning_figure(
                reasoning_buffer,
                episode_idx=len(rewards),
                prefix="eval",
            )
    tb_logger.flush()
    tb_logger.close()

    summary = {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "episode_rewards": rewards,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if plot:
        microgrid_df, _ = env.get_simulation_log()
        plot_results(microgrid_df, str(output_dir / "eval_results"), config["ems"]["timezone"])
        battery_module = env.battery_module
        bms_manager = getattr(battery_module, "battery_bms_manager", None)
        transition_model = getattr(battery_module, "battery_transition_model", None)
        history_owner = bms_manager if bms_manager is not None else transition_model
        transition_dir = output_dir / "battery_simulation_data"
        transition_dir.mkdir(parents=True, exist_ok=True)
        transition_base = transition_dir / f"transitions_{env.simulator.battery_chemistry}.png"
        if history_owner is not None and hasattr(history_owner, "plot_transition_history"):
            history_owner.plot_transition_history(save_path=str(transition_base), show=True)
        if history_owner is not None and hasattr(history_owner, "save_transition_history"):
            history_owner.save_transition_history(
                history_path=str(transition_dir / f"transitions_{env.simulator.battery_chemistry}.json")
            )

    return summary


def test_or_rollout(config_path: str, config: dict, output_dir: Path, model_path: str, device: str,
                    deterministic: bool, start_step: Optional[int], plot: bool, mode: str):
    env = OfflineMicrogridRLEnv(config_path=config_path, config=config, random_start=False)
    env.set_normalization_update(False)
    agent = EMS_RL_Agent(config=config, device=device)
    model = agent.load(model_path, env=env)

    transition_dir = output_dir / "battery_simulation_data"
    transition_dir.mkdir(parents=True, exist_ok=True)
    action_log_path = transition_dir / "battery_action_step_log.csv"

    tensorboard_dir = output_dir / "tensorboard"
    reasoning_every = int(config["rl"].get("tb_reasoning_plot_every", 10))
    tb_logger = None
    reasoning_buffer = None
    behavioural_buffer = None
    if reasoning_every > 0:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(
            log_dir=tensorboard_dir,
            step_downsample=int(config["rl"].get("tb_step_downsample", 1)),
            include_price=bool(config["rl"].get("include_price", False)),
            price_bands=(config.get("ems", {}) or {}).get("price_bands"),
            enable_figures=bool(config["rl"].get("tb_enable_figures", True)),
            enable_histograms=bool(config["rl"].get("tb_enable_histograms", True)),
        )
        reasoning_buffer = EpisodeSeriesBuffer()
        behavioural_buffer = EpisodeBehaviouralBuffer()

    episode_reward, episode_overshoots = _run_episode(
        env,
        model,
        deterministic=deterministic,
        random_start=False,
        start_step=start_step,
        diagnostics_episode_idx=1,
        action_log_path=action_log_path,
        reasoning_buffer=reasoning_buffer,
        behavioural_buffer=behavioural_buffer,
    )

    if reasoning_buffer is not None and tb_logger is not None and reasoning_every > 0:
        if 1 % reasoning_every == 0:
            prefix = "rollout" if mode == "rollout" else "test"
            tb_logger.log_reasoning_figure(
                reasoning_buffer,
                episode_idx=1,
                prefix=prefix,
            )
        if behavioural_buffer is not None:
            tb_logger.log_behavioural(
                behavioural_buffer,
                episode_idx=1,
                prefix=prefix,
            )
        tb_logger.log_episode(
            {
                "episode_return": episode_reward,
                "battery_overshoots_end": int(episode_overshoots),
            },
            episode_idx=1,
            prefix=prefix,
        )
        tb_logger.flush()
        tb_logger.close()

    microgrid_df, log = env.get_simulation_log()
    microgrid_df.to_csv(output_dir / "microgrid_log.csv", index=True)
    log.to_csv(output_dir / "microgrid_log_raw.csv", index=True)

    summary_buffer = StringIO()
    with redirect_stdout(summary_buffer):
        print_final_report(
            microgrid_df,
            control_strategy="RL_AGENT",
            battery_chemistry=env.simulator.battery_chemistry,
            soh_degradation_enabled=not bool(getattr(env.simulator, "disable_soh_degradation", False)),
        )
    summary_text = summary_buffer.getvalue()
    print(summary_text, end="")
    (output_dir / "final_report.txt").write_text(summary_text, encoding="utf-8")

    _plot_rl_request_vs_max(log, config, transition_dir, output_dir.name)
    _plot_battery_action_request_vs_actual(
        action_log_path,
        transition_dir / "action_req_vs_actual.png",
        output_dir.name,
    )

    battery_module = env.battery_module
    bms_manager = getattr(battery_module, "battery_bms_manager", None)
    transition_model = getattr(battery_module, "battery_transition_model", None)
    history_owner = bms_manager if bms_manager is not None else transition_model
    transition_base = transition_dir / f"transitions_{env.simulator.battery_chemistry}.png"
    if history_owner is not None and hasattr(history_owner, "plot_transition_history"):
        history_owner.plot_transition_history(save_path=str(transition_base), show=bool(plot))
    if history_owner is not None and hasattr(history_owner, "save_transition_history"):
        history_owner.save_transition_history(
            history_path=str(transition_dir / f"transitions_{env.simulator.battery_chemistry}.json")
        )

    if plot:
        plot_results(microgrid_df, str(output_dir / "rl_results"), config["ems"]["timezone"])


def main():
    parser = argparse.ArgumentParser(description="Offline RL agent for microgrid EMS (SB3 PPO/SAC).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "controllers" / "rl" / "opsd" / "params_RL_agent_OPSD_1_DAY.yml"),
        help="Path to RL config YAML.",
    )
    parser.add_argument("--mode", type=str, choices=["train", "eval", "test", "rollout"], default="train")
    parser.add_argument("--model-path", type=str, default="", help="Path to trained model.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training episodes.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override steps per episode.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override training steps per episode.")
    parser.add_argument("--test-steps", type=int, default=None, help="Override test/rollout steps per episode.")
    parser.add_argument("--output-dir", type=str, default="", help="Override base output directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    parser.add_argument("--algorithm", type=str, default=None, help="Override RL algorithm (ppo or sac).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--no-random-start", action="store_true", help="Disable random episode starts.")
    parser.add_argument("--eval-start-step", type=int, default=None, help="Override eval/test start step.")
    parser.add_argument("--train-start-step", type=int, default=None, help="Override training start step.")
    parser.add_argument("--test-start-step", type=int, default=None, help="Override test/rollout start step.")
    parser.add_argument("--plot", action="store_true", help="Save plots for evaluation/test.")
    parser.add_argument("--stochastic", action="store_true", help="Disable deterministic action selection.")
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default=None,
        help="Override train dataset path (CSV).",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=None,
        help="Path to eval dataset (required for --mode eval).",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default=None,
        help="Path to test dataset (required for --mode test/rollout).",
    )

    args = parser.parse_args()

    config = load_rl_agent_config(args.config)
    _select_dataset_for_mode(
        config,
        args.mode,
        train_dataset_path=args.train_dataset_path,
        eval_dataset_path=args.eval_dataset_path,
        test_dataset_path=args.test_dataset_path,
    )
    _select_episode_steps(config, args.mode)
    if args.episodes is not None:
        config["rl"]["train_num_episodes"] = int(args.episodes)
    if args.run_name is not None:
        config["rl"]["run_name"] = args.run_name
    if args.algorithm is not None:
        algorithm = str(args.algorithm).strip().upper()
        if algorithm not in {"PPO", "SAC"}:
            raise ValueError(f"Unsupported --algorithm {args.algorithm!r}. Supported values: ppo, sac.")
        config.setdefault("rl", {})["algorithm"] = algorithm
    if args.eval_start_step is not None:
        config["rl"]["eval_start_step"] = int(args.eval_start_step)
    if args.train_start_step is not None:
        config["rl"]["train_start_step"] = int(args.train_start_step)

    _apply_mode_step_split(config, args.mode)

    deterministic = not args.stochastic
    global_steps_override = int(args.steps) if args.steps is not None else None
    max_steps = _get_max_steps(args.config, config)

    base_output = Path(args.output_dir) if args.output_dir else Path(config["rl"]["output_dir"])
    base_output = _resolve_output_base(base_output)
    run_name = str(config["rl"].get("run_name", "ppo_offline"))
    output_dir = _prepare_output_dir(base_output, args.mode, run_name)

    try:
        shutil.copy2(args.config, output_dir / Path(args.config).name)
    except OSError:
        pass

    rl_cfg = config.setdefault("rl", {})
    if args.mode == "train":
        if not rl_cfg.get("observation_bounds_path"):
            rl_cfg["observation_bounds_path"] = str(output_dir / "obs_bounds.json")
        if rl_cfg.get("save_observation_bounds") is None:
            rl_cfg["save_observation_bounds"] = True
    else:
        if not rl_cfg.get("observation_bounds_path"):
            bounds_path = _resolve_obs_bounds_path(args.model_path)
            if bounds_path is not None:
                rl_cfg["observation_bounds_path"] = str(bounds_path)

    norm_cfg = rl_cfg.setdefault("normalization", {})
    norm_cfg.setdefault("observations", {})
    norm_cfg.setdefault("actions", {})
    norm_cfg.setdefault("reward", {})
    norm_obs_enabled = bool((norm_cfg.get("observations", {}) or {}).get("enabled", False))
    norm_reward_enabled = bool((norm_cfg.get("reward", {}) or {}).get("enabled", False))
    norm_enabled = norm_obs_enabled or norm_reward_enabled

    if args.mode == "train":
        if norm_enabled and not norm_cfg.get("stats_path"):
            norm_cfg["stats_path"] = str(output_dir / "normalization_state.json")
        if norm_enabled and norm_cfg.get("save_stats") is None:
            norm_cfg["save_stats"] = True
    else:
        if norm_enabled and not norm_cfg.get("stats_path"):
            stats_path = _resolve_normalization_stats_path(args.model_path)
            if stats_path is not None:
                norm_cfg["stats_path"] = str(stats_path)
        if norm_enabled and not norm_cfg.get("stats_path"):
            raise FileNotFoundError(
                "Normalization stats file not found. Provide rl.normalization.stats_path or "
                "ensure normalization_state.json exists alongside the model."
            )

    if args.mode == "train":
        if args.train_steps is not None:
            _set_episode_steps(config, "train", int(args.train_steps))
        elif global_steps_override is not None:
            _set_episode_steps(config, "train", int(global_steps_override))
        else:
            _select_episode_steps(config, "train")
        random_start = bool(config["rl"].get("train_random_start", True))
        if args.no_random_start:
            random_start = False
        if args.train_start_step is not None:
            random_start = False
        train_start_step = int(config["rl"].get("train_start_step", config["rl"].get("eval_start_step", 0)))
        if random_start:
            if config["rl"]["episode_steps"] > max_steps:
                print(
                    f"[WARN] train: episode_steps {config['rl']['episode_steps']} exceeds "
                    f"max_steps {max_steps}. Using episode_steps={max_steps} instead."
                )
                config["rl"]["episode_steps"] = max_steps
        else:
            config["rl"]["episode_steps"] = _resolve_episode_steps(
                mode="train",
                desired_steps=int(config["rl"]["episode_steps"]),
                start_step=train_start_step,
                max_steps=max_steps,
            )
        model_path = train(
            config_path=args.config,
            config=config,
            output_dir=output_dir,
            device=args.device,
            random_start=random_start,
            start_step_default=train_start_step,
        )
        print(f"Training completed. Model saved to {model_path}")
        return

    if not args.model_path:
        raise ValueError("Model path is required for eval/test/rollout.")

    random_start = bool(config["rl"].get("eval_random_start", False))
    if args.no_random_start:
        random_start = False

    if args.mode == "eval":
        if global_steps_override is not None:
            _set_episode_steps(config, "eval", int(global_steps_override))
        else:
            _select_episode_steps(config, "eval")
        eval_start_step = int(config["rl"].get("eval_start_step", 0))
        if not random_start:
            config["rl"]["episode_steps"] = _resolve_episode_steps(
                mode="eval",
                desired_steps=int(config["rl"]["episode_steps"]),
                start_step=eval_start_step,
                max_steps=max_steps,
            )
        summary = evaluate(
            config_path=args.config,
            config=config,
            output_dir=output_dir,
            model_path=args.model_path,
            device=args.device,
            deterministic=deterministic,
            random_start=random_start,
            plot=args.plot,
            start_step_default=eval_start_step,
        )
        print(f"Evaluation mean reward: {summary['mean_reward']:.3f}")
        return

    if args.test_steps is not None:
        _set_episode_steps(config, "eval", int(args.test_steps))
    elif global_steps_override is not None:
        _set_episode_steps(config, "eval", int(global_steps_override))
    else:
        _select_episode_steps(config, "eval")

    start_step = int(config["rl"].get("eval_start_step", 0))
    if args.test_start_step is not None:
        start_step = int(args.test_start_step)
    elif args.eval_start_step is not None:
        start_step = int(args.eval_start_step)
    config["rl"]["episode_steps"] = _resolve_episode_steps(
        mode=args.mode,
        desired_steps=int(config["rl"]["episode_steps"]),
        start_step=start_step,
        max_steps=max_steps,
    )

    test_or_rollout(
        config_path=args.config,
        config=config,
        output_dir=output_dir,
        model_path=args.model_path,
        device=args.device,
        deterministic=deterministic,
        start_step=start_step,
        plot=args.plot,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

