import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_AGENT.EMS_RL_agent import OfflineMicrogridRLEnv, load_rl_agent_config
from MODEL_PREDICTIVE.mpc_MILP import MPCController


def _resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _resolve_dataset_path(config: Dict[str, Any], dataset_path_arg: str | None) -> str:
    dataset_path = dataset_path_arg or config["rl"].get("dataset_path")
    resolved = _resolve_path(dataset_path)
    if not resolved:
        raise ValueError("Missing dataset path for BC/MPC generation.")
    config["rl"]["dataset_path"] = resolved
    return resolved


def _safe_steps(env: OfflineMicrogridRLEnv, mpc: MPCController) -> int:
    max_steps = min(env.max_steps, len(mpc.load_series) - mpc.horizon)
    if max_steps <= 0:
        raise ValueError(
            "Dataset troppo corto rispetto all'orizzonte MPC. "
            "Riduci l'orizzonte o usa un dataset più lungo."
        )
    mpc.steps = max_steps
    env.episode_length = max_steps
    return max_steps


def collect_dataset(
    rl_config_path: str,
    mpc_config_path: str,
    dataset_path: str,
    output_dir: Path,
    max_steps: int | None = None,
) -> Path:
    rl_config_path = _resolve_path(rl_config_path) or rl_config_path
    mpc_config_path = _resolve_path(mpc_config_path) or mpc_config_path
    output_dir = Path(_resolve_path(str(output_dir)) or output_dir)

    config = load_rl_agent_config(rl_config_path)
    dataset_path = _resolve_dataset_path(config, dataset_path)

    env = OfflineMicrogridRLEnv(
        config_path=rl_config_path,
        config=config,
        random_start=False,
        seed=int(config["rl"].get("seed") or 0),
        start_step_default=0,
    )

    mpc = MPCController(
        env.microgrid,
        config_path=mpc_config_path,
        data_path=dataset_path,
    )

    max_len = _safe_steps(env, mpc)
    if max_steps is not None:
        max_len = min(max_len, int(max_steps))
        env.episode_length = max_len
        mpc.steps = max_len

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    done_list: List[bool] = []

    obs, _ = env.reset(random_start=False, start_step=0)

    for _ in range(max_len):
        mpc_action = mpc.get_action(verbose=0)
        battery_action = float(mpc_action["battery"])
        if env._action_norm_enabled:
            battery_action = env.normalize_action(battery_action)
        action = np.array([battery_action], dtype=np.float32)

        obs_list.append(obs.astype(np.float32))
        act_list.append(action)

        obs, _, done, _, _ = env.step(action)
        next_obs_list.append(obs.astype(np.float32))
        done_list.append(bool(done))
        if done:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mpc_bc_dataset.npz"
    np.savez_compressed(
        output_path,
        obs=np.stack(obs_list),
        acts=np.stack(act_list),
        next_obs=np.stack(next_obs_list),
        dones=np.array(done_list, dtype=bool),
    )

    csv_path = output_dir / "mpc_bc_dataset.csv"
    obs_arr = np.stack(obs_list)
    act_arr = np.stack(act_list)
    next_obs_arr = np.stack(next_obs_list)
    done_arr = np.array(done_list, dtype=bool)
    _write_dataset_csv(csv_path, obs_arr, act_arr, next_obs_arr, done_arr)

    metadata = {
        "rl_config": rl_config_path,
        "mpc_config": mpc_config_path,
        "dataset_path": dataset_path,
        "num_samples": int(len(obs_list)),
        "action_normalized": bool(env._action_norm_enabled),
        "obs_normalized": bool(env._obs_norm_enabled),
    }
    with (output_dir / "mpc_bc_dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return output_path


def _write_dataset_csv(
    csv_path: Path,
    obs: np.ndarray,
    acts: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
) -> None:
    obs_dim = obs.shape[1] if obs.ndim > 1 else 1
    act_dim = acts.shape[1] if acts.ndim > 1 else 1
    next_obs_dim = next_obs.shape[1] if next_obs.ndim > 1 else 1

    header = (
        [f"obs_{i}" for i in range(obs_dim)]
        + [f"act_{i}" for i in range(act_dim)]
        + [f"next_obs_{i}" for i in range(next_obs_dim)]
        + ["done"]
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(len(obs)):
            obs_row = obs[i].tolist() if obs_dim > 1 else [float(obs[i])]
            act_row = acts[i].tolist() if act_dim > 1 else [float(acts[i])]
            next_obs_row = next_obs[i].tolist() if next_obs_dim > 1 else [float(next_obs[i])]
            row = obs_row + act_row + next_obs_row + [int(dones[i])]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera dataset BC usando l'MPC.")
    parser.add_argument(
        "--rl-config",
        required=True,
        help="Path al file di config RL (es. configs/controllers/rl/opsd/params_RL_agent_OPSD_1_DAY.yml)",
    )
    parser.add_argument(
        "--mpc-config",
        required=True,
        help="Path al file di config MPC (es. configs/controllers/mpc/params_OPSD.yml)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override dataset path (se diverso da rl.dataset_path).",
    )
    parser.add_argument(
        "--output-dir",
        default="BC_MPC/outputs",
        help="Directory di output.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limite massimo di step da registrare.",
    )

    args = parser.parse_args()
    output_path = collect_dataset(
        rl_config_path=args.rl_config,
        mpc_config_path=args.mpc_config,
        dataset_path=args.dataset_path,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    print(f"Dataset salvato in: {output_path}")


if __name__ == "__main__":
    main()
