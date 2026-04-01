import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_AGENT.EMS_RL_agent import EMS_RL_Agent, OfflineMicrogridRLEnv, load_rl_agent_config
from RL_AGENT.tb_logger import TensorboardTrainCallback


def _resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _parse_policy_net(value: str) -> list[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Training PPO a partire da una policy BC.")
    parser.add_argument(
        "--rl-config",
        required=True,
        help="Path al file di config RL (definisce reward, observation/action space).",
    )
    parser.add_argument("--bc-policy", required=True, help="Path al file bc_policy.pt.")
    parser.add_argument("--output-dir", default="BC_MPC/outputs/ppo_trained")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nome run (default: ppo_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=(
            "Override timesteps totali (default: train_num_episodes * train_episode_steps dal config RL)."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override del dataset path usato nel training (CSV).",
    )
    parser.add_argument(
        "--policy-net",
        default="",
        help=(
            "Override architettura MLP (es. 64,64). Se vuoto usa rl.ppo.policy_net. "
            "Deve essere compatibile con la policy BC salvata."
        ),
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Salva un checkpoint ogni N timesteps (override di rl.checkpoint_freq).",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rl_config_path = _resolve_path(args.rl_config) or args.rl_config
    bc_policy_path = _resolve_path(args.bc_policy) or args.bc_policy
    output_root = _resolve_path(args.output_dir) or args.output_dir
    dataset_path = _resolve_path(args.dataset_path) if args.dataset_path else None

    config = load_rl_agent_config(rl_config_path)
    rl_cfg = config.setdefault("rl", {})
    if dataset_path:
        rl_cfg["dataset_path"] = dataset_path
    elif rl_cfg.get("dataset_path") is None and rl_cfg.get("dataset_path_train"):
        rl_cfg["dataset_path"] = rl_cfg["dataset_path_train"]
    ppo_cfg = rl_cfg.setdefault("ppo", {})
    policy_net = _parse_policy_net(args.policy_net)
    if policy_net:
        ppo_cfg["policy_net"] = policy_net

    if args.timesteps is not None:
        total_timesteps = int(args.timesteps)
    else:
        total_timesteps = int(rl_cfg.get("train_num_episodes", 0)) * int(
            rl_cfg.get("train_episode_steps", 0)
        )
    if total_timesteps <= 0:
        total_timesteps = 200_000

    run_name = args.run_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    agent = EMS_RL_Agent(config=config, device="cpu", tensorboard_log=str(tensorboard_dir))
    env = OfflineMicrogridRLEnv(
        config_path=rl_config_path,
        config=config,
        random_start=bool(rl_cfg.get("train_random_start", True)),
        seed=args.seed,
        start_step_default=int(rl_cfg.get("train_start_step", 0)),
    )
    model = agent.build_model(env)

    bc_state = torch.load(bc_policy_path, map_location="cpu")
    missing, unexpected = model.policy.load_state_dict(bc_state, strict=False)

    checkpoint_freq = (
        int(args.checkpoint_freq)
        if args.checkpoint_freq is not None
        else int(rl_cfg.get("checkpoint_freq", 0) or 0)
    )
    checkpoint_cb = None
    if checkpoint_freq > 0:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
        )

    tb_callback = TensorboardTrainCallback(
        log_dir=tensorboard_dir,
        step_downsample=int(rl_cfg.get("tb_step_downsample", 1)),
        ma_window=int(rl_cfg.get("tb_ma_window", 20)),
        include_price=bool(rl_cfg.get("include_price", False)),
        reasoning_plot_every=int(rl_cfg.get("tb_reasoning_plot_every", 10)),
        price_bands=(config.get("ems", {}) or {}).get("price_bands"),
    )

    try:
        shutil.copy2(rl_config_path, output_dir / Path(rl_config_path).name)
    except OSError:
        pass
    bc_meta = Path(bc_policy_path).parent / "bc_metadata.json"
    if bc_meta.exists():
        try:
            shutil.copy2(bc_meta, output_dir / bc_meta.name)
        except OSError:
            pass

    metadata = {
        "rl_config": rl_config_path,
        "bc_policy": bc_policy_path,
        "timesteps": total_timesteps,
        "dataset_path": rl_cfg.get("dataset_path"),
        "policy_net": ppo_cfg.get("policy_net"),
        "checkpoint_freq": checkpoint_freq,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    with (output_dir / "ppo_trained_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    callbacks = [tb_callback]
    if checkpoint_cb is not None:
        callbacks.append(checkpoint_cb)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
    )
    model.save(str(output_dir / "ppo_trained.zip"))

    norm_cfg = rl_cfg.setdefault("normalization", {})
    obs_norm_enabled = bool((norm_cfg.get("observations") or {}).get("enabled", False))
    reward_norm_enabled = bool((norm_cfg.get("reward") or {}).get("enabled", False))
    norm_enabled = obs_norm_enabled or reward_norm_enabled
    save_stats = norm_cfg.get("save_stats")
    if save_stats is None:
        save_stats = True
    stats_path = norm_cfg.get("stats_path")
    if norm_enabled and save_stats:
        if not stats_path:
            stats_path = str(output_dir / "normalization_state.json")
        stats_path_obj = Path(stats_path)
        env.save_normalization_state(stats_path_obj)
        metadata["normalization_stats_path"] = str(stats_path_obj)
        if checkpoint_freq > 0:
            checkpoint_stats = output_dir / "checkpoints" / "normalization_state.json"
            env.save_normalization_state(checkpoint_stats)
            metadata["normalization_stats_path_checkpoints"] = str(checkpoint_stats)
        with (output_dir / "ppo_trained_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print(f"Modello PPO trained salvato in: {output_dir / 'ppo_trained.zip'}")


if __name__ == "__main__":
    main()
