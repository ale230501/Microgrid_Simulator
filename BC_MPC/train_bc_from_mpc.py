import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.algorithms import bc
from imitation.data import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_AGENT.EMS_RL_agent import OfflineMicrogridRLEnv, load_rl_agent_config


def _resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _load_transitions(dataset_path: Path) -> types.Transitions:
    data = np.load(dataset_path)
    obs = data["obs"]
    acts = data["acts"]
    next_obs = data["next_obs"]
    dones = data["dones"].astype(bool)
    infos = np.array([{} for _ in range(len(obs))], dtype=object)
    return types.Transitions(obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones)


def _parse_policy_net(value: str) -> list[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Allena un agente BC usando un dataset MPC.")
    parser.add_argument(
        "--rl-config",
        required=True,
        help="Path al file di config RL (per definire observation/action space).",
    )
    parser.add_argument("--dataset", required=True, help="Path al dataset .npz generato.")
    parser.add_argument("--output-dir", default="BC_MPC/outputs/bc_policies")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nome run (default: bc_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--policy-net",
        default="",
        help="Override architettura MLP (es. 64,64). Se vuoto usa rl.ppo.policy_net.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rl_config_path = _resolve_path(args.rl_config) or args.rl_config
    dataset_path = _resolve_path(args.dataset) or args.dataset
    output_root = _resolve_path(args.output_dir) or args.output_dir

    config = load_rl_agent_config(rl_config_path)
    env = OfflineMicrogridRLEnv(
        config_path=rl_config_path,
        config=config,
        random_start=False,
        seed=args.seed,
        start_step_default=0,
    )

    transitions = _load_transitions(Path(dataset_path))

    rng = np.random.default_rng(args.seed)
    ppo_cfg = (config.get("rl", {}) or {}).get("ppo", {}) or {}
    policy_net = _parse_policy_net(args.policy_net)
    policy_net_source = "args"
    if not policy_net:
        policy_net = list(ppo_cfg.get("policy_net") or [])
        policy_net_source = "config" if policy_net else "default"

    policy = None
    if policy_net:
        policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            activation_fn=nn.ReLU,
            net_arch=dict(pi=policy_net, vf=policy_net),
        )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        policy=policy,
        demonstrations=transitions,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.learning_rate},
    )

    bc_trainer.train(n_epochs=args.epochs)

    run_name = args.run_name or f"bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "bc_policy.pt"
    torch.save(bc_trainer.policy.state_dict(), model_path)

    try:
        shutil.copy2(rl_config_path, output_dir / Path(rl_config_path).name)
    except OSError:
        pass
    dataset_meta = Path(dataset_path).with_suffix(".json")
    if dataset_meta.exists():
        try:
            shutil.copy2(dataset_meta, output_dir / dataset_meta.name)
        except OSError:
            pass

    metadata = {
        "rl_config": rl_config_path,
        "dataset": dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "policy_net": policy_net or [32, 32],
        "policy_net_source": policy_net_source,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "action_normalized": bool(env._action_norm_enabled),
        "obs_normalized": bool(env._obs_norm_enabled),
    }
    with (output_dir / "bc_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Policy salvata in: {model_path}")


if __name__ == "__main__":
    main()
