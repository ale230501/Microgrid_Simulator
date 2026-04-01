import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imitation.policies.base import FeedForward32Policy
from RL_AGENT.EMS_RL_agent import OfflineMicrogridRLEnv, load_rl_agent_config


def _resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _select_dataset(config: dict, dataset_mode: str, dataset_path: str | None) -> str:
    rl_cfg = config.setdefault("rl", {})
    if dataset_path:
        resolved = _resolve_path(dataset_path)
        if not resolved:
            raise ValueError("Missing dataset path override.")
        rl_cfg["dataset_path"] = resolved
        return resolved
    key = f"dataset_path_{dataset_mode}"
    selected = rl_cfg.get(key) or rl_cfg.get("dataset_path")
    if not selected:
        raise ValueError(f"Missing dataset path for mode '{dataset_mode}'.")
    resolved = _resolve_path(selected)
    if not resolved:
        raise ValueError("Missing dataset path in config.")
    rl_cfg["dataset_path"] = resolved
    return resolved


def _parse_policy_net(value: str) -> list[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _run_episode(
    env: OfflineMicrogridRLEnv,
    policy: FeedForward32Policy,
    deterministic: bool,
    start_step: int | None,
    max_steps: int | None,
) -> tuple[float, int]:
    obs, _ = env.reset(random_start=False, start_step=start_step)
    total_reward = 0.0
    steps = 0
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(info.get("reward", reward))
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
        if terminated or truncated:
            break
    return total_reward, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta una policy BC su un episodio offline.")
    parser.add_argument("--rl-config", required=True, help="Path al file di config RL.")
    parser.add_argument("--bc-policy", required=True, help="Path al file bc_policy.pt.")
    parser.add_argument(
        "--dataset-mode",
        default="eval",
        choices=("train", "eval", "test"),
        help="Seleziona il dataset dal config RL.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override diretto del dataset (CSV).",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Azioni non deterministiche.")
    parser.add_argument(
        "--policy-net",
        default="",
        help="Override architettura MLP (es. 64,64). Se vuoto usa rl.ppo.policy_net.",
    )
    parser.add_argument(
        "--no-normalization-update",
        action="store_true",
        help="Disabilita l'aggiornamento delle statistiche di normalizzazione.",
    )
    parser.add_argument("--output-dir", default=None, help="Cartella dove salvare il summary JSON.")

    args = parser.parse_args()

    rl_config_path = _resolve_path(args.rl_config) or args.rl_config
    config = load_rl_agent_config(rl_config_path)
    dataset_path = _select_dataset(config, args.dataset_mode, args.dataset_path)

    env = OfflineMicrogridRLEnv(
        config_path=rl_config_path,
        config=config,
        random_start=False,
        seed=args.seed,
        start_step_default=int(args.start_step),
    )
    if args.no_normalization_update:
        env.set_normalization_update(False)

    ppo_cfg = (config.get("rl", {}) or {}).get("ppo", {}) or {}
    policy_net = _parse_policy_net(args.policy_net) or list(ppo_cfg.get("policy_net") or [])
    if policy_net:
        policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            activation_fn=nn.ReLU,
            net_arch=dict(pi=policy_net, vf=policy_net),
        )
    else:
        policy = FeedForward32Policy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 0.0,
        )
    policy.load_state_dict(torch.load(args.bc_policy, map_location="cpu"))

    total_reward, steps = _run_episode(
        env=env,
        policy=policy,
        deterministic=not args.stochastic,
        start_step=int(args.start_step),
        max_steps=args.max_steps,
    )

    summary = {
        "dataset_path": dataset_path,
        "steps": int(steps),
        "total_reward": float(total_reward),
    }
    print(json.dumps(summary, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "bc_eval_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
