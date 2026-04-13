import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import torch.nn as nn
import torch

#os.environ.setdefault("PYMGRID_BATTERY_DEBUG", "1")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
SIMULATOR_ROOT = PROJECT_ROOT / "SIMULATOR"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SIMULATOR.microgrid_simulator import MicrogridSimulator
from RL_utils_offline import load_offline_timeseries
from normalization import RunningMeanStd
from SIMULATOR.tools import add_module_columns, add_grid_cost_breakdown_columns


def load_rl_agent_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file) or {}

    for section in ("battery", "grid", "ems"):
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {config_path}")

    config.setdefault("rl", {})
    rl_cfg = config["rl"]
    scenario_cfg = config.setdefault("scenario", {})
    default_dataset = (
        rl_cfg.get("dataset_path")
        or scenario_cfg.get("inference_dataset_path")
        or "./data/scenario_0_timeseries_hourly.csv"
    )
    rl_cfg.setdefault("dataset_path", default_dataset)
    base_dataset = rl_cfg.get("dataset_path", default_dataset)
    rl_cfg.setdefault("dataset_path_train", base_dataset)
    rl_cfg.setdefault("dataset_path_eval", base_dataset)
    rl_cfg.setdefault("dataset_path_test", base_dataset)
    rl_cfg.setdefault("output_dir", "outputs/RL_AGENT")
    rl_cfg.setdefault("run_name", "ppo_offline")
    rl_cfg.setdefault("seed", None)
    episode_steps = rl_cfg.get("episode_steps")
    if episode_steps is not None:
        rl_cfg.setdefault("train_episode_steps", episode_steps)
        rl_cfg.setdefault("eval_episode_steps", episode_steps)
    rl_cfg.setdefault("train_episode_steps", 96)
    rl_cfg.setdefault("eval_episode_steps", rl_cfg["train_episode_steps"])
    rl_cfg.setdefault("episode_steps", rl_cfg["eval_episode_steps"])
    train_num_episodes = rl_cfg.get("num_episodes")
    if train_num_episodes is not None:
        rl_cfg.setdefault("train_num_episodes", train_num_episodes)
    rl_cfg.setdefault("train_num_episodes", 10)
    rl_cfg.setdefault("eval_episodes", 3)
    rl_cfg.setdefault("train_random_start", True)
    rl_cfg.setdefault("randomize_initial_SoE", rl_cfg.get("randomize_initial_SoC", False))
    rl_cfg.setdefault("eval_random_start", False)
    rl_cfg.setdefault("eval_start_step", 0)
    rl_cfg.setdefault("train_start_step", 0)
    rl_cfg.setdefault("action_deadband_kwh", 0.0)
    rl_cfg.setdefault("soe_action_guard_band", rl_cfg.get("soc_action_guard_band", 0.0))

    rl_cfg.setdefault("num_forecast", 0)
    rl_cfg.setdefault("include_time_feature", False)
    rl_cfg.setdefault("forecast_future", False)
    rl_cfg.setdefault("forecast_price", False)
    rl_cfg.setdefault("preprocessing", "simple")
    rl_cfg.setdefault("include_price", False)
    rl_cfg.setdefault("forecast_time", False)
    rl_cfg.setdefault("include_battery_power_limits_in_obs", True)
    rl_cfg.setdefault("allow_grid_charge_for_battery", True)
    rl_cfg.setdefault("allow_battery_export_to_grid", True)

    reward_cfg = rl_cfg.setdefault("reward", {})
    reward_cfg.setdefault("mode", "custom")
    reward_cfg.setdefault("use_pymgrid_reward", False)
    reward_cfg.setdefault("coeff_economic", 1.0)
    reward_cfg.setdefault("coeff_soe_violation", reward_cfg.get("coeff_soc_violation", 1.0))
    reward_cfg.setdefault("coeff_action_smoothness", 0.0)
    reward_cfg.setdefault("coeff_action_sign_change", 0.0)
    reward_cfg.setdefault("coeff_micro_throughput", 0.0)
    reward_cfg.setdefault("coeff_action_violation", 1.0)
    reward_cfg.setdefault("coeff_soe_boundary", reward_cfg.get("coeff_soc_boundary", 0.0))
    reward_cfg.setdefault("coeff_soh_calendar", 1.0)
    reward_cfg.setdefault("coeff_cyclic_aging", 1.0)
    reward_cfg.setdefault("coeff_wear_cost", 0.0)
    reward_cfg.setdefault("coeff_SSR", 0.0)
    reward_cfg.setdefault("sell_discount", 0.8)
    reward_cfg.setdefault("relative_reference_enabled", False)
    reward_cfg.setdefault("relative_reference_policy", "no_battery")
    reward_cfg.setdefault("relative_reference_normalize", True)
    reward_cfg.setdefault("relative_reference_eps", 1e-6)
    reward_cfg.setdefault("relative_reference_weight", 1.0)
    reward_cfg.setdefault("soe_tolerance", reward_cfg.get("soc_tolerance", 1e-6))
    reward_cfg.setdefault("action_tolerance", 1e-6)
    reward_cfg.setdefault("calendar_aging_per_step", 0.0)
    reward_cfg.setdefault("scale_micro_throughput", 1.0)
    reward_cfg.setdefault("scale_soe_boundary", reward_cfg.get("scale_soc_boundary", 1.0))
    reward_cfg.setdefault("micro_throughput_kwh", 0.0)
    reward_cfg.setdefault("scale_action_smoothness", 1.0)
    reward_cfg.setdefault("soe_boundary_margin", reward_cfg.get("soc_boundary_margin", 0.0))

    rl_cfg.setdefault("algorithm", "PPO")
    ppo_cfg = rl_cfg.setdefault("ppo", {})
    ppo_cfg.setdefault("learning_rate", 3e-4)
    ppo_cfg.setdefault("n_steps", 1024)
    ppo_cfg.setdefault("batch_size", 64)
    ppo_cfg.setdefault("n_epochs", 5)
    ppo_cfg.setdefault("gamma", 0.99)
    ppo_cfg.setdefault("gae_lambda", 0.95)
    ppo_cfg.setdefault("clip_range", 0.2)
    ppo_cfg.setdefault("ent_coef", 0.01)
    ppo_cfg.setdefault("policy_net", [256, 256])
    if "normalize_advantage" not in ppo_cfg:
        if "center_adv" in ppo_cfg:
            # External configs may use `center_adv`; map it to SB3 `normalize_advantage`.
            ppo_cfg["normalize_advantage"] = bool(ppo_cfg.get("center_adv", True))
        else:
            ppo_cfg["normalize_advantage"] = True
    sac_cfg = rl_cfg.setdefault("sac", {})
    sac_cfg.setdefault("learning_rate", 3e-4)
    sac_cfg.setdefault("buffer_size", 100000)
    sac_cfg.setdefault("learning_starts", 1000)
    sac_cfg.setdefault("batch_size", 256)
    sac_cfg.setdefault("tau", 0.005)
    sac_cfg.setdefault("gamma", 0.99)
    sac_cfg.setdefault("train_freq", 1)
    sac_cfg.setdefault("gradient_steps", 1)
    sac_cfg.setdefault("ent_coef", "auto")
    sac_cfg.setdefault("target_update_interval", 1)
    sac_cfg.setdefault("target_entropy", "auto")
    sac_cfg.setdefault("use_sde", False)
    sac_cfg.setdefault("sde_sample_freq", -1)
    sac_cfg.setdefault("policy_net", [256, 256])

    rl_cfg.setdefault("checkpoint_freq", 10000)
    rl_cfg.setdefault("num_envs", 1)
    rl_cfg.setdefault("vec_env_type", "auto")
    rl_cfg.setdefault("vec_env_start_method", None)
    rl_cfg.setdefault("observation_bounds_path", None)
    rl_cfg.setdefault("save_observation_bounds", None)
    obs_limits = rl_cfg.setdefault("observation_limits", {})
    obs_limits.setdefault("load", {})
    obs_limits.setdefault("pv", {})
    normalization_cfg = rl_cfg.setdefault("normalization", {})
    normalization_cfg.setdefault("stats_path", None)
    normalization_cfg.setdefault("save_stats", None)
    normalization_cfg.setdefault("epsilon", 1e-4)
    obs_norm_cfg = normalization_cfg.setdefault("observations", {})
    obs_norm_cfg.setdefault("enabled", False)
    obs_norm_cfg.setdefault("normalize_already_scaled", False)
    act_norm_cfg = normalization_cfg.setdefault("actions", {})
    act_norm_cfg.setdefault("enabled", False)
    reward_norm_cfg = normalization_cfg.setdefault("reward", {})
    reward_norm_cfg.setdefault("enabled", False)

    return config


def _resolve_worker_seed(base_seed: Optional[int], worker_idx: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + int(worker_idx)


def _resolve_vec_env_type(rl_cfg: Dict[str, Any], num_envs: int) -> str:
    raw = str(rl_cfg.get("vec_env_type", "auto")).strip().lower()
    if raw not in {"auto", "dummy", "subproc"}:
        raise ValueError(f"Unsupported rl.vec_env_type={raw!r}. Supported values: auto, dummy, subproc.")
    if num_envs <= 1:
        return "dummy"
    return "subproc" if raw == "auto" else raw


def _make_offline_env_factory(
    config_path: str,
    config: Dict[str, Any],
    random_start: bool,
    seed: Optional[int],
    start_step_default: Optional[int],
):
    def _make_env():
        return OfflineMicrogridRLEnv(
            config_path=config_path,
            config=deepcopy(config),
            random_start=random_start,
            seed=seed,
            start_step_default=start_step_default,
        )

    return _make_env

class OfflineMicrogridRLEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        config_path: str = str(ROOT_DIR / "PARAMS" / "OPSD" / "params_RL_agent_OPSD_1_DAY.yml"),
        config: Optional[Dict[str, Any]] = None,
        random_start: Optional[bool] = None,
        seed: Optional[int] = None,
        start_step_default: Optional[int] = None,
    ):
        self.config_path = config_path
        self.config = config or load_rl_agent_config(config_path)

        data = load_offline_timeseries(self.config)
        self.time_series = data["time_series"]
        self.load_series = data["load_series"]
        self.pv_series = data["pv_series"]
        self.timestamps = data["timestamps"]
        self.price_buy = data["price_buy"]
        self.price_sell = data["price_sell"]
        self.grid_series = data["grid_series"]

        timezone = self.config["ems"]["timezone"]
        self.timestamps_local = self.timestamps.dt.tz_convert(timezone)

        self.simulator = MicrogridSimulator(
            config_path=config_path,
            online=False,
            load_time_series=self.load_series,
            pv_time_series=self.pv_series,
            grid_time_series=self.grid_series,
        )
        self.microgrid = self.simulator.build_microgrid()

        self.load_module = self.microgrid.modules["load"][0]
        self.pv_module = self.microgrid.modules["pv"][0]
        self.grid_module = self.microgrid.modules["grid"][0]
        self.battery_module = self.microgrid.battery[0]
        try:
            self.balancing_module = self.microgrid.modules["balancing"][0]
        except Exception:
            self.balancing_module = None

        rl_cfg = self.config["rl"]
        self.num_forecast = int(rl_cfg.get("num_forecast", 0))
        self.include_time_feature = bool(rl_cfg.get("include_time_feature", False))
        self.forecast_future = bool(rl_cfg.get("forecast_future", False))
        self.forecast_price = bool(rl_cfg.get("forecast_price", False))
        self.preprocessing = str(rl_cfg.get("preprocessing", "simple")).lower()
        self.include_price = bool(rl_cfg.get("include_price", False))
        self.forecast_time = bool(rl_cfg.get("forecast_time", False))
        self.include_battery_power_limits_in_obs = bool(
            rl_cfg.get("include_battery_power_limits_in_obs", True)
        )
        self._allow_grid_charge_for_battery = bool(
            rl_cfg.get("allow_grid_charge_for_battery", True)
        )
        self._allow_battery_export_to_grid = bool(
            rl_cfg.get("allow_battery_export_to_grid", True)
        )
        normalization_cfg = rl_cfg.get("normalization", {}) or {}
        obs_norm_cfg = normalization_cfg.get("observations", {}) or {}
        act_norm_cfg = normalization_cfg.get("actions", {}) or {}
        reward_norm_cfg = normalization_cfg.get("reward", {}) or {}
        self._obs_norm_enabled = bool(obs_norm_cfg.get("enabled", False))
        self._normalize_scaled_obs = bool(obs_norm_cfg.get("normalize_already_scaled", False))
        self._action_norm_enabled = bool(act_norm_cfg.get("enabled", False))
        self._reward_norm_enabled = bool(reward_norm_cfg.get("enabled", False))
        self._norm_epsilon = float(normalization_cfg.get("epsilon", 1e-4))
        stats_path = normalization_cfg.get("stats_path")
        self._normalization_stats_path = Path(stats_path) if stats_path else None
        self._normalization_update = True

        self.max_steps = max(1, len(self.load_series) - self.num_forecast - 1)
        self.episode_length = int(rl_cfg.get("episode_steps", self.max_steps))
        if self.episode_length <= 0 or self.episode_length > self.max_steps:
            self.episode_length = self.max_steps

        self.random_start_default = (
            bool(rl_cfg.get("train_random_start", True))
            if random_start is None
            else bool(random_start)
        )
        self._randomize_initial_soc = bool(
            rl_cfg.get("randomize_initial_SoE", rl_cfg.get("randomize_initial_SoC", False))
        )
        self._action_deadband_kwh = float(rl_cfg.get("action_deadband_kwh", 0.0) or 0.0)
        self._soe_action_guard_band = float(
            rl_cfg.get("soe_action_guard_band", rl_cfg.get("soc_action_guard_band", 0.0)) or 0.0
        )
        self.eval_start_step = int(rl_cfg.get("eval_start_step", 0))
        if start_step_default is None:
            start_step_default = self.eval_start_step
        self.default_start_step = int(start_step_default)

        self.rng = np.random.default_rng(seed)
        self._start_step = 0
        self._final_step = self.episode_length
        self._last_soh = self._get_soh()
        self._prev_batt_action = None

        self._action_max_discharge = float(self.battery_module.max_discharge)
        self._action_max_charge = float(self.battery_module.max_charge)
        if self._action_norm_enabled:
            action_low, action_high = -1.0, 1.0
        else:
            action_low = -self._action_max_charge
            action_high = self._action_max_discharge
        self.action_space = spaces.Box(
            low=np.array([action_low], dtype=np.float32),
            high=np.array([action_high], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = self._get_obs_dim()
        self._obs_rms = RunningMeanStd(shape=(obs_dim,), epsilon=self._norm_epsilon) if self._obs_norm_enabled else None
        self._reward_rms = RunningMeanStd(shape=(), epsilon=self._norm_epsilon) if self._reward_norm_enabled else None
        self._obs_norm_mask = self._build_obs_norm_mask(obs_dim) if self._obs_norm_enabled else None
        if self._normalization_stats_path and self._normalization_stats_path.exists():
            self.load_normalization_state(self._normalization_stats_path)
        if self._obs_norm_enabled:
            obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
            obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        else:
            obs_low, obs_high = self._resolve_obs_bounds(obs_dim)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

    def _get_obs_dim(self) -> int:
        dim = 4  # load, pv, soe, soh
        if self.include_battery_power_limits_in_obs:
            dim += 2
        if self.include_time_feature:
            dim += 4
        if self.include_price:
            dim += 2
        if self.forecast_price and self.num_forecast > 0:
            dim += 2 * self.num_forecast
        if self.forecast_future and self.num_forecast > 0:
            if self.preprocessing == "simple":
                dim += 2 * self.num_forecast
            elif self.preprocessing == "net-load":
                dim += self.num_forecast
            else:
                raise ValueError(f"Unsupported preprocessing: {self.preprocessing}")
        return dim

    def _build_obs_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        low = np.full(dim, -np.inf, dtype=np.float32)
        high = np.full(dim, np.inf, dtype=np.float32)

        load_min, load_max, pv_min, pv_max = self._get_config_obs_limits()
        net_min = load_min - pv_max
        net_max = load_max - pv_min
        price_buy_min = float(np.nanmin(self.price_buy))
        price_buy_max = float(np.nanmax(self.price_buy))
        price_sell_min = float(np.nanmin(self.price_sell))
        price_sell_max = float(np.nanmax(self.price_sell))

        idx = 0
        low[idx], high[idx] = load_min, load_max
        idx += 1
        low[idx], high[idx] = pv_min, pv_max
        idx += 1
        low[idx] = float(self.battery_module.min_soc or 0.0)
        high[idx] = float(self.battery_module.max_soc or 1.0)
        idx += 1
        low[idx], high[idx] = 0.0, 1.0
        idx += 1
        if self.include_battery_power_limits_in_obs:
            low[idx], high[idx] = 0.0, float(self.battery_module.max_discharge)
            idx += 1
            low[idx], high[idx] = 0.0, float(self.battery_module.max_charge)
            idx += 1

        if self.include_time_feature:
            low[idx:idx + 4] = -1.0
            high[idx:idx + 4] = 1.0
            idx += 4

        if self.include_price:
            low[idx], high[idx] = price_buy_min, price_buy_max
            idx += 1
            low[idx], high[idx] = price_sell_min, price_sell_max
            idx += 1

        if self.forecast_price and self.num_forecast > 0:
            for _ in range(self.num_forecast):
                low[idx], high[idx] = price_buy_min, price_buy_max
                idx += 1
                low[idx], high[idx] = price_sell_min, price_sell_max
                idx += 1

        if self.forecast_future and self.num_forecast > 0:
            if self.preprocessing == "simple":
                for _ in range(self.num_forecast):
                    low[idx], high[idx] = load_min, load_max
                    idx += 1
                    low[idx], high[idx] = pv_min, pv_max
                    idx += 1
            elif self.preprocessing == "net-load":
                for _ in range(self.num_forecast):
                    low[idx], high[idx] = net_min, net_max
                    idx += 1

        return low, high

    def _get_config_obs_limits(self) -> Tuple[float, float, float, float]:
        rl_cfg = self.config.get("rl", {})
        limits_cfg = rl_cfg.get("observation_limits") or {}

        def parse_limits(label: str, cfg: Dict[str, Any]) -> Tuple[float, float]:
            if "min" not in cfg or "max" not in cfg:
                raise ValueError(
                    f"Missing observation_limits.{label}.min/max in RL config; "
                    "set explicit bounds in the YAML config."
                )
            low = float(cfg["min"])
            high = float(cfg["max"])
            if low > high:
                raise ValueError(
                    f"Invalid observation_limits.{label}: min {low} exceeds max {high}."
                )
            return low, high

        load_min, load_max = parse_limits("load", limits_cfg.get("load") or {})
        pv_min, pv_max = parse_limits("pv", limits_cfg.get("pv") or {})
        return load_min, load_max, pv_min, pv_max

    def _resolve_obs_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        rl_cfg = self.config.get("rl", {})
        bounds_path = rl_cfg.get("observation_bounds_path")
        save_bounds = bool(rl_cfg.get("save_observation_bounds", False))
        path = Path(bounds_path) if bounds_path else None

        if path is not None:
            if path.exists():
                return self._load_obs_bounds(path, dim)
            if not save_bounds:
                raise FileNotFoundError(f"Observation bounds file not found: {path}")

        low, high = self._build_obs_bounds(dim)

        if path is not None and save_bounds:
            self._save_obs_bounds(path, low, high)

        return low, high

    @staticmethod
    def _load_obs_bounds(path: Path, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        low = np.asarray(payload.get("low", []), dtype=np.float32)
        high = np.asarray(payload.get("high", []), dtype=np.float32)
        if low.shape != (dim,) or high.shape != (dim,):
            raise ValueError(
                f"Observation bounds in {path} have shape {low.shape}/{high.shape}; expected {(dim,)}."
            )
        return low, high

    @staticmethod
    def _save_obs_bounds(path: Path, low: np.ndarray, high: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"low": low.astype(float).tolist(), "high": high.astype(float).tolist()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def set_normalization_update(self, enabled: bool) -> None:
        self._normalization_update = bool(enabled)

    def _build_obs_norm_mask(self, dim: int) -> np.ndarray:
        mask = np.ones(dim, dtype=bool)
        if self._normalize_scaled_obs:
            return mask

        idx = 0
        idx += 1  # load
        idx += 1  # pv
        mask[idx] = False  # soe
        idx += 1
        mask[idx] = False  # soh
        idx += 1
        if self.include_battery_power_limits_in_obs:
            idx += 1  # max production
            idx += 1  # max consumption

        if self.include_time_feature:
            # Time features are already in [-1, 1], so keep them unnormalized.
            mask[idx:idx + 4] = False
            idx += 4

        if self.include_price:
            mask[idx] = False
            idx += 1
            mask[idx] = False
            idx += 1

        if self.forecast_price and self.num_forecast > 0:
            for _ in range(self.num_forecast):
                mask[idx] = False
                idx += 1
                mask[idx] = False
                idx += 1

        if self.forecast_future and self.num_forecast > 0:
            if self.preprocessing == "simple":
                idx += 2 * self.num_forecast
            elif self.preprocessing == "net-load":
                idx += self.num_forecast
        return mask

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._obs_norm_enabled or self._obs_rms is None:
            return obs
        if self._normalization_update:
            self._obs_rms.update(obs)
        mean = self._obs_rms.mean
        std = self._obs_rms.std
        normalized = (obs - mean) / std
        if not self._normalize_scaled_obs and self._obs_norm_mask is not None:
            normalized = normalized.astype(np.float32)
            normalized[~self._obs_norm_mask] = obs[~self._obs_norm_mask]
        return normalized.astype(np.float32)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._obs_norm_enabled or self._obs_rms is None:
            return obs
        mean = self._obs_rms.mean
        std = self._obs_rms.std
        denormalized = obs * std + mean
        if not self._normalize_scaled_obs and self._obs_norm_mask is not None:
            denormalized[~self._obs_norm_mask] = obs[~self._obs_norm_mask]
        return denormalized.astype(np.float32)

    def _normalize_reward(self, reward: float) -> float:
        reward = float(reward)
        if not self._reward_norm_enabled or self._reward_rms is None:
            return reward
        if self._normalization_update:
            self._reward_rms.update(np.array([reward], dtype=np.float64))
        std = float(self._reward_rms.std)
        if std <= 0.0:
            std = self._norm_epsilon
        return float(reward / std)

    def denormalize_reward(self, reward: float) -> float:
        reward = float(reward)
        if not self._reward_norm_enabled or self._reward_rms is None:
            return reward
        std = float(self._reward_rms.std)
        if std <= 0.0:
            std = self._norm_epsilon
        return float(reward * std)

    def normalize_action(self, action: float) -> float:
        action = float(action)
        if not self._action_norm_enabled:
            return action
        if action >= 0.0:
            return float(action / self._action_max_discharge) if self._action_max_discharge > 0 else 0.0
        return float(action / self._action_max_charge) if self._action_max_charge > 0 else 0.0

    def denormalize_action(self, action: float) -> float:
        action = float(action)
        if not self._action_norm_enabled:
            return action
        if action >= 0.0:
            return float(action * self._action_max_discharge)
        return float(action * self._action_max_charge)

    def _build_normalization_payload(self) -> Dict[str, Any]:
        payload = {"version": 1}
        if self._obs_rms is not None:
            payload["obs_dim"] = int(self._get_obs_dim())
            payload["obs"] = self._obs_rms.to_dict()
        if self._reward_rms is not None:
            payload["reward"] = self._reward_rms.to_dict()
        return payload

    def _apply_normalization_payload(self, payload: Dict[str, Any], source: str = "payload") -> None:
        payload = payload or {}

        if self._obs_rms is not None and payload.get("obs") is not None:
            obs_dim = payload.get("obs_dim")
            if obs_dim is not None and int(obs_dim) != int(self._get_obs_dim()):
                raise ValueError(
                    f"Normalization stats in {source} have obs_dim={obs_dim}, expected {self._get_obs_dim()}."
                )
            obs_rms = RunningMeanStd.from_dict(payload["obs"], epsilon=self._norm_epsilon)
            if obs_rms.mean.shape != self._obs_rms.mean.shape:
                raise ValueError(
                    f"Normalization stats in {source} have obs shape {obs_rms.mean.shape}, "
                    f"expected {self._obs_rms.mean.shape}."
                )
            self._obs_rms = obs_rms

        if self._reward_rms is not None and payload.get("reward") is not None:
            reward_rms = RunningMeanStd.from_dict(payload["reward"], epsilon=self._norm_epsilon)
            if reward_rms.mean.shape != self._reward_rms.mean.shape:
                raise ValueError(
                    f"Normalization stats in {source} have reward shape {reward_rms.mean.shape}, "
                    f"expected {self._reward_rms.mean.shape}."
                )
            self._reward_rms = reward_rms

    def export_normalization_state(self) -> Dict[str, Any]:
        return self._build_normalization_payload()

    def import_normalization_state(self, payload: Dict[str, Any]) -> None:
        self._apply_normalization_payload(payload=payload, source="payload")

    def save_normalization_state(self, path: Path) -> None:
        path = Path(path)
        payload = self._build_normalization_payload()

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_normalization_state(self, path: Path) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        self._apply_normalization_payload(payload=payload, source=str(path))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None,
              random_start: Optional[bool] = None, start_step: Optional[int] = None):
        super().reset(seed=seed)

        if start_step is None:
            if random_start is None:
                random_start = self.random_start_default
            if random_start:
                latest_start = max(0, self.max_steps - self.episode_length)
                start_step = int(self.rng.integers(0, latest_start + 1))
            else:
                start_step = int(self.default_start_step)

        start_step = max(0, min(int(start_step), self.max_steps - 1))
        final_step = min(start_step + self.episode_length, self.max_steps)
        if final_step <= start_step:
            final_step = min(start_step + 1, self.max_steps)

        self._start_step = start_step
        self._final_step = final_step

        self.microgrid.initial_step = start_step
        self.microgrid.final_step = final_step
        self.microgrid.reset()
        if self._randomize_initial_soc:
            self._randomize_battery_soc()
        self._prime_transition_model()

        self._last_soh = self._get_soh()
        self._prev_batt_action = None
        obs = self._normalize_obs(self._build_obs())
        info = {"current_step": int(self.microgrid.current_step)}
        return obs, info

    def _randomize_battery_soc(self) -> None:
        soc_min = float(self.battery_module.min_soc or 0.0)
        soc_max = float(self.battery_module.max_soc or 1.0)
        if soc_max < soc_min:
            soc_min, soc_max = soc_max, soc_min
        if np.isclose(soc_min, soc_max, atol=1e-9):
            soc_value = soc_min
        else:
            soc_value = float(self.rng.uniform(soc_min, soc_max))

        self.battery_module.soc = soc_value

        transition_model = getattr(self.battery_module, "battery_transition_model", None)
        if transition_model is None:
            return
        if hasattr(transition_model, "soc"):
            transition_model.soc = soc_value
        if hasattr(transition_model, "soe"):
            transition_model.soe = soc_value
        if hasattr(transition_model, "_interp_voc_r0"):
            temperature_c = float(getattr(transition_model, "temperature_c", 25.0))
            soh = float(getattr(transition_model, "soh", 1.0) or 1.0)
            try:
                voc, _ = transition_model._interp_voc_r0(soc_value, temperature_c, soh)
                transition_model.v_prev = voc
            except Exception:
                return

    def step(self, action):
        step_idx = min(int(self.microgrid.current_step), self.max_steps - 1)

        load_kwh = float(self.load_module.current_load)
        pv_kwh = float(self.pv_module.current_renewable)
        net_load = load_kwh - pv_kwh

        requested_batt_input = float(np.array(action).reshape(-1)[0])
        requested_batt_norm = None
        if self._action_norm_enabled:
            requested_batt_norm = float(np.clip(requested_batt_input, -1.0, 1.0))
            requested_batt = self.denormalize_action(requested_batt_norm)
        else:
            requested_batt = requested_batt_input
        # Suppress micro-movements and enforce SoE guard bands.
        if self._action_deadband_kwh > 0.0 and abs(requested_batt) < self._action_deadband_kwh:
            requested_batt = 0.0
        if self._soe_action_guard_band > 0.0:
            soe_now = float(self.battery_module.soc)
            soe_min = float(self.battery_module.min_soc or 0.0)
            soe_max = float(self.battery_module.max_soc or 1.0)
            if soe_now <= soe_min + self._soe_action_guard_band:
                requested_batt = min(requested_batt, 0.0)
            if soe_now >= soe_max - self._soe_action_guard_band:
                requested_batt = max(requested_batt, 0.0)
        requested_batt_after_policy, batt_policy = self._apply_battery_action_policy_constraints(
            requested=requested_batt,
            net_load=net_load,
        )
        battery_action, batt_physically_clipped = self._clip_battery_action(requested_batt_after_policy)
        batt_clipped = bool(batt_policy["was_clipped"] or batt_physically_clipped)

        requested_grid = net_load - battery_action
        grid_action, grid_clipped = self._clip_grid_action(requested_grid)

        prev_soh = self._get_soh()
        _, base_reward, done, info = self.microgrid.step(
            {"battery": battery_action, "grid": grid_action},
            normalized=False,
        )
        bms_battery_info = self._extract_battery_step_info(info)
        grid_step_info = {}
        balancing_step_info = {}
        if isinstance(info, dict):
            grid_entries = info.get("grid")
            if isinstance(grid_entries, list):
                for entry in grid_entries:
                    if isinstance(entry, dict):
                        grid_step_info = entry
                        break
            balancing_entries = info.get("balancing")
            if isinstance(balancing_entries, list):
                for entry in balancing_entries:
                    if isinstance(entry, dict):
                        balancing_step_info = entry
                        break
        current_soh = self._get_soh()

        batt_actual, batt_discharge, batt_charge = self._get_battery_flow()
        grid_import, grid_export = self._get_grid_flow()

        price_buy = float(self.price_buy[step_idx])
        price_sell = float(self.price_sell[step_idx])
        co2_production_step = 0.0
        try:
            co2_production_step = float(grid_step_info.get("co2_production", 0.0) or 0.0)
        except (TypeError, ValueError):
            co2_production_step = 0.0
        co2_per_kwh_step = 0.0
        if grid_import > 1e-9:
            co2_per_kwh_step = max(0.0, co2_production_step / max(grid_import, 1e-9))
        if co2_per_kwh_step <= 0.0:
            try:
                co2_per_kwh_step = float(self.grid_series[step_idx, 2])
            except Exception:
                co2_per_kwh_step = 0.0
        if not np.isfinite(co2_per_kwh_step) or co2_per_kwh_step < 0.0:
            co2_per_kwh_step = 0.0
        grid_co2_cost_per_unit = float(getattr(self.grid_module, "cost_per_unit_co2", 0.0) or 0.0)
        co2_cost_step = co2_production_step * grid_co2_cost_per_unit
        balancing_module = getattr(self, "balancing_module", None)
        loss_load_cost_coeff = float(getattr(balancing_module, "loss_load_cost", 0.0) or 0.0)
        overgeneration_cost_coeff = float(getattr(balancing_module, "overgeneration_cost", 0.0) or 0.0)

        reward_cfg = self.config["rl"]["reward"]
        reward_mode_raw = reward_cfg.get("mode", "custom")
        reward_mode = str(reward_mode_raw).strip().lower()
        use_pymgrid_reward = bool(reward_cfg.get("use_pymgrid_reward", False)) or reward_mode in {
            "pymgrid",
            "base",
            "module",
        }
        relative_reference_enabled = bool(reward_cfg.get("relative_reference_enabled", False))
        relative_reference_policy = str(reward_cfg.get("relative_reference_policy", "no_battery")).strip().lower()
        relative_reference_normalize = bool(reward_cfg.get("relative_reference_normalize", True))
        relative_reference_eps = float(reward_cfg.get("relative_reference_eps", 1e-6) or 1e-6)
        relative_reference_weight = float(reward_cfg.get("relative_reference_weight", 1.0))
        if relative_reference_eps <= 0.0:
            relative_reference_eps = 1e-6
        sell_discount = float(reward_cfg.get("sell_discount", 0.8))
        cost_economic = grid_import * price_buy - grid_export * price_sell * sell_discount
        coeff_economic = float(reward_cfg.get("coeff_economic", 1.0))
        coeff_soe_violation = float(reward_cfg.get("coeff_soe_violation", reward_cfg.get("coeff_soc_violation", 1.0)))
        coeff_action_violation = float(reward_cfg.get("coeff_action_violation", 1.0))
        coeff_action_smoothness = float(reward_cfg.get("coeff_action_smoothness", 0.0))
        coeff_action_sign_change = float(reward_cfg.get("coeff_action_sign_change", 0.0))
        coeff_micro_throughput = float(reward_cfg.get("coeff_micro_throughput", 0.0))
        coeff_soe_boundary = float(reward_cfg.get("coeff_soe_boundary", reward_cfg.get("coeff_soc_boundary", 0.0)))
        coeff_cyclic_aging = float(reward_cfg.get("coeff_cyclic_aging", 1.0))
        coeff_soh_calendar = float(reward_cfg.get("coeff_soh_calendar", 1.0))
        coeff_wear_cost = float(reward_cfg.get("coeff_wear_cost", 0.0))
        coeff_ssr = float(reward_cfg.get("coeff_SSR", 0.0))
        scale_components = bool(reward_cfg.get("scale_reward_components", False))
        scale_economic = float(reward_cfg.get("scale_economic", 1.0))
        scale_action_violation = float(reward_cfg.get("scale_action_violation", 1.0))
        scale_action_smoothness = float(reward_cfg.get("scale_action_smoothness", 1.0))
        scale_micro_throughput = float(reward_cfg.get("scale_micro_throughput", 1.0))
        scale_soe_boundary = float(reward_cfg.get("scale_soe_boundary", reward_cfg.get("scale_soc_boundary", 1.0)))
        micro_throughput_kwh = float(reward_cfg.get("micro_throughput_kwh", 0.0))
        soe_boundary_margin = float(reward_cfg.get("soe_boundary_margin", reward_cfg.get("soc_boundary_margin", 0.0)))
        scale_wear_cost = float(reward_cfg.get("scale_wear_cost", 1.0))
        scale_ssr = float(reward_cfg.get("scale_ssr", 1.0))
        if scale_economic <= 0.0:
            scale_economic = 1.0
        if scale_action_violation <= 0.0:
            scale_action_violation = 1.0
        if scale_action_smoothness <= 0.0:
            scale_action_smoothness = 1.0
        if scale_micro_throughput <= 0.0:
            scale_micro_throughput = 1.0
        if scale_soe_boundary <= 0.0:
            scale_soe_boundary = 1.0
        if scale_economic <= 0.0:
            scale_economic = 1.0
        if scale_wear_cost <= 0.0:
            scale_wear_cost = 1.0
        if scale_ssr <= 0.0:
            scale_ssr = 1.0
        bms_manager = getattr(self.battery_module, "battery_bms_manager", None)
        transition_model = getattr(self.battery_module, "battery_transition_model", None)
        wear_cost = None
        if bms_battery_info:
            raw_bms_wear_cost = bms_battery_info.get("wear_cost_step", bms_battery_info.get("wear_cost"))
            try:
                if raw_bms_wear_cost is not None:
                    wear_cost = float(raw_bms_wear_cost)
            except (TypeError, ValueError):
                wear_cost = None
        if wear_cost is None and bms_manager is not None:
            try:
                bms_wear = float(getattr(bms_manager, "last_wear_cost", np.nan))
                if np.isfinite(bms_wear):
                    wear_cost = bms_wear
            except (TypeError, ValueError):
                wear_cost = None
        if wear_cost is None and transition_model is not None:
            try:
                wear_cost = float(getattr(transition_model, "last_wear_cost", 0.0) or 0.0)
            except (TypeError, ValueError):
                wear_cost = 0.0
        if wear_cost is None:
            wear_cost = 0.0

        soe = float(self.battery_module.soc)
        soe_min = float(self.battery_module.min_soc or 0.0)
        soe_max = float(self.battery_module.max_soc or 1.0)
        soe_tol = float(reward_cfg.get("soe_tolerance", reward_cfg.get("soc_tolerance", 1e-6)))
        soe_violation = None
        if bms_battery_info:
            raw_bms_violation = bms_battery_info.get("soe_violation", bms_battery_info.get("clip_soe_limit"))
            try:
                if raw_bms_violation is not None:
                    soe_violation = float(raw_bms_violation)
            except (TypeError, ValueError):
                soe_violation = None
        if soe_violation is None:
            soe_violation = 1.0 if (soe < soe_min - soe_tol or soe > soe_max + soe_tol) else 0.0
        soe_boundary_penalty = 0.0
        if soe_boundary_margin > 0.0:
            if soe <= soe_min + soe_boundary_margin and requested_batt > 0.0:
                soe_boundary_penalty = abs(requested_batt)
            elif soe >= soe_max - soe_boundary_margin and requested_batt < 0.0:
                soe_boundary_penalty = abs(requested_batt)

        action_tol = float(reward_cfg.get("action_tolerance", 1e-6))
        action_violation_amount = abs(requested_batt - battery_action) + abs(requested_grid - grid_action)
        action_violation = float(action_violation_amount)
        action_violation_flag = 1.0 if action_violation_amount > action_tol else 0.0
        if self._prev_batt_action is None:
            action_smoothness = 0.0
        else:
            action_smoothness = abs(battery_action - self._prev_batt_action)
        if self._prev_batt_action is None:
            action_sign_change = 0.0
        else:
            prev_action = self._prev_batt_action
            if abs(prev_action) <= action_tol or abs(battery_action) <= action_tol:
                action_sign_change = 0.0
            else:
                action_sign_change = abs(battery_action - prev_action) if (prev_action * battery_action) < 0.0 else 0.0
        self._prev_batt_action = battery_action
        step_throughput = abs(batt_charge) + abs(batt_discharge)
        if micro_throughput_kwh > 0.0 and 0.0 < step_throughput < micro_throughput_kwh:
            micro_throughput = step_throughput
        else:
            micro_throughput = 0.0

        cyclic_aging = max(0.0, prev_soh - current_soh)
        calendar_aging = max(0.0, 1.0 - current_soh)
        calendar_aging += float(reward_cfg.get("calendar_aging_per_step", 0.0))

        ssr = self._self_sufficiency_ratio(load_kwh, pv_kwh, batt_actual)

        if scale_components:
            economic_for_reward = cost_economic / scale_economic
            action_violation_for_reward = action_violation / scale_action_violation
            action_smoothness_for_reward = action_smoothness / scale_action_smoothness
            micro_throughput_for_reward = micro_throughput / scale_micro_throughput
            soe_boundary_for_reward = soe_boundary_penalty / scale_soe_boundary
            wear_cost_for_reward = wear_cost / scale_wear_cost
            ssr_for_reward = ssr / scale_ssr
        else:
            economic_for_reward = cost_economic
            action_violation_for_reward = action_violation
            action_smoothness_for_reward = action_smoothness
            micro_throughput_for_reward = micro_throughput
            soe_boundary_for_reward = soe_boundary_penalty
            wear_cost_for_reward = wear_cost
            ssr_for_reward = ssr

        reference_battery_action = 0.0
        reference_grid_action_requested = 0.0
        reference_grid_action = 0.0
        reference_grid_import = 0.0
        reference_grid_export = 0.0
        reference_cost_economic = 0.0
        reference_reward_term_economic = 0.0
        reference_loss_load_energy = 0.0
        reference_overgeneration_energy = 0.0
        reference_internal_energy_delta = 0.0
        reference_cycle_cost_step = 0.0
        reference_wear_cost_step = 0.0
        reference_co2_production_step = 0.0
        reference_co2_cost_step = 0.0
        reference_reward_term_pymgrid_proxy = 0.0
        reward_term_reference = 0.0
        reward_term_reference_delta_raw = 0.0
        reward_term_reference_delta_scaled = 0.0
        reward_term_reference_denom = 1.0
        if relative_reference_enabled:
            if relative_reference_policy == "no_battery":
                reference_battery_action = 0.0
            elif relative_reference_policy == "rbc":
                max_discharge = max(
                    0.0,
                    min(
                        float(getattr(self.battery_module, "max_discharge", 0.0) or 0.0),
                        float(getattr(self.battery_module, "max_production", 0.0) or 0.0),
                    ),
                )
                max_charge = max(
                    0.0,
                    min(
                        float(getattr(self.battery_module, "max_charge", 0.0) or 0.0),
                        float(getattr(self.battery_module, "max_consumption", 0.0) or 0.0),
                    ),
                )
                if float(net_load) >= 0.0:
                    reference_battery_action = min(float(net_load), max_discharge)
                else:
                    reference_battery_action = -min(-float(net_load), max_charge)
            else:
                raise ValueError(
                    f"Unsupported rl.reward.relative_reference_policy={relative_reference_policy!r}. "
                    "Supported values: no_battery, rbc."
                )

            reference_grid_action_requested = float(net_load) - reference_battery_action
            reference_grid_action, _ = self._clip_grid_action(reference_grid_action_requested)
            reference_grid_import = max(reference_grid_action, 0.0)
            reference_grid_export = max(-reference_grid_action, 0.0)
            reference_cost_economic = (
                reference_grid_import * price_buy - reference_grid_export * price_sell * sell_discount
            )
            reference_economic_for_reward = (
                reference_cost_economic / scale_economic if scale_components else reference_cost_economic
            )
            reference_reward_term_economic = -coeff_economic * reference_economic_for_reward
            reference_net_balance = float(net_load) - reference_battery_action - reference_grid_action
            if reference_net_balance >= 0.0:
                reference_loss_load_energy = reference_net_balance
                reference_overgeneration_energy = 0.0
            else:
                reference_loss_load_energy = 0.0
                reference_overgeneration_energy = -reference_net_balance
            reference_eta = float(getattr(self.battery_module, "efficiency", 1.0) or 1.0)
            if not np.isfinite(reference_eta) or reference_eta <= 0.0:
                reference_eta = 1.0
            if reference_battery_action > 0.0:
                reference_internal_energy_delta = -reference_battery_action / reference_eta
            elif reference_battery_action < 0.0:
                reference_internal_energy_delta = (-reference_battery_action) * reference_eta
            else:
                reference_internal_energy_delta = 0.0
            reference_cycle_cost_step = abs(reference_internal_energy_delta) * float(
                getattr(self.battery_module, "battery_cost_cycle", 0.0) or 0.0
            )
            reference_wear_cost_step = 0.0
            reference_co2_production_step = reference_grid_import * co2_per_kwh_step
            reference_co2_cost_step = reference_co2_production_step * grid_co2_cost_per_unit
            reference_loss_load_cost_step = reference_loss_load_energy * loss_load_cost_coeff
            reference_overgeneration_cost_step = reference_overgeneration_energy * overgeneration_cost_coeff
            reference_reward_term_pymgrid_proxy = (
                -reference_grid_import * price_buy
                + reference_grid_export * price_sell
                - reference_co2_cost_step
                - reference_loss_load_cost_step
                - reference_overgeneration_cost_step
                - reference_cycle_cost_step
                - reference_wear_cost_step
            )

        if use_pymgrid_reward:
            reward_term_economic = 0.0
            reward_term_soe_violation = 0.0
            reward_term_action_violation = 0.0
            reward_term_action_smoothness = 0.0
            reward_term_action_sign_change = 0.0
            reward_term_micro_throughput = 0.0
            reward_term_soe_boundary = 0.0
            reward_term_cyclic_aging = 0.0
            reward_term_calendar_aging = 0.0
            reward_term_wear_cost = 0.0
            reward_term_ssr = 0.0
            reward_term_pymgrid = float(base_reward)
            if relative_reference_enabled:
                reward_term_reference_delta_raw = reward_term_pymgrid - reference_reward_term_pymgrid_proxy
                reward_term_reference_denom = (
                    max(abs(reference_reward_term_pymgrid_proxy), relative_reference_eps)
                    if relative_reference_normalize
                    else 1.0
                )
                reward_term_reference_delta_scaled = reward_term_reference_delta_raw / reward_term_reference_denom
                reward_term_reference = relative_reference_weight * reward_term_reference_delta_scaled
                reward_raw = reward_term_reference
            else:
                reward_raw = reward_term_pymgrid
        else:
            reward_term_economic = -coeff_economic * economic_for_reward
            if relative_reference_enabled:
                reward_term_reference_delta_raw = reward_term_economic - reference_reward_term_economic
                reward_term_reference_denom = (
                    max(abs(reference_reward_term_economic), relative_reference_eps)
                    if relative_reference_normalize
                    else 1.0
                )
                reward_term_reference_delta_scaled = reward_term_reference_delta_raw / reward_term_reference_denom
                reward_term_reference = relative_reference_weight * reward_term_reference_delta_scaled
                reward_term_economic = reward_term_reference

            reward_term_soe_violation = -coeff_soe_violation * soe_violation
            reward_term_action_violation = -coeff_action_violation * action_violation_for_reward
            reward_term_action_smoothness = -coeff_action_smoothness * action_smoothness_for_reward
            reward_term_action_sign_change = -coeff_action_sign_change * action_sign_change
            reward_term_micro_throughput = -coeff_micro_throughput * micro_throughput_for_reward
            reward_term_soe_boundary = -coeff_soe_boundary * soe_boundary_for_reward
            reward_term_cyclic_aging = -coeff_cyclic_aging * cyclic_aging
            reward_term_calendar_aging = -coeff_soh_calendar * calendar_aging
            reward_term_wear_cost = -coeff_wear_cost * wear_cost_for_reward
            reward_term_ssr = coeff_ssr * ssr_for_reward
            reward_term_pymgrid = 0.0

            reward_raw = (
                reward_term_economic
                + reward_term_soe_violation
                + reward_term_action_smoothness
                + reward_term_action_sign_change
                + reward_term_micro_throughput
                + reward_term_action_violation
                + reward_term_soe_boundary
                + reward_term_cyclic_aging
                + reward_term_calendar_aging
                + reward_term_wear_cost
                + reward_term_ssr
            )

        obs = self._normalize_obs(self._build_obs())
        reward = self._normalize_reward(reward_raw)

        info = dict(info or {})
        info.update(
            {
                "current_step": int(self.microgrid.current_step),
                "step_idx": int(step_idx),
                "reward": float(reward_raw),
                "reward_normalized": float(reward) if self._reward_norm_enabled else float(reward_raw),
                "base_reward": float(base_reward),
                "reward_mode": "pymgrid" if use_pymgrid_reward else "custom",
                "reward_term_pymgrid": float(reward_term_pymgrid),
                "reward_term_economic": float(reward_term_economic),
                "relative_reference_enabled": bool(relative_reference_enabled),
                "relative_reference_policy": str(relative_reference_policy),
                "relative_reference_normalize": bool(relative_reference_normalize),
                "relative_reference_eps": float(relative_reference_eps),
                "relative_reference_weight": float(relative_reference_weight),
                "reward_term_reference": float(reward_term_reference),
                "reward_term_reference_delta_raw": float(reward_term_reference_delta_raw),
                "reward_term_reference_delta_scaled": float(reward_term_reference_delta_scaled),
                "reward_term_reference_denom": float(reward_term_reference_denom),
                "reference_reward_term_economic": float(reference_reward_term_economic),
                "reference_reward_term_pymgrid_proxy": float(reference_reward_term_pymgrid_proxy),
                "reference_grid_import": float(reference_grid_import),
                "reference_grid_export": float(reference_grid_export),
                "reference_battery_action": float(reference_battery_action),
                "reference_grid_action_requested": float(reference_grid_action_requested),
                "reference_grid_action": float(reference_grid_action),
                "reference_cost_economic": float(reference_cost_economic),
                "reference_loss_load_energy_step": float(reference_loss_load_energy),
                "reference_overgeneration_energy_step": float(reference_overgeneration_energy),
                "reference_internal_energy_delta": float(reference_internal_energy_delta),
                "reference_cycle_cost_step": float(reference_cycle_cost_step),
                "reference_wear_cost_step": float(reference_wear_cost_step),
                "reference_co2_production_step": float(reference_co2_production_step),
                "reference_co2_cost_step": float(reference_co2_cost_step),
                "reward_term_soe_violation": float(reward_term_soe_violation),
                "reward_term_action_violation": float(reward_term_action_violation),
                "reward_term_action_smoothness": float(reward_term_action_smoothness),
                "reward_term_action_sign_change": float(reward_term_action_sign_change),
                "reward_term_micro_throughput": float(reward_term_micro_throughput),
                "reward_term_soe_boundary": float(reward_term_soe_boundary),
                "reward_term_cyclic_aging": float(reward_term_cyclic_aging),
                "reward_term_calendar_aging": float(reward_term_calendar_aging),
                "reward_term_wear_cost": float(reward_term_wear_cost),
                "reward_term_ssr": float(reward_term_ssr),
                "cost_economic": float(cost_economic),
                "co2_production_step": float(co2_production_step),
                "co2_per_kwh_step": float(co2_per_kwh_step),
                "co2_cost_step": float(co2_cost_step),
                "wear_cost": float(wear_cost),
                "load_kwh": float(load_kwh),
                "pv_kwh": float(pv_kwh),
                "soe": soe,
                "soh": float(current_soh),
                "soh_delta": float(prev_soh - current_soh),
                "cyclic_aging": float(cyclic_aging),
                "calendar_aging": float(calendar_aging),
                "soe_violation": float(soe_violation),
                "soe_boundary_penalty": float(soe_boundary_penalty),
                "action_violation": float(action_violation),
                "action_smoothness": float(action_smoothness),
                "micro_throughput": float(micro_throughput),
                "action_violation_flag": float(action_violation_flag),
                "self_sufficiency_ratio": float(ssr),
                "battery_action_requested": float(requested_batt),
                "battery_action_after_policy": float(requested_batt_after_policy),
                "battery_action_policy_was_clipped": bool(batt_policy["was_clipped"]),
                "grid_charge_blocked_by_policy_kwh": float(batt_policy["grid_charge_blocked_kwh"]),
                "battery_export_blocked_by_policy_kwh": float(
                    batt_policy["battery_export_blocked_kwh"]
                ),
                "battery_action_clipped": float(battery_action),
                "battery_action_was_clipped": bool(batt_clipped),
                "battery_action_actual": float(batt_actual),
                "battery_discharge": float(batt_discharge),
                "battery_charge": float(batt_charge),
                "grid_action_requested": float(requested_grid),
                "grid_action_clipped": float(grid_action),
                "grid_action_was_clipped": bool(grid_clipped),
                "grid_import": float(grid_import),
                "grid_export": float(grid_export),
                "net_load": float(net_load),
                "price_buy": float(price_buy),
                "price_sell": float(price_sell),
                "battery_overshoots": int(getattr(self.battery_module, "num_overshoots", 0)),
            }
        )
        if self._action_norm_enabled:
            info["battery_action_requested_norm"] = float(requested_batt_norm)

        self._last_soh = current_soh

        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def _apply_battery_action_policy_constraints(
        self,
        requested: float,
        net_load: float,
    ) -> Tuple[float, Dict[str, float]]:
        clipped = float(requested)
        grid_charge_blocked_kwh = 0.0
        battery_export_blocked_kwh = 0.0

        if not self._allow_grid_charge_for_battery and clipped < 0.0:
            max_charge_from_pv_surplus = max(0.0, -float(net_load))
            min_battery_action = -max_charge_from_pv_surplus
            if clipped < min_battery_action:
                grid_charge_blocked_kwh = float(min_battery_action - clipped)
                clipped = min_battery_action

        if not self._allow_battery_export_to_grid and clipped > 0.0:
            max_discharge_for_local_load = max(0.0, float(net_load))
            if clipped > max_discharge_for_local_load:
                battery_export_blocked_kwh = float(clipped - max_discharge_for_local_load)
                clipped = max_discharge_for_local_load

        return float(clipped), {
            "was_clipped": not np.isclose(requested, clipped, atol=1e-9),
            "grid_charge_blocked_kwh": float(grid_charge_blocked_kwh),
            "battery_export_blocked_kwh": float(battery_export_blocked_kwh),
        }

    def _clip_battery_action(self, requested: float) -> Tuple[float, bool]:
        battery = self.battery_module
        soc_min = float(np.clip(getattr(battery, "min_soc", 0.0), 0.0, 1.0))
        soc_max = float(np.clip(getattr(battery, "max_soc", 1.0), soc_min, 1.0))
        max_capacity = float(getattr(battery, "max_capacity", 0.0))
        current_charge = float(getattr(battery, "current_charge", 0.0))

        e_min = soc_min * max_capacity
        e_max = soc_max * max_capacity
        max_discharge_internal = min(
            float(getattr(battery, "max_discharge", 0.0)),
            max(0.0, current_charge - e_min),
        )
        max_charge_internal = min(
            float(getattr(battery, "max_charge", 0.0)),
            max(0.0, e_max - current_charge),
        )

        if hasattr(battery, "model_transition"):
            discharge_soc_limit = max(0.0, float(battery.model_transition(max_discharge_internal)))
            charge_soc_limit = max(0.0, float(-battery.model_transition(-max_charge_internal)))
        else:
            discharge_soc_limit = max(0.0, max_discharge_internal)
            charge_soc_limit = max(0.0, max_charge_internal)

        max_discharge = max(
            0.0,
            min(float(getattr(battery, "max_production", discharge_soc_limit)), discharge_soc_limit),
        )
        max_charge = max(
            0.0,
            min(float(getattr(battery, "max_consumption", charge_soc_limit)), charge_soc_limit),
        )

        if requested >= 0:
            clipped = min(requested, max_discharge)
        else:
            clipped = max(requested, -max_charge)

        clipped_flag = not np.isclose(requested, clipped, atol=1e-9)
        return float(clipped), clipped_flag

    def _prime_transition_model(self) -> None:
        transition_model = getattr(self.battery_module, "battery_transition_model", None)
        if transition_model is None:
            return
        if getattr(transition_model, "v_prev", None) is not None:
            return
        try:
            soc = float(self.battery_module.soc)
        except (TypeError, ValueError):
            soc = 0.0
        try:
            transition_model.soc = soc
            transition_model.soe = soc
        except Exception:
            return
        if hasattr(transition_model, "_interp_voc_r0"):
            temperature_c = float(getattr(transition_model, "temperature_c", 25.0))
            soh = float(getattr(transition_model, "soh", 1.0) or 1.0)
            try:
                voc, _ = transition_model._interp_voc_r0(soc, temperature_c, soh)
                transition_model.v_prev = voc
            except Exception:
                return

    def _clip_grid_action(self, requested: float) -> Tuple[float, bool]:
        grid = self.grid_module
        max_import = max(0.0, float(grid.max_production))
        max_export = max(0.0, float(grid.max_consumption))

        clipped = min(max(requested, -max_export), max_import)
        clipped_flag = not np.isclose(requested, clipped, atol=1e-9)
        return float(clipped), clipped_flag

    def _get_battery_flow(self) -> Tuple[float, float, float]:
        last = self._get_last_log_entry(self.battery_module)
        discharge = float(last.get("discharge_amount", 0.0))
        charge = float(last.get("charge_amount", 0.0))
        return discharge - charge, discharge, charge

    def _get_grid_flow(self) -> Tuple[float, float]:
        last = self._get_last_log_entry(self.grid_module)
        grid_import = float(last.get("grid_import", 0.0))
        grid_export = float(last.get("grid_export", 0.0))
        return grid_import, grid_export

    @staticmethod
    def _get_last_log_entry(module) -> Dict[str, float]:
        data = {}
        try:
            data = module.log_dict()
        except Exception:
            try:
                data = module.logger.to_dict()
            except Exception:
                return {}

        last = {}
        for key, values in data.items():
            if not values:
                continue
            try:
                last[key] = values[-1]
            except Exception:
                continue
        return last

    @staticmethod
    def _extract_battery_step_info(step_info: Any) -> Dict[str, Any]:
        if not isinstance(step_info, dict):
            return {}
        battery_entries = step_info.get("battery")
        if isinstance(battery_entries, list):
            for entry in battery_entries:
                if isinstance(entry, dict):
                    return entry
        return {}

    def _self_sufficiency_ratio(self, load_kwh: float, pv_kwh: float, p_batt: float) -> float:
        if load_kwh <= 0:
            return 1.0
        if load_kwh > pv_kwh:
            p_pv_2_load = min(load_kwh, pv_kwh)
            p_batt_2_load = min(max(p_batt, 0.0), load_kwh - p_pv_2_load)
        else:
            p_pv_2_load = load_kwh
            p_batt_2_load = 0.0
        return float((p_pv_2_load + p_batt_2_load) / load_kwh)

    def _build_obs(self) -> np.ndarray:
        step_idx = min(int(self.microgrid.current_step), self.max_steps - 1)

        load_kwh = float(self.load_series[step_idx])
        pv_kwh = float(self.pv_series[step_idx])
        soe = float(self.battery_module.soc)
        obs = [load_kwh, pv_kwh, soe, float(self._get_soh())]
        if self.include_battery_power_limits_in_obs:
            obs.append(float(self.battery_module.max_production))
            obs.append(float(self.battery_module.max_consumption))

        if self.include_time_feature:
            timestamp = self.timestamps_local.iloc[step_idx]
            hour = float(timestamp.hour) + float(timestamp.minute) / 60.0
            day_of_week = float(timestamp.dayofweek)
            obs.extend([
                math.sin(2 * math.pi * hour / 24.0),
                math.cos(2 * math.pi * hour / 24.0),
                math.sin(2 * math.pi * day_of_week / 7.0),
                math.cos(2 * math.pi * day_of_week / 7.0),
            ])

        if self.include_price:
            obs.append(float(self.price_buy[step_idx]))
            obs.append(float(self.price_sell[step_idx]))

        if self.forecast_price and self.num_forecast > 0:
            for j in range(1, self.num_forecast + 1):
                idx = min(step_idx + j, len(self.price_buy) - 1)
                obs.append(float(self.price_buy[idx]))
                obs.append(float(self.price_sell[idx]))

        if self.forecast_future and self.num_forecast > 0:
            for j in range(1, self.num_forecast + 1):
                idx = min(step_idx + j, len(self.load_series) - 1)
                future_load = float(self.load_series[idx])
                future_pv = float(self.pv_series[idx])
                if self.preprocessing == "simple":
                    obs.extend([future_load, future_pv])
                elif self.preprocessing == "net-load":
                    obs.append(future_load - future_pv)

        return np.array(obs, dtype=np.float32)

    def _get_soh(self) -> float:
        transition_model = getattr(self.battery_module, "battery_transition_model", None)
        if transition_model is None:
            return 1.0
        return float(getattr(transition_model, "soh", 1.0) or 1.0)

    def get_simulation_log(self):
        microgrid_df, log = self.simulator.get_simulation_log(self.microgrid)
        battery_module = self.battery_module
        bms_manager = getattr(battery_module, "battery_bms_manager", None)
        transition_model = getattr(battery_module, "battery_transition_model", None)
        if bms_manager is not None and hasattr(bms_manager, "get_transition_history"):
            history = bms_manager.get_transition_history()
        elif transition_model is not None and hasattr(transition_model, "get_transition_history"):
            history = transition_model.get_transition_history()
        else:
            history = []
        eta = [entry.get("efficiency", np.nan) for entry in history]
        soh_hist = [entry.get("soh", np.nan) for entry in history]

        start = self._start_step
        end = start + len(microgrid_df)

        additional_columns = {
            ("datetime", 0, "timestamp"): self.timestamps.to_numpy()[start:end],
            ("pv", 0, "pv_prod_input"): self.pv_series[start:end],
            ("load", 0, "consumption_input"): self.load_series[start:end],
            ("battery", 0, "eta"): np.asarray(eta)[: len(microgrid_df)],
            ("battery", 0, "soh"): np.asarray(soh_hist)[: len(microgrid_df)],
            ("price", 0, "price_buy"): self.price_buy[start:end],
            ("price", 0, "price_sell"): self.price_sell[start:end],
        }
        microgrid_df = add_module_columns(microgrid_df, additional_columns)
        microgrid_df = add_grid_cost_breakdown_columns(microgrid_df)
        return microgrid_df, log


class EMS_RL_Agent:
    def __init__(self, config: Dict[str, Any], device: str = "cpu", tensorboard_log: Optional[str] = None):
        self.config = config
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.model = None

    @staticmethod
    def _normalize_train_freq(raw_train_freq: Any) -> Any:
        if isinstance(raw_train_freq, (list, tuple)) and len(raw_train_freq) == 2:
            return (int(raw_train_freq[0]), str(raw_train_freq[1]))
        return int(raw_train_freq)

    @staticmethod
    def _normalize_policy_net(raw_policy_net: Any, default: Optional[list] = None) -> list:
        if default is None:
            default = [256, 256]
        if isinstance(raw_policy_net, (list, tuple)) and len(raw_policy_net) > 0:
            try:
                return [int(x) for x in raw_policy_net]
            except (TypeError, ValueError):
                pass
        return list(default)

    def _build_ppo_model(self, env: gym.Env, rl_cfg: Dict[str, Any]) -> PPO:
        ppo_cfg = rl_cfg.get("ppo", {}) or {}
        ent_coef_raw = ppo_cfg.get("ent_coef", 0.01)
        if isinstance(ent_coef_raw, (list, tuple)) and len(ent_coef_raw) == 1:
            ent_coef_raw = ent_coef_raw[0]
        try:
            ent_coef_value = float(ent_coef_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid ppo.ent_coef value: {ent_coef_raw!r}") from None
        normalize_advantage = bool(ppo_cfg.get("normalize_advantage", True))
        policy_net = self._normalize_policy_net(ppo_cfg.get("policy_net", [256, 256]))
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
            net_arch=dict(pi=policy_net, vf=policy_net),
        )

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 1024),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 5),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef_value,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
            tensorboard_log=self.tensorboard_log,
            seed=rl_cfg.get("seed", None),
        )
        return self.model

    def _build_sac_model(self, env: gym.Env, rl_cfg: Dict[str, Any]) -> SAC:
        sac_cfg = rl_cfg.get("sac", {}) or {}
        ent_coef_raw = sac_cfg.get("ent_coef", "auto")
        ent_coef_value: Any
        if isinstance(ent_coef_raw, str):
            ent_coef_value = ent_coef_raw
        else:
            try:
                ent_coef_value = float(ent_coef_raw)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid sac.ent_coef value: {ent_coef_raw!r}") from None
        target_entropy_raw = sac_cfg.get("target_entropy", "auto")
        target_entropy_value: Any
        if isinstance(target_entropy_raw, str):
            target_entropy_value = target_entropy_raw
        else:
            try:
                target_entropy_value = float(target_entropy_raw)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid sac.target_entropy value: {target_entropy_raw!r}") from None
        policy_net = self._normalize_policy_net(sac_cfg.get("policy_net", [256, 256]))
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=policy_net, qf=policy_net),
        )
        train_freq = self._normalize_train_freq(sac_cfg.get("train_freq", 1))
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=int(sac_cfg.get("buffer_size", 100000)),
            learning_starts=int(sac_cfg.get("learning_starts", 1000)),
            batch_size=int(sac_cfg.get("batch_size", 256)),
            tau=float(sac_cfg.get("tau", 0.005)),
            gamma=float(sac_cfg.get("gamma", 0.99)),
            train_freq=train_freq,
            gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
            ent_coef=ent_coef_value,
            target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
            target_entropy=target_entropy_value,
            use_sde=bool(sac_cfg.get("use_sde", False)),
            sde_sample_freq=int(sac_cfg.get("sde_sample_freq", -1)),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
            tensorboard_log=self.tensorboard_log,
            seed=rl_cfg.get("seed", None),
        )
        return model

    def build_model(self, env: gym.Env) -> BaseAlgorithm:
        rl_cfg = self.config["rl"]
        algo = str(rl_cfg.get("algorithm", "PPO")).upper()
        if algo == "PPO":
            self.model = self._build_ppo_model(env, rl_cfg)
        elif algo == "SAC":
            self.model = self._build_sac_model(env, rl_cfg)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: PPO, SAC.")
        return self.model

    def load(self, model_path: str, env: Optional[gym.Env] = None) -> BaseAlgorithm:
        path = Path(model_path)
        if path.is_dir():
            final_model = path / "model_final.zip"
            if final_model.exists():
                path = final_model
            else:
                candidates = sorted(path.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    raise FileNotFoundError(f"No .zip model found in directory: {path}")
                path = candidates[0]
        algo = str((self.config.get("rl", {}) or {}).get("algorithm", "PPO")).upper()
        if algo == "PPO":
            self.model = PPO.load(str(path), env=env, device=self.device)
        elif algo == "SAC":
            self.model = SAC.load(str(path), env=env, device=self.device)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: PPO, SAC.")
        return self.model

    def save(self, model_path: str) -> None:
        if self.model is None:
            raise ValueError("Model not initialized.")
        self.model.save(model_path)

    def predict_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def make_vec_env(self, config_path: str, random_start: bool, start_step_default: Optional[int] = None) -> VecEnv:
        rl_cfg = self.config["rl"]
        num_envs = max(1, int(rl_cfg.get("num_envs", 1)))
        vec_env_type = _resolve_vec_env_type(rl_cfg, num_envs)
        start_method = rl_cfg.get("vec_env_start_method")
        if isinstance(start_method, str):
            start_method = start_method.strip() or None

        base_seed = rl_cfg.get("seed", None)
        env_fns = [
            _make_offline_env_factory(
                config_path=config_path,
                config=self.config,
                random_start=random_start,
                seed=_resolve_worker_seed(base_seed, env_idx),
                start_step_default=start_step_default,
            )
            for env_idx in range(num_envs)
        ]

        if vec_env_type == "dummy":
            return DummyVecEnv(env_fns)

        subproc_kwargs = {}
        if start_method is not None:
            subproc_kwargs["start_method"] = str(start_method)

        return SubprocVecEnv(env_fns, **subproc_kwargs)



