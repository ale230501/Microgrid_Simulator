from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class MovingAverage:
    def __init__(self, window: int = 20):
        self.window = max(1, int(window))
        self._values = deque(maxlen=self.window)

    def update(self, value: float) -> float:
        value = _safe_float(value)
        self._values.append(value)
        return float(sum(self._values) / len(self._values))


class EpisodeMetricsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.length = 0
        self.episode_return = 0.0
        self.episode_return_normalized = 0.0
        self.cost_economic_sum = 0.0
        self.bad_logic_penalty_sum = 0.0
        self.cyclic_aging_sum = 0.0
        self.calendar_aging_sum = 0.0
        self.wear_cost_sum = 0.0
        self.soe_violation_sum = 0.0
        self.action_violation_sum = 0.0
        self.action_smoothness_sum = 0.0
        self.micro_throughput_sum = 0.0
        self.self_sufficiency_ratio_sum = 0.0
        self.battery_overshoots_end = 0

    def update(self, reward: float, info: Optional[dict]) -> None:
        if not isinstance(info, dict):
            info = {}
        reward_raw = _safe_float(info.get("reward", reward))
        reward_norm = _safe_float(info.get("reward_normalized", reward_raw))
        self.episode_return += reward_raw
        self.episode_return_normalized += reward_norm
        self.length += 1

        self.cost_economic_sum += _safe_float(info.get("cost_economic", 0.0))
        self.bad_logic_penalty_sum += _safe_float(info.get("bad_logic_penalty", 0.0))
        self.cyclic_aging_sum += _safe_float(info.get("cyclic_aging", 0.0))
        self.calendar_aging_sum += _safe_float(info.get("calendar_aging", 0.0))
        self.wear_cost_sum += _safe_float(info.get("wear_cost", 0.0))

        self.soe_violation_sum += _safe_float(info.get("soe_violation", info.get("soc_violation", 0.0)))
        action_violation_flag = info.get("action_violation_flag", info.get("action_violation", 0.0))
        self.action_violation_sum += _safe_float(action_violation_flag)
        self.action_smoothness_sum += _safe_float(info.get("action_smoothness", 0.0))
        self.micro_throughput_sum += _safe_float(info.get("micro_throughput", 0.0))

        self.self_sufficiency_ratio_sum += _safe_float(info.get("self_sufficiency_ratio", 0.0))
        if "battery_overshoots" in info:
            try:
                self.battery_overshoots_end = int(info.get("battery_overshoots") or 0)
            except (TypeError, ValueError):
                self.battery_overshoots_end = 0

    def summarize(self) -> dict:
        if self.length <= 0:
            return {
                "episode_return": 0.0,
                "episode_return_normalized": 0.0,
                "episode_reward_mean": 0.0,
                "episode_reward_normalized_mean": 0.0,
                "episode_length": 0,
                "cost_economic_sum": 0.0,
                "bad_logic_penalty_sum": 0.0,
                "soe_violation_rate": 0.0,
                "action_violation_rate": 0.0,
                "action_smoothness_mean": 0.0,
                "micro_throughput_mean": 0.0,
                "cyclic_aging_sum": 0.0,
                "calendar_aging_sum": 0.0,
                "wear_cost_sum": 0.0,
                "self_sufficiency_ratio_mean": 0.0,
                "battery_overshoots_end": 0,
            }
        length = float(self.length)
        return {
            "episode_return": self.episode_return,
            "episode_return_normalized": self.episode_return_normalized,
            "episode_reward_mean": self.episode_return / length,
            "episode_reward_normalized_mean": self.episode_return_normalized / length,
            "episode_length": int(self.length),
            "cost_economic_sum": self.cost_economic_sum,
            "bad_logic_penalty_sum": self.bad_logic_penalty_sum,
            "soe_violation_rate": self.soe_violation_sum / length,
            "action_violation_rate": self.action_violation_sum / length,
            "action_smoothness_mean": self.action_smoothness_sum / length,
            "micro_throughput_mean": self.micro_throughput_sum / length,
            "cyclic_aging_sum": self.cyclic_aging_sum,
            "calendar_aging_sum": self.calendar_aging_sum,
            "wear_cost_sum": self.wear_cost_sum,
            "self_sufficiency_ratio_mean": self.self_sufficiency_ratio_sum / length,
            "battery_overshoots_end": int(self.battery_overshoots_end),
        }


class EpisodeSeriesBuffer:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.steps = []
        self.series = {
            "reward": [],
            "reward_normalized": [],
            "cost_economic": [],
            "soe_violation": [],
            "action_violation": [],
            "action_smoothness": [],
            "micro_throughput": [],
            "bad_logic_penalty": [],
            "cyclic_aging": [],
            "calendar_aging": [],
            "self_sufficiency_ratio": [],
            "reward_term_economic": [],
            "reward_term_soe_violation": [],
            "reward_term_action_violation": [],
            "reward_term_action_smoothness": [],
            "reward_term_micro_throughput": [],
            "reward_term_bad_logic": [],
            "reward_term_cyclic_aging": [],
            "reward_term_calendar_aging": [],
            "reward_term_wear_cost": [],
            "reward_term_ssr": [],
            "price_buy": [],
            "price_sell": [],
            "soe": [],
            "pv_kwh": [],
            "load_kwh": [],
            "battery_charge": [],
            "battery_discharge": [],
            "grid_import": [],
            "grid_export": [],
            "wear_cost": [],
        }

    def update(self, info: Optional[dict]) -> None:
        if not isinstance(info, dict):
            return
        step_value = info.get("step_idx", len(self.steps))
        self.steps.append(int(step_value))
        for key, values in self.series.items():
            values.append(_safe_float(info.get(key), np.nan))

    def as_arrays(self, step_downsample: int = 1) -> dict:
        step = max(1, int(step_downsample))
        idx = slice(None, None, step)
        return {
            "steps": np.asarray(self.steps, dtype=float)[idx],
            **{key: np.asarray(values, dtype=float)[idx] for key, values in self.series.items()},
        }


class EpisodeBehaviouralBuffer:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.steps = []
        self.series = {
            "load_kwh": [],
            "pv_kwh": [],
            "soe": [],
            "soh": [],
            "soh_delta": [],
            "net_load": [],
            "price_buy": [],
            "price_sell": [],
            "battery_action_clipped": [],
            "battery_action_actual": [],
            "battery_charge": [],
            "battery_discharge": [],
            "grid_import": [],
            "grid_export": [],
            "cost_economic": [],
            "wear_cost": [],
            "bad_logic": [],
            "battery_action_was_clipped": [],
            "grid_action_was_clipped": [],
        }

    def update(self, info: Optional[dict]) -> None:
        if not isinstance(info, dict):
            return
        step_value = info.get("step_idx", len(self.steps))
        self.steps.append(int(step_value))
        for key, values in self.series.items():
            values.append(_safe_float(info.get(key), np.nan))

    def as_arrays(self, step_downsample: int = 1) -> dict:
        step = max(1, int(step_downsample))
        idx = slice(None, None, step)
        return {
            "steps": np.asarray(self.steps, dtype=float)[idx],
            **{key: np.asarray(values, dtype=float)[idx] for key, values in self.series.items()},
        }


class TensorboardLogger:
    def __init__(
        self,
        log_dir: Path,
        step_downsample: int = 1,
        include_price: bool = True,
        price_bands: Optional[dict] = None,
        enable_figures: bool = True,
        enable_histograms: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step_downsample = max(1, int(step_downsample))
        self.include_price = bool(include_price)
        self.enable_figures = bool(enable_figures)
        self.enable_histograms = bool(enable_histograms)
        self._price_pair_to_band = {}
        if isinstance(price_bands, dict):
            for band_name, band_cfg in price_bands.items():
                if not isinstance(band_cfg, dict):
                    continue
                buy = band_cfg.get("buy")
                sell = band_cfg.get("sell")
                try:
                    buy_f = float(buy)
                    sell_f = float(sell)
                except (TypeError, ValueError):
                    continue
                key = (round(buy_f, 6), round(sell_f, 6))
                self._price_pair_to_band.setdefault(key, str(band_name))
        self._overall_soe_bins = np.linspace(0.0, 1.0, 11)
        self._overall_band_action = {}
        self._overall_band_soe_sum = {}
        self._overall_band_soe_count = {}
        self._overall_action_sample_max = 50000

    def log_step(self, info: dict, step: int, prefix: str) -> None:
        if not isinstance(info, dict):
            return
        if int(step) % self.step_downsample != 0:
            return
        step_items = [
            ("reward/total_raw", "reward"),
            ("reward/total_norm", "reward_normalized"),
            ("reward/base", "base_reward"),
            ("reward/components/economic_cost", "cost_economic"),
            ("reward/components/soe_violation", "soe_violation"),
            ("reward/components/soe_boundary_penalty", "soe_boundary_penalty"),
            ("reward/components/action_violation", "action_violation"),
            ("reward/components/action_smoothness", "action_smoothness"),
            ("reward/components/micro_throughput", "micro_throughput"),
            ("reward/components/bad_logic_penalty", "bad_logic_penalty"),
            ("reward/components/cyclic_aging", "cyclic_aging"),
            ("reward/components/calendar_aging", "calendar_aging"),
            ("reward/components/wear_cost", "wear_cost"),
            ("reward/components/ssr", "self_sufficiency_ratio"),
            ("reward/terms/economic", "reward_term_economic"),
            ("reward/terms/soe_violation", "reward_term_soe_violation"),
            ("reward/terms/soe_boundary", "reward_term_soe_boundary"),
            ("reward/terms/action_violation", "reward_term_action_violation"),
            ("reward/terms/action_smoothness", "reward_term_action_smoothness"),
            ("reward/terms/micro_throughput", "reward_term_micro_throughput"),
            ("reward/terms/bad_logic", "reward_term_bad_logic"),
            ("reward/terms/cyclic_aging", "reward_term_cyclic_aging"),
            ("reward/terms/calendar_aging", "reward_term_calendar_aging"),
            ("reward/terms/wear_cost", "reward_term_wear_cost"),
            ("reward/terms/ssr", "reward_term_ssr"),
            ("soe", "soe"),
            ("soh", "soh"),
            ("soh_delta", "soh_delta"),
            ("battery_action_requested", "battery_action_requested"),
            ("battery_action_clipped", "battery_action_clipped"),
            ("battery_charge", "battery_charge"),
            ("battery_discharge", "battery_discharge"),
            ("battery_overshoots", "battery_overshoots"),
            ("grid_import", "grid_import"),
            ("grid_export", "grid_export"),
            ("net_load", "net_load"),
            ("load_kwh", "load_kwh"),
            ("pv_kwh", "pv_kwh"),
            ("wear_cost", "wear_cost"),
            ("flags/bad_logic", "bad_logic"),
            ("flags/battery_action_was_clipped", "battery_action_was_clipped"),
            ("flags/grid_action_was_clipped", "grid_action_was_clipped"),
        ]
        if self.include_price:
            step_items.extend(
                [
                    ("price_buy", "price_buy"),
                    ("price_sell", "price_sell"),
                ]
            )
        for tag, key in step_items:
            if key in info:
                self.writer.add_scalar(f"{prefix}/{tag}", _safe_float(info.get(key)), int(step))

    def log_episode(self, episode_metrics: dict, episode_idx: int, prefix: str) -> None:
        if not isinstance(episode_metrics, dict):
            return
        for key, value in episode_metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", _safe_float(value), int(episode_idx))

    def log_reasoning_figure(self, buffer: EpisodeSeriesBuffer, episode_idx: int, prefix: str) -> None:
        if not self.enable_figures:
            return
        if buffer is None:
            return
        data = buffer.as_arrays(step_downsample=self.step_downsample)
        steps = data.get("steps")
        if steps is None or len(steps) == 0:
            return

        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(12, 9))

        axes[0].plot(steps, data["price_buy"], label="Price Buy (eur/kWh)", linewidth=1.6)
        axes[0].plot(steps, data["price_sell"], label="Price Sell (eur/kWh)", linewidth=1.6)
        axes[0].set_ylabel("eur/kWh")
        axes[0].set_title("RL Reasoning Overview")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.35)

        axes[1].plot(steps, data["soe"] * 100.0, color="tab:blue", label="SoE (%)", linewidth=1.6)
        axes[1].set_ylabel("SoE [%]")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.35)

        axes[2].plot(steps, data["pv_kwh"], label="PV (kWh)", linewidth=1.6)
        axes[2].plot(steps, data["load_kwh"], label="Load (kWh)", linewidth=1.6)
        axes[2].set_ylabel("Energy [kWh]")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle="--", alpha=0.35)

        axes[3].bar(steps, data["battery_charge"], label="Charge (kWh)", color="tab:green", alpha=0.5)
        axes[3].bar(steps, -data["battery_discharge"], label="Discharge (kWh)", color="tab:red", alpha=0.5)
        axes[3].axhline(0.0, color="black", linewidth=0.8)
        axes[3].set_ylabel("Battery [kWh]")
        axes[3].legend(loc="upper left")
        axes[3].grid(True, linestyle="--", alpha=0.35)

        axes[4].bar(steps, data["grid_import"], label="Import (kWh)", color="tab:blue", alpha=0.5)
        axes[4].bar(steps, -data["grid_export"], label="Export (kWh)", color="tab:orange", alpha=0.5)
        axes[4].axhline(0.0, color="black", linewidth=0.8)
        axes[4].set_ylabel("Grid [kWh]")
        axes[4].set_xlabel("Step")
        axes[4].legend(loc="upper left")
        axes[4].grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        self.writer.add_figure(f"{prefix}/reasoning", fig, global_step=int(episode_idx))

        if "wear_cost" in data:
            wear_fig, wear_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 2.6))
            wear_ax.plot(steps, data["wear_cost"], label="Wear cost (eur/step)", linewidth=1.6, color="tab:green")
            wear_ax.set_ylabel("eur/step")
            wear_ax.set_xlabel("Step")
            wear_ax.legend(loc="upper left")
            wear_ax.grid(True, linestyle="--", alpha=0.35)
            wear_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/wear_cost", wear_fig, global_step=int(episode_idx))
            plt.close(wear_fig)
        plt.close(fig)

        penalty_items = [
            ("reward_term_economic", "Economic"),
            ("reward_term_soe_violation", "SoE violation"),
            ("reward_term_action_violation", "Action violation"),
            ("reward_term_bad_logic", "Bad logic"),
            ("reward_term_cyclic_aging", "Cyclic aging"),
            ("reward_term_calendar_aging", "Calendar aging"),
            ("reward_term_wear_cost", "Wear cost"),
        ]
        penalty_series = []
        penalty_labels = []
        for key, label in penalty_items:
            if key not in data:
                continue
            series = np.nan_to_num(data.get(key), nan=0.0)
            penalty_series.append(np.clip(-series, 0.0, None))
            penalty_labels.append(label)

        if penalty_series:
            penalty_total = np.sum(penalty_series, axis=0)
            penalty_fig, penalty_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
            penalty_ax.stackplot(steps, penalty_series, labels=penalty_labels, alpha=0.85)
            penalty_ax.plot(steps, penalty_total, color="black", linewidth=1.2, label="Total penalties")
            penalty_ax.set_title("Reward penalties (weighted)")
            penalty_ax.set_ylabel("Reward units/step")
            penalty_ax.set_xlabel("Step")
            penalty_ax.legend(loc="upper left", ncol=2)
            penalty_ax.grid(True, linestyle="--", alpha=0.35)
            penalty_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/reward_penalties_stack", penalty_fig, global_step=int(episode_idx))
            plt.close(penalty_fig)

            ssr_term = np.nan_to_num(data.get("reward_term_ssr"), nan=0.0)
            ssr_term = np.clip(ssr_term, 0.0, None)
            balance_fig, balance_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 2.8))
            balance_ax.stackplot(
                steps,
                [penalty_total, ssr_term],
                labels=["Penalties", "SSR reward"],
                alpha=0.85,
            )
            if "reward" in data:
                balance_ax.plot(steps, data["reward"], color="black", linewidth=1.2, label="Reward (raw)")
            balance_ax.axhline(0.0, color="black", linewidth=0.8)
            balance_ax.set_title("Reward balance (penalties vs SSR)")
            balance_ax.set_ylabel("Reward units/step")
            balance_ax.set_xlabel("Step")
            balance_ax.legend(loc="upper left")
            balance_ax.grid(True, linestyle="--", alpha=0.35)
            balance_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/reward_balance_stack", balance_fig, global_step=int(episode_idx))
            plt.close(balance_fig)

    def log_behavioural(self, buffer: EpisodeBehaviouralBuffer, episode_idx: int, prefix: str) -> None:
        if buffer is None:
            return
        data = buffer.as_arrays(step_downsample=self.step_downsample)
        steps = data.get("steps")
        if steps is None or len(steps) == 0:
            return

        def arr(name: str) -> np.ndarray:
            return np.asarray(data.get(name, []), dtype=float)

        load_kwh = arr("load_kwh")
        pv_kwh = arr("pv_kwh")
        soe = arr("soe")
        soh = arr("soh")
        net_load = arr("net_load")
        price_buy = arr("price_buy")
        price_sell = arr("price_sell")
        spread = price_buy - price_sell if price_buy.size and price_sell.size else np.asarray([], dtype=float)
        batt_action = arr("battery_action_actual")
        batt_action_clipped = arr("battery_action_clipped")
        batt_charge = arr("battery_charge")
        batt_discharge = arr("battery_discharge")
        grid_import = arr("grid_import")
        grid_export = arr("grid_export")
        cost_economic = arr("cost_economic")
        wear_cost = arr("wear_cost")
        bad_logic = arr("bad_logic")
        batt_clipped_flag = arr("battery_action_was_clipped")
        grid_clipped_flag = arr("grid_action_was_clipped")

        def safe_mean(x: np.ndarray) -> float:
            if x.size == 0:
                return 0.0
            return float(np.nanmean(x))

        def safe_min(x: np.ndarray) -> float:
            if x.size == 0:
                return 0.0
            return float(np.nanmin(x))

        def safe_max(x: np.ndarray) -> float:
            if x.size == 0:
                return 0.0
            return float(np.nanmax(x))

        battery_throughput = float(np.nansum(np.clip(batt_charge, 0.0, None) + np.clip(batt_discharge, 0.0, None)))
        grid_import_total = float(np.nansum(np.clip(grid_import, 0.0, None)))
        grid_export_total = float(np.nansum(np.clip(grid_export, 0.0, None)))
        cost_economic_sum = float(np.nansum(np.nan_to_num(cost_economic, nan=0.0)))
        wear_cost_sum = float(np.nansum(np.nan_to_num(wear_cost, nan=0.0)))
        batt_clip_rate = safe_mean((batt_clipped_flag > 0.5).astype(float))
        grid_clip_rate = safe_mean((grid_clipped_flag > 0.5).astype(float))
        bad_logic_rate = safe_mean((bad_logic > 0.5).astype(float))

        charge_while_deficit_rate = safe_mean(((net_load > 0.0) & (batt_action < 0.0)).astype(float))
        discharge_while_surplus_rate = safe_mean(((net_load < 0.0) & (batt_action > 0.0)).astype(float))

        self.writer.add_scalar(f"{prefix}/behavioural/battery/throughput_kwh", battery_throughput, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/grid/import_kwh", grid_import_total, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/grid/export_kwh", grid_export_total, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/cost/economic_sum", cost_economic_sum, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/cost/wear_sum", wear_cost_sum, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/flags/battery_clip_rate", batt_clip_rate, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/flags/grid_clip_rate", grid_clip_rate, int(episode_idx))
        self.writer.add_scalar(f"{prefix}/behavioural/flags/bad_logic_rate", bad_logic_rate, int(episode_idx))
        self.writer.add_scalar(
            f"{prefix}/behavioural/logic/charge_while_deficit_rate",
            charge_while_deficit_rate,
            int(episode_idx),
        )
        self.writer.add_scalar(
            f"{prefix}/behavioural/logic/discharge_while_surplus_rate",
            discharge_while_surplus_rate,
            int(episode_idx),
        )

        if soe.size:
            self.writer.add_scalar(f"{prefix}/behavioural/soe/mean", safe_mean(soe), int(episode_idx))
            self.writer.add_scalar(f"{prefix}/behavioural/soe/min", safe_min(soe), int(episode_idx))
            self.writer.add_scalar(f"{prefix}/behavioural/soe/max", safe_max(soe), int(episode_idx))
            if self.enable_histograms:
                self.writer.add_histogram(
                    f"{prefix}/behavioural/hist/soe",
                    soe,
                    global_step=int(episode_idx),
                )

        if soh.size:
            self.writer.add_scalar(f"{prefix}/behavioural/soh/start", float(soh[0]), int(episode_idx))
            self.writer.add_scalar(f"{prefix}/behavioural/soh/end", float(soh[-1]), int(episode_idx))
            self.writer.add_scalar(f"{prefix}/behavioural/soh/delta", float(soh[0] - soh[-1]), int(episode_idx))
            if self.enable_histograms:
                self.writer.add_histogram(
                    f"{prefix}/behavioural/hist/soh",
                    soh,
                    global_step=int(episode_idx),
                )

        if batt_action.size:
            self.writer.add_scalar(f"{prefix}/behavioural/action/mean", safe_mean(batt_action), int(episode_idx))
            self.writer.add_scalar(f"{prefix}/behavioural/action/mean_abs", safe_mean(np.abs(batt_action)), int(episode_idx))
            if self.enable_histograms:
                self.writer.add_histogram(
                    f"{prefix}/behavioural/hist/battery_action_actual",
                    batt_action,
                    global_step=int(episode_idx),
                )

        if batt_action_clipped.size and self.enable_histograms:
            self.writer.add_histogram(
                f"{prefix}/behavioural/hist/battery_action_clipped",
                batt_action_clipped,
                global_step=int(episode_idx),
            )

        if self.enable_histograms:
            if price_buy.size:
                self.writer.add_histogram(
                    f"{prefix}/behavioural/hist/price_buy",
                    price_buy,
                    global_step=int(episode_idx),
                )
            if price_sell.size:
                self.writer.add_histogram(
                    f"{prefix}/behavioural/hist/price_sell",
                    price_sell,
                    global_step=int(episode_idx),
                )

        if self.enable_figures and cost_economic.size and wear_cost.size:
            cum_cost = np.cumsum(np.nan_to_num(cost_economic, nan=0.0))
            cum_wear = np.cumsum(np.nan_to_num(wear_cost, nan=0.0))
            cum_total = cum_cost + cum_wear
            cost_fig, cost_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
            cost_ax.plot(steps, cum_cost, label="Cumulative economic cost", linewidth=1.6)
            cost_ax.plot(steps, cum_wear, label="Cumulative wear cost", linewidth=1.6)
            cost_ax.plot(steps, cum_total, label="Cumulative (economic + wear)", linewidth=1.8, color="black")
            cost_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            cost_ax.set_title("Cumulative costs (episode)")
            cost_ax.set_ylabel("Cost units (as defined in env)")
            cost_ax.set_xlabel("Step")
            cost_ax.legend(loc="upper left")
            cost_ax.grid(True, linestyle="--", alpha=0.35)
            cost_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/behavioural/costs_cumsum", cost_fig, global_step=int(episode_idx))
            plt.close(cost_fig)

        if self.enable_figures and load_kwh.size and pv_kwh.size and grid_import.size and grid_export.size and batt_charge.size and batt_discharge.size:
            n = min(load_kwh.size, pv_kwh.size, grid_import.size, grid_export.size, batt_charge.size, batt_discharge.size)
            l = np.nan_to_num(load_kwh[:n], nan=0.0)
            pv = np.nan_to_num(pv_kwh[:n], nan=0.0)
            gi = np.clip(np.nan_to_num(grid_import[:n], nan=0.0), 0.0, None)
            ge = np.clip(np.nan_to_num(grid_export[:n], nan=0.0), 0.0, None)
            bc = np.clip(np.nan_to_num(batt_charge[:n], nan=0.0), 0.0, None)
            bd = np.clip(np.nan_to_num(batt_discharge[:n], nan=0.0), 0.0, None)

            pv_to_load = np.minimum(pv, l)
            remaining_load = np.maximum(0.0, l - pv_to_load)
            batt_to_load = np.minimum(bd, remaining_load)
            remaining_load = np.maximum(0.0, remaining_load - batt_to_load)
            grid_to_load = np.minimum(gi, remaining_load)

            pv_surplus = np.maximum(0.0, pv - pv_to_load)
            pv_to_batt = np.minimum(bc, pv_surplus)
            pv_surplus = np.maximum(0.0, pv_surplus - pv_to_batt)
            pv_to_export = np.minimum(ge, pv_surplus)
            pv_curtail_est = np.maximum(0.0, pv_surplus - pv_to_export)
            grid_to_batt = np.maximum(0.0, bc - pv_to_batt)
            export_from_batt_est = np.maximum(0.0, ge - pv_to_export)

            balance_fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 5.2))
            axes[0].stackplot(
                steps[:n],
                [pv_to_load, batt_to_load, grid_to_load],
                labels=["PV→Load (est.)", "Batt→Load (est.)", "Grid→Load (est.)"],
                alpha=0.85,
            )
            axes[0].plot(steps[:n], l, color="black", linewidth=1.0, alpha=0.9, label="Load")
            axes[0].set_title("Estimated load supply decomposition")
            axes[0].set_ylabel("kWh/step")
            axes[0].legend(loc="upper left", ncol=2)
            axes[0].grid(True, linestyle="--", alpha=0.35)

            axes[1].stackplot(
                steps[:n],
                [pv_to_load, pv_to_batt, pv_to_export, pv_curtail_est],
                labels=["PV→Load (est.)", "PV→Batt (est.)", "PV→Export (est.)", "PV curtail (est.)"],
                alpha=0.85,
            )
            axes[1].plot(steps[:n], pv, color="black", linewidth=1.0, alpha=0.9, label="PV")
            axes[1].set_title("Estimated PV allocation (note: export split is approximate)")
            axes[1].set_ylabel("kWh/step")
            axes[1].set_xlabel("Step")
            axes[1].legend(loc="upper left", ncol=2)
            axes[1].grid(True, linestyle="--", alpha=0.35)

            balance_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/behavioural/energy_balance_stack", balance_fig, global_step=int(episode_idx))
            plt.close(balance_fig)

            self.writer.add_scalar(
                f"{prefix}/behavioural/energy/grid_to_batt_kwh_est",
                float(np.sum(grid_to_batt)),
                int(episode_idx),
            )
            self.writer.add_scalar(
                f"{prefix}/behavioural/energy/export_from_batt_kwh_est",
                float(np.sum(export_from_batt_est)),
                int(episode_idx),
            )

        if self.enable_figures and batt_action.size and price_buy.size:
            n = min(int(batt_action.size), int(price_buy.size))
            x = price_buy[:n]
            y = batt_action[:n]
            if n > 5000:
                idx = np.linspace(0, n - 1, num=5000, dtype=int)
                x = x[idx]
                y = y[idx]
            scatter_fig, scatter_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
            scatter_ax.scatter(x, y, s=8, alpha=0.25, edgecolors="none")
            scatter_ax.axhline(0.0, color="black", linewidth=0.8)
            scatter_ax.set_title("Battery action vs buy price (sampled)")
            scatter_ax.set_xlabel("Buy price")
            scatter_ax.set_ylabel("Battery action (actual)")
            scatter_ax.grid(True, linestyle="--", alpha=0.35)
            scatter_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/behavioural/action_vs_price_buy", scatter_fig, global_step=int(episode_idx))
            plt.close(scatter_fig)

        if self.enable_figures and batt_action.size and spread.size:
            n = min(int(batt_action.size), int(spread.size))
            x = spread[:n]
            y = batt_action[:n]
            if n > 5000:
                idx = np.linspace(0, n - 1, num=5000, dtype=int)
                x = x[idx]
                y = y[idx]
            scatter_fig, scatter_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
            scatter_ax.scatter(x, y, s=8, alpha=0.25, edgecolors="none")
            scatter_ax.axhline(0.0, color="black", linewidth=0.8)
            scatter_ax.set_title("Battery action vs price spread (buy - sell, sampled)")
            scatter_ax.set_xlabel("Price spread")
            scatter_ax.set_ylabel("Battery action (actual)")
            scatter_ax.grid(True, linestyle="--", alpha=0.35)
            scatter_fig.tight_layout()
            self.writer.add_figure(f"{prefix}/behavioural/action_vs_spread", scatter_fig, global_step=int(episode_idx))
            plt.close(scatter_fig)

        if self.enable_figures and batt_action.size and price_buy.size:
            rounded = np.round(price_buy.astype(float), 6)
            unique_prices = np.unique(rounded[~np.isnan(rounded)])
            if unique_prices.size > 1 and unique_prices.size <= 10:
                groups = []
                labels = []
                for p in unique_prices:
                    mask = rounded == p
                    vals = batt_action[mask]
                    vals = vals[~np.isnan(vals)]
                    if vals.size == 0:
                        continue
                    groups.append(vals)
                    labels.append(str(p))
                if groups:
                    box_fig, box_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
                    box_ax.boxplot(groups, labels=labels, showfliers=False)
                    box_ax.axhline(0.0, color="black", linewidth=0.8)
                    box_ax.set_title("Battery action distribution by buy price")
                    box_ax.set_xlabel("Buy price")
                    box_ax.set_ylabel("Battery action (actual)")
                    box_ax.grid(True, linestyle="--", alpha=0.35)
                    box_fig.tight_layout()
                    self.writer.add_figure(
                        f"{prefix}/behavioural/action_by_price_buy",
                        box_fig,
                        global_step=int(episode_idx),
                    )
                    plt.close(box_fig)

        band_labels = None
        if price_buy.size and price_sell.size and getattr(self, "_price_pair_to_band", None):
            n = min(int(price_buy.size), int(price_sell.size))
            pairs = np.stack(
                [np.round(price_buy[:n].astype(float), 6), np.round(price_sell[:n].astype(float), 6)],
                axis=1,
            )
            labels = []
            for buy_val, sell_val in pairs:
                if np.isnan(buy_val) or np.isnan(sell_val):
                    labels.append("unknown")
                    continue
                labels.append(self._price_pair_to_band.get((float(buy_val), float(sell_val)), "unknown"))
            band_labels = np.asarray(labels, dtype=object)

        if self.enable_figures and batt_action.size and band_labels is not None and band_labels.size:
            n = min(int(batt_action.size), int(band_labels.size))
            labels = band_labels[:n]
            values = batt_action[:n]
            unique = [lab for lab in np.unique(labels) if lab != "unknown"]
            if unique:
                order_pref = ["offpeak", "standard", "peak"]
                ordered = [u for u in order_pref if u in unique] + [u for u in unique if u not in order_pref]
                groups = []
                group_labels = []
                for lab in ordered:
                    vals = values[labels == lab]
                    vals = vals[~np.isnan(vals)]
                    if vals.size == 0:
                        continue
                    groups.append(vals)
                    group_labels.append(lab)
                if groups:
                    box_fig, box_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
                    box_ax.boxplot(groups, labels=group_labels, showfliers=False)
                    box_ax.axhline(0.0, color="black", linewidth=0.8)
                    box_ax.set_title("Battery action distribution by TOU band")
                    box_ax.set_xlabel("TOU band")
                    box_ax.set_ylabel("Battery action (actual)")
                    box_ax.grid(True, linestyle="--", alpha=0.35)
                    box_fig.tight_layout()
                    self.writer.add_figure(
                        f"{prefix}/behavioural/action_by_band",
                        box_fig,
                        global_step=int(episode_idx),
                    )
                    plt.close(box_fig)

                for lab in unique:
                    vals = values[labels == lab]
                    vals = vals[~np.isnan(vals)]
                    if vals.size == 0:
                        continue
                    store = self._overall_band_action.get(lab)
                    if store is None or not isinstance(store, deque):
                        store = deque(maxlen=self._overall_action_sample_max)
                        self._overall_band_action[lab] = store
                    store.extend(vals.tolist())

        if self.enable_figures and batt_action.size and soe.size and band_labels is not None and band_labels.size:
            n = min(int(batt_action.size), int(soe.size), int(band_labels.size))
            soe_n = soe[:n]
            act_n = batt_action[:n]
            labels_n = band_labels[:n]
            unique = [lab for lab in np.unique(labels_n) if lab != "unknown"]
            if unique:
                order_pref = ["offpeak", "standard", "peak"]
                bands = [u for u in order_pref if u in unique] + [u for u in unique if u not in order_pref]
                soe_bins = np.linspace(0.0, 1.0, 11)
                soe_idx = np.digitize(soe_n, soe_bins, right=False) - 1
                soe_idx = np.clip(soe_idx, 0, len(soe_bins) - 2)
                heat = np.full((len(bands), len(soe_bins) - 1), np.nan, dtype=float)
                for bi, band in enumerate(bands):
                    mask_band = labels_n == band
                    for si in range(len(soe_bins) - 1):
                        mask = mask_band & (soe_idx == si)
                        vals = act_n[mask]
                        vals = vals[~np.isnan(vals)]
                        if vals.size:
                            heat[bi, si] = float(np.mean(vals))

                if np.isfinite(heat).any():
                    vmax = float(np.nanmax(np.abs(heat)))
                    if vmax <= 0 or not np.isfinite(vmax):
                        vmax = 1.0
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.2))
                    im = ax.imshow(
                        heat,
                        aspect="auto",
                        interpolation="nearest",
                        cmap="coolwarm",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    ax.set_yticks(range(len(bands)))
                    ax.set_yticklabels(bands)
                    ax.set_xticks(range(len(soe_bins) - 1))
                    ax.set_xticklabels([f"{soe_bins[i]:.1f}-{soe_bins[i+1]:.1f}" for i in range(len(soe_bins) - 1)], rotation=45, ha="right")
                    ax.set_title("Mean battery action by TOU band and SoE bin")
                    ax.set_xlabel("SoE bin")
                    ax.set_ylabel("TOU band")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean action")
                    fig.tight_layout()
                    self.writer.add_figure(
                        f"{prefix}/behavioural/heatmap_soe_vs_band_action_mean",
                        fig,
                        global_step=int(episode_idx),
                    )
                    plt.close(fig)

                for bi, band in enumerate(bands):
                    mask_band = labels_n == band
                    if not np.any(mask_band):
                        continue
                    soe_idx = np.digitize(soe_n[mask_band], soe_bins, right=False) - 1
                    soe_idx = np.clip(soe_idx, 0, len(soe_bins) - 2)
                    sums = self._overall_band_soe_sum.setdefault(band, np.zeros(len(soe_bins) - 1, dtype=float))
                    counts = self._overall_band_soe_count.setdefault(band, np.zeros(len(soe_bins) - 1, dtype=float))
                    for si in range(len(soe_bins) - 1):
                        mask_bin = soe_idx == si
                        if not np.any(mask_bin):
                            continue
                        vals = act_n[mask_band][mask_bin]
                        vals = vals[~np.isnan(vals)]
                        if vals.size:
                            sums[si] += float(np.sum(vals))
                            counts[si] += float(vals.size)

        if self.enable_figures and self._overall_band_action:
            order_pref = ["offpeak", "standard", "peak"]
            unique = list(self._overall_band_action.keys())
            ordered = [u for u in order_pref if u in unique] + [u for u in unique if u not in order_pref]
            groups = []
            labels = []
            for lab in ordered:
                stored = self._overall_band_action.get(lab)
                if stored is None:
                    continue
                if isinstance(stored, deque):
                    vals = np.asarray(list(stored), dtype=float)
                else:
                    vals = np.asarray(stored, dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                groups.append(vals)
                labels.append(lab)
            if groups:
                box_fig, box_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.0))
                box_ax.boxplot(groups, labels=labels, showfliers=False)
                box_ax.axhline(0.0, color="black", linewidth=0.8)
                box_ax.set_title("Battery action distribution by TOU band (overall)")
                box_ax.set_xlabel("TOU band")
                box_ax.set_ylabel("Battery action (actual)")
                box_ax.grid(True, linestyle="--", alpha=0.35)
                box_fig.tight_layout()
                self.writer.add_figure(
                    f"{prefix}/behavioural/action_by_band_overall",
                    box_fig,
                    global_step=int(episode_idx),
                )
                plt.close(box_fig)

        if self.enable_figures and self._overall_band_soe_sum and self._overall_band_soe_count:
            order_pref = ["offpeak", "standard", "peak"]
            unique = list(self._overall_band_soe_sum.keys())
            bands = [u for u in order_pref if u in unique] + [u for u in unique if u not in order_pref]
            if bands:
                heat = np.full((len(bands), len(self._overall_soe_bins) - 1), np.nan, dtype=float)
                for bi, band in enumerate(bands):
                    sums = self._overall_band_soe_sum.get(band)
                    counts = self._overall_band_soe_count.get(band)
                    if sums is None or counts is None:
                        continue
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean_vals = np.where(counts > 0, sums / counts, np.nan)
                    heat[bi, :] = mean_vals
                if np.isfinite(heat).any():
                    vmax = float(np.nanmax(np.abs(heat)))
                    if vmax <= 0 or not np.isfinite(vmax):
                        vmax = 1.0
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.2))
                    im = ax.imshow(
                        heat,
                        aspect="auto",
                        interpolation="nearest",
                        cmap="coolwarm",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    ax.set_yticks(range(len(bands)))
                    ax.set_yticklabels(bands)
                    ax.set_xticks(range(len(self._overall_soe_bins) - 1))
                    ax.set_xticklabels(
                        [
                            f"{self._overall_soe_bins[i]:.1f}-{self._overall_soe_bins[i+1]:.1f}"
                            for i in range(len(self._overall_soe_bins) - 1)
                        ],
                        rotation=45,
                        ha="right",
                    )
                    ax.set_title("Mean battery action by TOU band and SoE bin (overall)")
                    ax.set_xlabel("SoE bin")
                    ax.set_ylabel("TOU band")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean action")
                    fig.tight_layout()
                    self.writer.add_figure(
                        f"{prefix}/behavioural/heatmap_soe_vs_band_action_mean_overall",
                        fig,
                        global_step=int(episode_idx),
                    )
                    plt.close(fig)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class TensorboardTrainCallback(BaseCallback):
    def __init__(
        self,
        log_dir: Path,
        step_downsample: int = 1,
        ma_window: int = 20,
        include_price: bool = True,
        reasoning_plot_every: int = 0,
        price_bands: Optional[dict] = None,
        enable_figures: bool = True,
        enable_histograms: bool = True,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.step_downsample = int(step_downsample)
        self.ma_window = int(ma_window)
        self.include_price = bool(include_price)
        self.reasoning_plot_every = max(0, int(reasoning_plot_every))
        self.price_bands = price_bands if isinstance(price_bands, dict) else None
        self.enable_figures = bool(enable_figures)
        self.enable_histograms = bool(enable_histograms)
        self._tb_logger = None
        self._episode_stats = []
        self._episode_counts = []
        self._episode_series = []
        self._behavioural_series = []
        self._return_ma = None

    def _on_training_start(self) -> None:
        self._tb_logger = TensorboardLogger(
            self.log_dir,
            step_downsample=self.step_downsample,
            include_price=self.include_price,
            price_bands=self.price_bands,
            enable_figures=self.enable_figures,
            enable_histograms=self.enable_histograms,
        )
        num_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_stats = [EpisodeMetricsAccumulator() for _ in range(num_envs)]
        self._episode_counts = [0 for _ in range(num_envs)]
        if self.reasoning_plot_every > 0:
            self._episode_series = [EpisodeSeriesBuffer() for _ in range(num_envs)]
        else:
            self._episode_series = []
        self._behavioural_series = [EpisodeBehaviouralBuffer() for _ in range(num_envs)]
        self._return_ma = MovingAverage(self.ma_window)

    def _on_training_end(self) -> None:
        if self._tb_logger is not None:
            self._tb_logger.flush()
            self._tb_logger.close()
            self._tb_logger = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        num_envs = len(self._episode_stats)
        step = int(self.num_timesteps)

        for env_idx in range(num_envs):
            info = infos[env_idx] if env_idx < len(infos) else {}
            reward = rewards[env_idx] if env_idx < len(rewards) else 0.0
            self._episode_stats[env_idx].update(reward, info)
            if self._episode_series:
                self._episode_series[env_idx].update(info)
            if self._behavioural_series:
                self._behavioural_series[env_idx].update(info)

            if env_idx == 0 and self._tb_logger is not None:
                self._tb_logger.log_step(info, step, prefix="train")

            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if done:
                self._episode_counts[env_idx] += 1
                metrics = self._episode_stats[env_idx].summarize()
                if env_idx == 0 and self._return_ma is not None:
                    metrics["episode_return_ma"] = self._return_ma.update(
                        metrics.get("episode_return", 0.0)
                    )
                prefix = "train" if env_idx == 0 else f"train/env_{env_idx}"
                if self._tb_logger is not None:
                    self._tb_logger.log_episode(metrics, self._episode_counts[env_idx], prefix=prefix)
                    if (
                        env_idx == 0
                        and self.reasoning_plot_every > 0
                        and self._episode_counts[env_idx] % self.reasoning_plot_every == 0
                    ):
                        self._tb_logger.log_reasoning_figure(
                            self._episode_series[env_idx],
                            episode_idx=self._episode_counts[env_idx],
                            prefix=prefix,
                        )
                    self._tb_logger.log_behavioural(
                        self._behavioural_series[env_idx],
                        episode_idx=self._episode_counts[env_idx],
                        prefix=prefix,
                    )
                self._episode_stats[env_idx].reset()
                if self._episode_series:
                    self._episode_series[env_idx].reset()
                if self._behavioural_series:
                    self._behavioural_series[env_idx].reset()
        return True
