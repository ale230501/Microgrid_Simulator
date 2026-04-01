import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandasgui import show

_SIMULATOR_ROOT = Path(__file__).resolve().parent
_SIMULATOR_SRC = _SIMULATOR_ROOT / "src"
if str(_SIMULATOR_SRC) not in sys.path:
    sys.path.insert(0, str(_SIMULATOR_SRC))

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    import gymnasium as _gymnasium

    sys.modules.setdefault("gym", _gymnasium)
    if hasattr(_gymnasium, "spaces"):
        sys.modules.setdefault("gym.spaces", _gymnasium.spaces)
    if hasattr(_gymnasium, "utils"):
        sys.modules.setdefault("gym.utils", _gymnasium.utils)

from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, GridModule, LoadModule, RenewableModule


class MicrogridSimulator:
    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _as_action_bounds(value, default=(0.0, 1.0)):
        if value is None:
            return tuple(default)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                return tuple(default)
        return tuple(default)

    def __init__(
        self,
        config_path,
        online,
        load_time_series=None,
        pv_time_series=None,
        grid_time_series=None,
        battery_chemistry=None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.online = online
        self.load_time_series = load_time_series
        self.pv_time_series = pv_time_series
        self.grid_time_series = grid_time_series

        battery_cfg = self.config["battery"]
        capacity = float(battery_cfg["capacity"])
        power_max = float(battery_cfg["power_max"])
        sample_time = float(battery_cfg["sample_time"])

        self.battery_chemistry = battery_chemistry or battery_cfg.get("chemistry") or "BASE"
        self.battery_transition_model = battery_cfg.get("transition_model")

        self.nominal_capacity = capacity
        self.max_charge_per_step = power_max * sample_time
        self.max_discharge_per_step = power_max * sample_time
        self.battery_efficiency = float(battery_cfg["efficiency"])
        self.battery_cost_cycle = float(battery_cfg.get("battery_cost_cycle", 0.0))
        init_charge_cfg = battery_cfg.get("init_charge")
        self.init_charge = None if init_charge_cfg is None else float(init_charge_cfg)
        init_soc_cfg = battery_cfg.get("init_soc")
        self.init_soc = None if init_soc_cfg is None else float(init_soc_cfg)
        self.min_soc = float(battery_cfg.get("min_soc", 0.0))
        self.max_soc = float(battery_cfg.get("max_soc", 1.0))
        disable_soh_cfg = battery_cfg.get("disable_soh_degradation", False)
        self.disable_soh_degradation = self._as_bool(disable_soh_cfg, default=False)

        grid_cfg = self.config["grid"]
        self.max_grid_import_power = float(grid_cfg["max_import_power"])
        self.max_grid_export_power = float(grid_cfg["max_export_power"])
        self.max_grid_import_energy = self.max_grid_import_power * sample_time
        self.max_grid_export_energy = self.max_grid_export_power * sample_time
        self.present_grid_prices = np.asarray(grid_cfg["prices"], dtype=float)

        scenario_cfg = self.config.get("scenario") or {}
        self.module_initial_step = int(scenario_cfg.get("initial_step", 0))
        self.module_final_step = int(scenario_cfg.get("final_step", -1))
        self.module_forecaster = scenario_cfg.get("forecaster", None)
        self.module_forecast_horizon = int(scenario_cfg.get("forecast_horizon", 23))
        self.module_forecaster_increase_uncertainty = self._as_bool(
            scenario_cfg.get("forecaster_increase_uncertainty", False),
            default=False,
        )
        self.module_forecaster_relative_noise = self._as_bool(
            scenario_cfg.get("forecaster_relative_noise", False),
            default=False,
        )
        self.module_raise_errors = self._as_bool(scenario_cfg.get("raise_errors", False), default=False)
        self.module_normalized_action_bounds = self._as_action_bounds(
            scenario_cfg.get("normalized_action_bounds"),
            default=(0.0, 1.0),
        )
        self.add_unbalanced_module = self._as_bool(
            scenario_cfg.get("add_unbalanced_module", True),
            default=True,
        )
        self.loss_load_cost = float(scenario_cfg.get("loss_load_cost", 10.0))
        self.overgeneration_cost = float(scenario_cfg.get("overgeneration_cost", 2.0))
        self.grid_cost_per_unit_co2 = float(scenario_cfg.get("grid_cost_per_unit_co2", 0.0))

    def _build_transition_model(self):
        if self.battery_transition_model is None:
            return None
        return self.battery_transition_model

    def build_microgrid(self):
        transition_model = self._build_transition_model()

        min_soc = float(np.clip(self.min_soc, 0.0, 1.0))
        max_soc = float(np.clip(self.max_soc, min_soc, 1.0))
        min_capacity = float(self.nominal_capacity * min_soc)
        max_capacity = float(self.nominal_capacity)
        target_max_capacity = float(max_capacity * max_soc)
        if self.init_charge is not None:
            init_charge_raw = float(self.init_charge)
        elif self.init_soc is not None:
            init_charge_raw = float(self.init_soc * self.nominal_capacity)
        else:
            init_charge_raw = float(min_capacity)
        init_charge = float(np.clip(init_charge_raw, min_capacity, target_max_capacity))

        battery_kwargs = dict(
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            max_charge=self.max_charge_per_step,
            max_discharge=self.max_discharge_per_step,
            efficiency=self.battery_efficiency,
            battery_cost_cycle=self.battery_cost_cycle,
            init_charge=init_charge,
            initial_step=self.module_initial_step,
            normalized_action_bounds=self.module_normalized_action_bounds,
            raise_errors=self.module_raise_errors,
        )
        if transition_model is not None:
            battery_kwargs["battery_transition_model"] = transition_model
        battery = BatteryModule(**battery_kwargs)

        # Preserve configured operating SoC bounds for EMS logic (RBC/MPC/RL),
        # while using the base pymgrid BatteryModule directly (no BMS layer).
        battery.min_soc = min_soc
        battery.max_soc = max_soc

        timeseries_module_kwargs = dict(
            forecaster=self.module_forecaster,
            forecast_horizon=self.module_forecast_horizon,
            forecaster_increase_uncertainty=self.module_forecaster_increase_uncertainty,
            forecaster_relative_noise=self.module_forecaster_relative_noise,
            initial_step=self.module_initial_step,
            final_step=self.module_final_step,
            normalized_action_bounds=self.module_normalized_action_bounds,
            raise_errors=self.module_raise_errors,
        )

        load_module = LoadModule(time_series=self.load_time_series, **timeseries_module_kwargs)
        pv_module = RenewableModule(time_series=self.pv_time_series, **timeseries_module_kwargs)
        grid_module = GridModule(
            max_import=self.max_grid_import_energy,
            max_export=self.max_grid_export_energy,
            time_series=self.grid_time_series,
            cost_per_unit_co2=self.grid_cost_per_unit_co2,
            **timeseries_module_kwargs,
        )

        microgrid = Microgrid(
            modules=[
                battery,
                ("load", load_module),
                ("pv", pv_module),
                grid_module,
            ],
            add_unbalanced_module=self.add_unbalanced_module,
            loss_load_cost=self.loss_load_cost,
            overgeneration_cost=self.overgeneration_cost,
        )
        return microgrid

    @staticmethod
    def _first_available_column(columns, candidates):
        for col in candidates:
            if col in columns:
                return col
        return None

    def get_simulation_log(self, microgrid):
        log_df = microgrid.log.copy()
        available = log_df.columns

        # Canonical columns expected by downstream EMS scripts/tools.
        canonical_columns = [
            ("load", 0, "load_met"),
            ("pv", 0, "renewable_used"),
            ("pv", 0, "curtailment"),
            ("pv", 0, "reward"),
            ("battery", 0, "soc"),
            ("battery", 0, "current_charge"),
            ("battery", 0, "discharge_amount"),
            ("battery", 0, "charge_amount"),
            ("battery", 0, "reward"),
            ("grid", 0, "grid_import"),
            ("grid", 0, "grid_export"),
            ("grid", 0, "export_price_current"),
            ("grid", 0, "import_price_current"),
            ("grid", 0, "reward"),
            ("balancing", 0, "loss_load"),
            ("balancing", 0, "overgeneration"),
            ("balance", 0, "reward"),
        ]

        aliases = {
            ("grid", 0, "export_price_current"): [
                ("grid", 0, "export_price_current"),
                ("grid", 0, "export_price"),
            ],
            ("grid", 0, "import_price_current"): [
                ("grid", 0, "import_price_current"),
                ("grid", 0, "import_price"),
            ],
            ("balancing", 0, "loss_load"): [
                ("balancing", 0, "loss_load"),
                ("unbalanced_energy", 0, "loss_load"),
            ],
            ("balancing", 0, "overgeneration"): [
                ("balancing", 0, "overgeneration"),
                ("unbalanced_energy", 0, "overgeneration"),
            ],
        }

        for canonical_col, candidates in aliases.items():
            if canonical_col in available:
                continue
            source_col = self._first_available_column(available, candidates)
            if source_col is not None:
                log_df[canonical_col] = log_df[source_col]
            else:
                log_df[canonical_col] = np.nan

        for canonical_col in canonical_columns:
            if canonical_col not in log_df.columns:
                log_df[canonical_col] = np.nan

        microgrid_df = log_df[canonical_columns]

        log = log_df.copy()
        log.columns = ["{}_{}_{}".format(*col) for col in log.columns]
        return microgrid_df, log

    def sum_module_info(self, info_dict, module_name, key):
        total = 0.0
        for entry in info_dict.get(module_name, []):
            if not isinstance(entry, dict):
                continue
            value = entry.get(key)
            if value is None:
                continue
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        return total
