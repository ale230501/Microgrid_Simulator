from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys
import cvxpy as cp
import numpy as np
import pandas as pd
import yaml
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SIMULATOR_ROOT = PROJECT_ROOT / "SIMULATOR"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from SIMULATOR.tools import compute_offline_tariff_vectors


# -----------------------------
# Data and config
# -----------------------------
def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as cfg_file:
        return yaml.safe_load(cfg_file)


def _acc_curve(dod: np.ndarray, wear_a: float, wear_b: float, wear_c: float, eps: float = 1e-6) -> np.ndarray:
    """Battery aging acceleration curve as function of depth of discharge (DoD)."""
    dod = np.maximum(dod, eps)
    return wear_a * np.power(dod, -wear_b) * np.exp(-wear_c * dod)


def build_wear_linearization(
    *,
    wear_a: float | None,
    wear_b: float | None,
    wear_c: float | None,
    wear_B: float | None,
    soc_min: float,
    soc_max: float,
    segments: int,
    temp_factor: float = 1.0,
    wear_cost_scale: float = 1.0,
) -> dict[str, Any] | None:
    """Build piecewise linear approximation of battery wear function over SOC range."""
    if wear_a is None or wear_b is None or wear_c is None or wear_B is None:
        return None
    if wear_B <= 0.0 or segments < 2:
        return None

    soc_min = float(np.clip(soc_min, 0.0, 1.0))
    soc_max = float(np.clip(soc_max, 0.0, 1.0))
    if soc_max <= soc_min + 1e-6:
        return None

    soc_breakpoints = np.linspace(soc_min, soc_max, segments + 1)
    dod_breakpoints = 1.0 - soc_breakpoints
    acc_breakpoints = _acc_curve(dod_breakpoints, wear_a, wear_b, wear_c)
    phi_breakpoints = 1.0 / np.maximum(acc_breakpoints, 1e-6)

    slopes = np.diff(phi_breakpoints) / np.diff(soc_breakpoints)
    intercepts = phi_breakpoints[:-1] - slopes * soc_breakpoints[:-1]

    scale = max(0.0, float(wear_cost_scale))
    coeff = float(temp_factor) * (float(wear_B) / 2.0) * scale

    return {
        "soc_breakpoints": soc_breakpoints,
        "phi_breakpoints": phi_breakpoints,
        "slopes": slopes,
        "intercepts": intercepts,
        "coeff": coeff,
        "phi_min": float(np.min(phi_breakpoints)),
        "phi_max": float(np.max(phi_breakpoints)),
    }



# -----------------------------
# Time series loading
# -----------------------------
def load_time_series(
    data_path: str | Path,
    *,
    steps: int,
    horizon: int,
    start_step: int = 0,
    end_step: int | None = None,
    sample_time: float,
    timezone: str,
    price_bands: dict[str, Any],
) -> dict[str, Any]:
    
    time_series = pd.read_csv(data_path)
    required_columns = {"datetime", "solar", "load"}
    missing_columns = required_columns - set(time_series.columns)
    if missing_columns:
        raise KeyError(f"Missing columns in MPC dataset: {sorted(missing_columns)}")

    keep_columns = ["datetime", "solar", "load"]
    if "price_buy" in time_series.columns:
        keep_columns.append("price_buy")
    if "price_sell" in time_series.columns:
        keep_columns.append("price_sell")
    time_series = time_series[keep_columns].copy()

    numeric_block = time_series[["solar", "load"]].apply(pd.to_numeric, errors="coerce")
    if numeric_block.isna().to_numpy().any():
        numeric_block = numeric_block.interpolate(limit_direction="both")
    if numeric_block.isna().to_numpy().any():
        raise ValueError("Dataset MPC contiene valori non numerici/non interpolabili in solar/load.")
    time_series[["solar", "load"]] = numeric_block
    time_series["datetime"] = pd.to_datetime(time_series["datetime"], utc=True, errors="coerce")

    if start_step < 0:
        raise ValueError("start_step non puo' essere negativo")
    if end_step is None:
        end_step = start_step + steps
    else:
        end_step = int(end_step)
    if end_step <= start_step:
        raise ValueError(
            f"end_step deve essere > start_step (start_step={start_step}, end_step={end_step})"
        )
    forecast_end = end_step + horizon
    if forecast_end > len(time_series):
        raise ValueError(
            f"Dataset troppo corto: richiesti {forecast_end} punti, disponibili {len(time_series)}."
        )

    window = time_series.iloc[start_step:forecast_end]

    pv_series_kwh = window['solar'].clip(lower=0).to_numpy()  # kWh
    load_series_kwh = window['load'].to_numpy()  # kWh
    timestamps = window['datetime']

    # kW = kWh / dt
    pv_series = pv_series_kwh / sample_time  # kW
    load_series = load_series_kwh / sample_time  # kW

    if {"price_buy", "price_sell"}.issubset(window.columns):
        # Prefer dataset prices when available (scenario bundle-aligned mode).
        price_buy = pd.to_numeric(window["price_buy"], errors="coerce").to_numpy(dtype=float)
        price_sell = pd.to_numeric(window["price_sell"], errors="coerce").to_numpy(dtype=float)

        if np.isnan(price_buy).any() or np.isnan(price_sell).any():
            fallback_buy, fallback_sell = compute_offline_tariff_vectors(
                ts_series=timestamps,
                local_timezone=timezone,
                price_config=price_bands,
            )
            nan_buy = np.isnan(price_buy)
            nan_sell = np.isnan(price_sell)
            if nan_buy.any():
                price_buy[nan_buy] = fallback_buy[nan_buy]
            if nan_sell.any():
                price_sell[nan_sell] = fallback_sell[nan_sell]
    else:
        price_buy, price_sell = compute_offline_tariff_vectors(
            ts_series=timestamps,
            local_timezone=timezone,
            price_config=price_bands,
        )

    return {
        "pv_series": pv_series,
        "load_series": load_series,
        "price_buy": price_buy,
        "price_sell": price_sell,
        "timestamps": timestamps,
        "pv_series_kwh": pv_series_kwh,
        "load_series_kwh": load_series_kwh,
    }


#-----------------------------
# Forecast (oracle) 
#-----------------------------
def _forecast_series(series: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    if t0 < 0:
        raise IndexError("negative start index not allowed")
    if t0 + horizon > len(series):
        raise IndexError("Requested forecast exceeds series length")
    return series[t0 : t0 + horizon].reshape(1, horizon)


def forecast_load(series: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    return _forecast_series(series, t0, horizon)


def forecast_pv(series: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    return _forecast_series(series, t0, horizon)


def forecast_price_buy(series: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    return _forecast_series(series, t0, horizon)


def forecast_price_sell(series: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    return _forecast_series(series, t0, horizon)


#-----------------------------
# Battery state
#-----------------------------
@dataclass
class BatteryState:
    current_charge: float
    soc_frac: float
    e_min: float
    e_max: float
    e_min_mpc: float
    e_max_mpc: float
    max_charge_kwh: float
    max_discharge_kwh: float
    p_max_charge: float
    p_max_discharge: float
    max_internal_charge: float
    max_internal_discharge: float
    effective_capacity: float
    soh: float


def _battery_energy_bounds(battery_module) -> tuple[float, float, float, float, float]:
    current_charge = float(getattr(battery_module, "current_charge", 0.0))
    max_capacity = float(getattr(battery_module, "max_capacity", max(current_charge, 0.0)))
    min_capacity = float(getattr(battery_module, "min_capacity", 0.0))

    if max_capacity <= 0.0:
        return 0.0, 0.0, 0.0, max(current_charge, 0.0), 0.0

    soc_min = float(np.clip(getattr(battery_module, "min_soc", min_capacity / max_capacity), 0.0, 1.0))
    soc_max = float(np.clip(getattr(battery_module, "max_soc", 1.0), soc_min, 1.0))

    e_min = max(min_capacity, soc_min * max_capacity)
    e_max = min(max_capacity, soc_max * max_capacity)
    e_min_mpc = min(e_min, current_charge)
    e_max_mpc = max(e_max, current_charge)
    return e_min, e_max, e_min_mpc, e_max_mpc, max_capacity


def _battery_step_energy_limits(battery_module, e_min: float, e_max: float) -> tuple[float, float, float, float]:
    current_charge = float(getattr(battery_module, "current_charge", 0.0))
    max_discharge_internal = min(
        float(getattr(battery_module, "max_discharge", 0.0)),
        max(0.0, current_charge - e_min),
    )
    max_charge_internal = min(
        float(getattr(battery_module, "max_charge", 0.0)),
        max(0.0, e_max - current_charge),
    )

    if hasattr(battery_module, "model_transition"):
        discharge_soc_limit = max(0.0, float(battery_module.model_transition(max_discharge_internal)))
        charge_soc_limit = max(0.0, float(-battery_module.model_transition(-max_charge_internal)))
    else:
        discharge_soc_limit = max(0.0, max_discharge_internal)
        charge_soc_limit = max(0.0, max_charge_internal)

    max_discharge_kwh = max(
        0.0,
        min(float(getattr(battery_module, "max_production", discharge_soc_limit)), discharge_soc_limit),
    )
    max_charge_kwh = max(
        0.0,
        min(float(getattr(battery_module, "max_consumption", charge_soc_limit)), charge_soc_limit),
    )
    return max_charge_kwh, max_discharge_kwh, max_charge_internal, max_discharge_internal


def read_battery_state(battery_module, sample_time: float) -> BatteryState:
    current_charge = float(battery_module.current_charge)  # kWh
    soc_frac = float(battery_module.soc)  # 0..1

    e_min, e_max, e_min_mpc, e_max_mpc, max_capacity = _battery_energy_bounds(battery_module)
    (
        max_charge_kwh,
        max_discharge_kwh,
        max_charge_internal,
        max_discharge_internal,
    ) = _battery_step_energy_limits(battery_module, e_min, e_max)

    p_max_charge = max_charge_kwh / sample_time
    p_max_discharge = max_discharge_kwh / sample_time

    max_internal_charge = float(getattr(battery_module, "max_internal_charge", max_charge_internal))
    max_internal_discharge = float(getattr(battery_module, "max_internal_discharge", max_discharge_internal))

    effective_capacity = float(getattr(battery_module, "effective_capacity", max_capacity))

    transition_model = battery_module.battery_transition_model
    soh = float(getattr(transition_model, "soh", 1.0) or 1.0)

    return BatteryState(
        current_charge=current_charge,
        soc_frac=soc_frac,
        e_min=e_min,
        e_max=e_max,
        e_min_mpc=e_min_mpc,
        e_max_mpc=e_max_mpc,
        max_charge_kwh=max_charge_kwh,
        max_discharge_kwh=max_discharge_kwh,
        p_max_charge=p_max_charge,
        p_max_discharge=p_max_discharge,
        max_internal_charge=max_internal_charge,
        max_internal_discharge=max_internal_discharge,
        effective_capacity=effective_capacity,
        soh=soh,
    )
    

# ----------------------------- 
# MILP model builder
# -----------------------------
def build_mpc_problem(
    *,
    A: np.ndarray,
    B: np.ndarray,
    N: int,
    charge_min_max: tuple[float, float],
    storage_power_min_max: tuple[float, float],
    grid_power_min_max: tuple[float, float],
    sample_time: float,
    efficiency: float,
    loss_load_cost: float,
    overgeneration_cost: float,
    simple_battery_cost_cycle: float = 0.0,
    ramp_penalty: float = 0.0,
    ramp_kw_per_h: float = 20.0,
    wear_linearization: dict[str, Any] | None = None,
    ):
    
    # Parameters
    price_buy = cp.Parameter((1, N))
    price_sell = cp.Parameter((1, N))
    load = cp.Parameter((1, N))
    pv = cp.Parameter((1, N))
    SoE_0 = cp.Parameter((1, 1))
    p_in_prev = cp.Parameter((1, 1))
    p_out_prev = cp.Parameter((1, 1))

    e_min = cp.Parameter((1, N + 1))
    e_max = cp.Parameter((1, N + 1))
    p_charge_max = cp.Parameter((1, N))
    p_discharge_max = cp.Parameter((1, N))

    e_min.value = np.full((1, N + 1), charge_min_max[0], dtype=float)
    e_max.value = np.full((1, N + 1), charge_min_max[1], dtype=float)
    p_charge_max.value = np.full((1, N), storage_power_min_max[1], dtype=float)
    p_discharge_max.value = np.full((1, N), storage_power_min_max[1], dtype=float)

    # Decision variables
    SoE = cp.Variable((1, N + 1))
    Ps_in = cp.Variable((1, N))
    Ps_out = cp.Variable((1, N))
    u = cp.Variable((1, N), boolean=True)

    E_imp = cp.Variable((1, N))
    E_exp = cp.Variable((1, N))
    b_ie = cp.Variable((1, N), boolean=True)

    loss_load = cp.Variable((1, N))
    overgeneration = cp.Variable((1, N))

    ramp_slack_in = cp.Variable((1, N), nonneg=True)
    ramp_slack_out = cp.Variable((1, N), nonneg=True)

    wear = None
    if wear_linearization is not None:
        segments = len(wear_linearization["slopes"])
        inv_capacity = cp.Parameter(nonneg=True, value=1.0)
        wear_phi = cp.Variable((1, N + 1))
        wear_delta = cp.Variable((1, N), nonneg=True)
        wear_seg = cp.Variable((segments, N + 1), boolean=True)

    # Objective
    dt = sample_time
    M_grid = max(abs(grid_power_min_max[0]), grid_power_min_max[1]) * dt

    ramp_penalty = max(0.0, float(ramp_penalty))
    eff_safe = max(float(efficiency), 1e-6)
    simple_wear_coeff = max(0.0, float(simple_battery_cost_cycle))
    ene_cost = cp.multiply(price_buy, E_imp) - cp.multiply(price_sell, E_exp)
    imbalance_cost = loss_load_cost * cp.sum(loss_load) * dt
    imbalance_cost += overgeneration_cost * cp.sum(overgeneration) * dt
    ramp_cost = ramp_penalty * cp.sum(ramp_slack_in + ramp_slack_out) * dt
    # Base BatteryModule wear cost: cost_cycle * internal_energy_throughput.
    simple_wear_cost = simple_wear_coeff * dt * cp.sum(eff_safe * Ps_in + (1.0 / eff_safe) * Ps_out)
    objective_expr = cp.sum(ene_cost) + imbalance_cost + ramp_cost + simple_wear_cost

    # Constraints
    constraints = [SoE[:, 0] == SoE_0]

    ramp_kw_per_h = max(0.0, float(ramp_kw_per_h))
    RAMP = ramp_kw_per_h * dt

    for k in range(N):
        """if k == 0:
            constraints += [
                Ps_in[:, 0] - p_in_prev <= RAMP + ramp_slack_in[:, 0],  # first step ramping constraint
                p_in_prev - Ps_in[:, 0] <= RAMP + ramp_slack_in[:, 0],  
                Ps_out[:, 0] - p_out_prev <= RAMP + ramp_slack_out[:, 0],
                p_out_prev - Ps_out[:, 0] <= RAMP + ramp_slack_out[:, 0],
            ]
        else:
            constraints += [
                Ps_in[:, k] - Ps_in[:, k - 1] <= RAMP + ramp_slack_in[:, k],  # ramping constraint after first step
                Ps_in[:, k - 1] - Ps_in[:, k] <= RAMP + ramp_slack_in[:, k],
                Ps_out[:, k] - Ps_out[:, k - 1] <= RAMP + ramp_slack_out[:, k],
                Ps_out[:, k - 1] - Ps_out[:, k] <= RAMP + ramp_slack_out[:, k],
            ]"""

        constraints += [
            SoE[:, k + 1] == A @ SoE[:, k] + B[0, 0] * Ps_in[:, k] + B[0, 1] * Ps_out[:, k],  # system dynamics 
            SoE[:, k] >= e_min[:, k],  # constraints on SoE
            SoE[:, k] <= e_max[:, k],
            Ps_in[:, k] >= 0,  # constraints on power flows
            Ps_out[:, k] >= 0,
            Ps_in[:, k] <= p_charge_max[:, k],
            Ps_out[:, k] <= p_discharge_max[:, k],
            Ps_in[:, k] <= cp.multiply(u[:, k], p_charge_max[:, k]), 
            Ps_out[:, k] <= cp.multiply(1 - u[:, k], p_discharge_max[:, k]),
        ]

        net_balance = pv[:, k] - load[:, k] - Ps_in[:, k] + Ps_out[:, k]  # net power balance at grid connection point
        P_grid_k = net_balance + loss_load[:, k] - overgeneration[:, k]  # grid power including losses and overgeneration
        constraints += [
            P_grid_k >= grid_power_min_max[0],  # grid power limits
            P_grid_k <= grid_power_min_max[1],
            E_exp[:, k] - E_imp[:, k] == P_grid_k * dt,  # energy imported/exported from/to grid
            E_imp[:, k] >= 0,  
            E_exp[:, k] >= 0,
            E_imp[:, k] <= M_grid * b_ie[:, k],  
            E_exp[:, k] <= M_grid * (1 - b_ie[:, k]),
            loss_load[:, k] >= 0,
            overgeneration[:, k] >= 0,
        ]

    constraints += [
        SoE[:, N] >= e_min[:, N],  # final SoE constraints
        SoE[:, N] <= e_max[:, N],
    ]

    if wear_linearization is not None:
        soc_breakpoints = np.asarray(wear_linearization["soc_breakpoints"], dtype=float)
        slopes = np.asarray(wear_linearization["slopes"], dtype=float)
        intercepts = np.asarray(wear_linearization["intercepts"], dtype=float)
        phi_min = float(wear_linearization["phi_min"])
        phi_max = float(wear_linearization["phi_max"])
        M_soc = float(soc_breakpoints[-1] - soc_breakpoints[0])
        M_phi = float(phi_max - phi_min + 1e-6)
        soc = cp.multiply(inv_capacity, SoE)

        for k in range(N + 1):
            constraints += [cp.sum(wear_seg[:, k]) == 1]
            for i in range(segments):
                x0 = float(soc_breakpoints[i])
                x1 = float(soc_breakpoints[i + 1])
                m = float(slopes[i])
                b = float(intercepts[i])
                constraints += [
                    soc[:, k] >= x0 - M_soc * (1 - wear_seg[i, k]),
                    soc[:, k] <= x1 + M_soc * (1 - wear_seg[i, k]),
                    wear_phi[:, k] >= m * soc[:, k] + b - M_phi * (1 - wear_seg[i, k]),
                    wear_phi[:, k] <= m * soc[:, k] + b + M_phi * (1 - wear_seg[i, k]),
                ]

        for k in range(N):
            constraints += [
                wear_delta[:, k] >= wear_phi[:, k + 1] - wear_phi[:, k],
                wear_delta[:, k] >= -(wear_phi[:, k + 1] - wear_phi[:, k]),
            ]

        wear_cost = float(wear_linearization["coeff"]) * cp.sum(wear_delta)
        objective_expr += wear_cost
        wear = {
            "inv_capacity": inv_capacity,
            "delta_phi": wear_delta,
            "phi": wear_phi,
            "segments": wear_seg,
            "coeff": float(wear_linearization["coeff"]),
            "cost_expr": wear_cost,
        }

    objective = cp.Minimize(objective_expr)

    prob = cp.Problem(objective, constraints)
    params = (
        price_buy,
        price_sell,
        load,
        pv,
        SoE_0,
        p_in_prev,
        p_out_prev,
        e_min,
        e_max,
        p_charge_max,
        p_discharge_max,
    )
    variables = (Ps_in, Ps_out, SoE, u, E_imp, E_exp, loss_load, overgeneration)
    simple_wear = {
        "coeff": simple_wear_coeff,
        "cost_expr": simple_wear_cost,
    }
    return prob, params, variables, wear, simple_wear


# -----------------------------
# Controller class
# -----------------------------
class MPCController:
    def __init__(
        self,
        microgrid,
        *,
        config_path: str | Path = PROJECT_ROOT / "MODEL_PREDICTIVE" / "params_OPSD.yml",
        data_path: str | Path | None = None,
        solver: str | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        log_path: str ="mpc_log.csv",
    ):
        self.microgrid = microgrid

        # Balancing costs (loss load / overgeneration)
        balancing_list = self.microgrid.modules.get("balancing")
        balancing = balancing_list[0] if balancing_list else None
        if balancing:
            self.loss_load_cost = balancing.loss_load_cost
            self.overgeneration_cost = balancing.overgeneration_cost
        else:
            self.loss_load_cost = 10.0
            self.overgeneration_cost = 2.0

        # Load config
        cfg = load_config(config_path)
        battery_cfg = cfg["battery"]
        grid_cfg = cfg["grid"]
        mpc_cfg = cfg["mpc"]
        ems_cfg = cfg["ems"]

        # Battery params
        self.capacity = battery_cfg["capacity"]
        self.power_max = battery_cfg["power_max"]
        self.efficiency = battery_cfg["efficiency"]
        self.sample_time = battery_cfg["sample_time"]
        self.init_charge = battery_cfg["init_charge"]
        self.charge_bounds = (0.0, float(self.capacity))

        # Grid params (values from YAML are in kW).
        self.grid_import = float(grid_cfg["max_import_power"])
        self.grid_export = float(grid_cfg["max_export_power"])

        # MPC params
        self.horizon = mpc_cfg["mpc_horizon"]
        self.steps = ems_cfg["steps"]
        start_step_cfg = ems_cfg.get("start_step", 0)
        end_step_cfg = ems_cfg.get("end_step", None)
        if start_step is not None:
            start_step_cfg = int(start_step)
        if end_step is not None:
            end_step_cfg = int(end_step)

        self.start_step = int(start_step_cfg)
        if end_step_cfg is None:
            self.steps = int(ems_cfg["steps"])
            self.end_step = None
        else:
            self.end_step = int(end_step_cfg)
            if self.end_step <= self.start_step:
                raise ValueError(
                    f"end_step deve essere > start_step (start_step={self.start_step}, end_step={self.end_step})"
                )
            self.steps = self.end_step - self.start_step
        self.timezone = ems_cfg["timezone"]
        self.price_bands = ems_cfg["price_bands"]
        self.ramp_penalty = float(mpc_cfg.get("ramp_penalty", 0.0))
        self.ramp_kw_per_h = float(mpc_cfg.get("ramp_kw_per_h", 20.0))
        solver_name = solver if solver is not None else mpc_cfg.get("solver", "GUROBI")
        self._solver = self._resolve_solver(solver_name)

        # Logging options
        self.log_full_forecast = bool(mpc_cfg.get("log_full_forecast", False))
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_header_written = self.log_path.exists()
        self._log_fields = None

        # Dynamics
        A = np.array([[1.0]])
        b_in = self.sample_time * self.efficiency
        b_out = -self.sample_time / self.efficiency
        B = np.array([[b_in, b_out]])

        battery_module = self.microgrid.battery[0] 
        self.simple_battery_cost_cycle = max(
            0.0,
            float(getattr(battery_module, "battery_cost_cycle", battery_cfg.get("battery_cost_cycle", 0.0)) or 0.0),
        )
        wear_segments = int(mpc_cfg.get("wear_segments", 0))
        wear_cost_scale = float(mpc_cfg.get("wear_cost_scale", 1.0))
        wear_linearization = None
        transition_model = getattr(battery_module, "battery_transition_model", None)
        if transition_model is not None and wear_segments >= 2:
            wear_a = getattr(transition_model, "wear_a", None)
            wear_b = getattr(transition_model, "wear_b", None)
            wear_c = getattr(transition_model, "wear_c", None)
            wear_B = getattr(transition_model, "wear_B", None)
            if wear_B in (None, 0):
                wear_B = battery_cfg.get("battery_replacement_cost", None)

            soc_min = float(getattr(battery_module, "min_soc", battery_cfg.get("min_soc", 0.0)))
            soc_max = float(getattr(battery_module, "max_soc", battery_cfg.get("max_soc", 1.0)))

            temp_factor = 1.0
            if getattr(transition_model, "wear_use_temperature", False):
                temp_c = float(getattr(transition_model, "temperature_c", 25.0))
                temp_coeff = float(getattr(transition_model, "wear_temp_coeff", 0.0035))
                temp_ref = float(getattr(transition_model, "wear_temp_ref_c", 25.0))
                temp_factor = float(np.exp(temp_coeff * abs(temp_c - temp_ref)))

            wear_linearization = build_wear_linearization(
                wear_a=wear_a,
                wear_b=wear_b,
                wear_c=wear_c,
                wear_B=wear_B,
                soc_min=soc_min,
                soc_max=soc_max,
                segments=wear_segments,
                temp_factor=temp_factor,
                wear_cost_scale=wear_cost_scale,
            )

        # Build problem
        self.prob, params, variables, wear, simple_wear = build_mpc_problem(
            A=A,
            B=B,
            N=self.horizon,
            charge_min_max=self.charge_bounds,
            storage_power_min_max=(0.0, self.power_max),
            grid_power_min_max=(-self.grid_import, self.grid_export),
            sample_time=self.sample_time,
            efficiency=self.efficiency,
            simple_battery_cost_cycle=self.simple_battery_cost_cycle,
            loss_load_cost=self.loss_load_cost,
            overgeneration_cost=self.overgeneration_cost,
            ramp_penalty=self.ramp_penalty,
            ramp_kw_per_h=self.ramp_kw_per_h,
            wear_linearization=wear_linearization,
        )
        self.wear = wear
        self.simple_wear = simple_wear
        self.inv_capacity_p = wear["inv_capacity"] if wear is not None else None

        (
            self.price_buy_p,
            self.price_sell_p,
            self.load_p,
            self.pv_p,
            self.SoE_p,
            self.p_in_prev_p,
            self.p_out_prev_p,
            self.e_min_p,
            self.e_max_p,
            self.p_charge_max_p,
            self.p_discharge_max_p,
        ) = params

        (
            self.Ps_in,
            self.Ps_out,
            self.SoE,
            self.u,
            self.E_imp,
            self.E_exp,
            self.loss_load,
            self.overgeneration,
        ) = variables

        # State
        self.SoE_val = np.array([[self.init_charge]], dtype=float)
        self.prev_charge = 0.0
        self.prev_discharge = 0.0
        self.current_step = 0
        self.records = []

        # Time series
        if data_path is None:
            raise ValueError("Missing MPC forecast dataset path. Pass data_path explicitly.")
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"MPC forecast dataset not found: {data_path}")
        ts = load_time_series(
            data_path,
            steps=self.steps,
            horizon=self.horizon,
            start_step=self.start_step,
            end_step=self.end_step,
            sample_time=self.sample_time,
            timezone=self.timezone,
            price_bands=self.price_bands,
        )
        self.pv_series = ts["pv_series"]
        self.load_series = ts["load_series"]
        self.price_buy = ts["price_buy"]
        self.price_sell = ts["price_sell"]
        self.timestamps = ts["timestamps"]

    def get_action(self, verbose: int = 0) -> dict[str, float]:

        if self.current_step >= self.steps:
            raise StopIteration("Simulation steps exhausted")

        # Read battery state at current time step
        battery_module = self.microgrid.battery[0]
        batt = read_battery_state(battery_module, self.sample_time) 

        self.SoE_val = np.array([[batt.current_charge]], dtype=float)

        # MPC limits ("feasibility-safe")
        self.e_min_p.value = np.full((1, self.horizon + 1), batt.e_min_mpc, dtype=float)
        self.e_max_p.value = np.full((1, self.horizon + 1), batt.e_max_mpc, dtype=float)
        nominal_p_charge = float(battery_module.max_charge) / self.sample_time
        nominal_p_discharge = float(battery_module.max_discharge) / self.sample_time
        # Use nominal power for future steps so the MPC can recharge then discharge later.
        p_charge_max = np.full((1, self.horizon), nominal_p_charge, dtype=float)
        p_discharge_max = np.full((1, self.horizon), nominal_p_discharge, dtype=float)
        p_charge_max[0, 0] = batt.p_max_charge
        p_discharge_max[0, 0] = batt.p_max_discharge
        self.p_charge_max_p.value = p_charge_max
        self.p_discharge_max_p.value = p_discharge_max

        # Previous power flows
        self.SoE_p.value = self.SoE_val
        self.p_in_prev_p.value = [[self.prev_charge]]
        self.p_out_prev_p.value = [[self.prev_discharge]]
        if self.inv_capacity_p is not None:
            inv_capacity = 1.0 / max(batt.effective_capacity, 1e-6)
            self.inv_capacity_p.value = inv_capacity

        # Forecasts
        load_hat = forecast_load(self.load_series, self.current_step, self.horizon)
        pv_hat = forecast_pv(self.pv_series, self.current_step, self.horizon)
        price_buy_hat = forecast_price_buy(self.price_buy, self.current_step, self.horizon)
        price_sell_hat = forecast_price_sell(self.price_sell, self.current_step, self.horizon)

        self.load_p.value = load_hat
        self.pv_p.value = pv_hat
        self.price_buy_p.value = price_buy_hat
        self.price_sell_p.value = price_sell_hat

        # Solve
        self._solve_problem(verbose)

        # Extract first-step actions
        p_charge = float(self.Ps_in.value[0, 0])
        p_discharge = float(self.Ps_out.value[0, 0])

        # Energies (kWh)
        e_grid_kwh = float(self.E_imp.value[0, 0] - self.E_exp.value[0, 0])
        e_batt_kwh = (p_discharge - p_charge) * self.sample_time

        # Update internal state
        self.prev_charge = p_charge
        self.prev_discharge = p_discharge

        # Log step
        self._log_step(
            batt=batt,
            load_hat=load_hat,
            pv_hat=pv_hat,
            price_buy_hat=price_buy_hat,
            price_sell_hat=price_sell_hat,
            p_charge=p_charge,
            p_discharge=p_discharge,
            e_grid_kwh=e_grid_kwh,
            e_batt_kwh=e_batt_kwh,
        )

        self.current_step += 1

        return {"battery": e_batt_kwh, "grid": e_grid_kwh}
    

    def _solve_problem(self, verbose: int) -> None:
        solve_kwargs = {"warm_start": True, "verbose": verbose > 1}
        if self._solver is not None:
            solve_kwargs["solver"] = self._solver
        self.prob.solve(**solve_kwargs)

    @staticmethod
    def _resolve_solver(solver_name):
        if solver_name is None:
            return None
        if isinstance(solver_name, str):
            solver_upper = solver_name.upper()
            if hasattr(cp, solver_upper):
                return getattr(cp, solver_upper)
            return solver_upper
        return solver_name


    def _log_step(
        self,
        *,
        batt,
        load_hat,
        pv_hat,
        price_buy_hat,
        price_sell_hat,
        p_charge,
        p_discharge,
        e_grid_kwh,
        e_batt_kwh,
    ) -> None:
        grid_power = float(pv_hat[0, 0] - load_hat[0, 0] - p_charge + p_discharge)
        timestamp = None
        if hasattr(self, "timestamps"):
            try:
                timestamp = self.timestamps.iloc[self.current_step]
            except Exception:
                timestamp = None

        record = {
            "step": self.current_step,
            "timestamp": timestamp,
            # battery state
            "current_charge_kwh": batt.current_charge,
            "soc_frac": batt.soc_frac,
            "e_min_kwh": batt.e_min,
            "e_max_kwh": batt.e_max,
            "e_min_mpc_kwh": batt.e_min_mpc,
            "e_max_mpc_kwh": batt.e_max_mpc,
            "max_charge_kwh": batt.max_charge_kwh,
            "max_discharge_kwh": batt.max_discharge_kwh,
            "p_max_charge_kw": batt.p_max_charge,
            "p_max_discharge_kw": batt.p_max_discharge,
            "max_internal_charge_kwh": batt.max_internal_charge,
            "max_internal_discharge_kwh": batt.max_internal_discharge,
            "effective_capacity_kwh": batt.effective_capacity,
            "soh": batt.soh,
            # decisions
            "p_charge_kw": p_charge,
            "p_discharge_kw": p_discharge,
            "grid_power_kw": grid_power,
            "e_grid_kwh": e_grid_kwh,
            "e_batt_kwh": e_batt_kwh,
            # first-step forecasts and prices
            "load_hat_kw": float(load_hat[0, 0]),
            "pv_hat_kw": float(pv_hat[0, 0]),
            "price_buy": float(price_buy_hat[0, 0]),
            "price_sell": float(price_sell_hat[0, 0]),
            # solver outputs
            "E_imp_kwh": float(self.E_imp.value[0, 0]),
            "E_exp_kwh": float(self.E_exp.value[0, 0]),
            "loss_load_kwh": float(self.loss_load.value[0, 0]) * self.sample_time,
            "overgeneration_kwh": float(self.overgeneration.value[0, 0]) * self.sample_time,
            "objective": float(self.prob.value) if self.prob.value is not None else np.nan,
            "status": self.prob.status,
        }
        simple_wear_cost_horizon = np.nan
        if self.simple_wear["cost_expr"].value is not None:
            simple_wear_cost_horizon = float(self.simple_wear["cost_expr"].value)

        eff_safe = max(float(self.efficiency), 1e-6)
        simple_wear_throughput_step = self.sample_time * (
            eff_safe * max(0.0, float(p_charge)) + max(0.0, float(p_discharge)) / eff_safe
        )
        simple_wear_cost_step = float(self.simple_wear["coeff"] * simple_wear_throughput_step)
        record["wear_cost_simple_coeff"] = float(self.simple_wear["coeff"])
        record["wear_throughput_simple_step_kwh"] = simple_wear_throughput_step
        record["wear_cost_simple_step"] = simple_wear_cost_step
        record["wear_cost_simple_horizon"] = simple_wear_cost_horizon

        if self.wear is not None:
            wear_cost_horizon = np.nan
            if self.wear["cost_expr"].value is not None:
                wear_cost_horizon = float(self.wear["cost_expr"].value)
            wear_cost_step = np.nan
            wear_delta = self.wear["delta_phi"].value
            if wear_delta is not None:
                wear_cost_step = float(self.wear["coeff"] * wear_delta[0, 0])
            record["wear_cost_step"] = wear_cost_step
            record["wear_cost_horizon"] = wear_cost_horizon

        if self.log_full_forecast:
            record["load_forecast"] = load_hat.flatten().tolist()
            record["pv_forecast"] = pv_hat.flatten().tolist()
            record["price_buy_forecast"] = price_buy_hat.flatten().tolist()
            record["price_sell_forecast"] = price_sell_hat.flatten().tolist()

        self.records.append(record)
        if self._log_fields is None:
            self._log_fields = list(record.keys())

        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._log_fields)
            if not self._log_header_written:
                writer.writeheader()
                self._log_header_written = True
            writer.writerow(record)



