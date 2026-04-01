"""Nonlinear MPC using UnipiChemistryTransitionModel.

Single-shooting MPC that forward-simulates the Unipi transition model
for a given sequence of battery powers and optimizes it via scipy SLSQP.
This includes the nonlinear efficiency, Voc/R0, and SOH logic from the
Unipi model. Designed to be self-contained and minimally invasive to
the existing linear/MILP MPC implementation.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from src.pymgrid.modules.battery.transition_models.custom_transition_model import (
    CustomChemistryTransitionModel,
)


class NonlinearUnipiMPC:
    """
    Nonlinear MPC that uses UnipiChemistryTransitionModel for battery dynamics.

    Convention:
    - p_batt[k] > 0 => discharge (power delivered by battery to grid/load)
    - p_batt[k] < 0 => charge
    """

    def __init__(
        self,
        transition_model: CustomChemistryTransitionModel,
        horizon: int,
        delta_t_hours: float,
        max_capacity_kwh: float,
        min_capacity_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        p_max_import_kw: float,
        p_max_export_kw: float,
        target_soc: Optional[float] = None,
        wear_cost_weight: float = 0.0,
        soc_target_weight: float = 0.0,
    ) -> None:
        self.tm = transition_model
        self.h = horizon
        self.dt = delta_t_hours
        self.e_max = max_capacity_kwh
        self.e_min = min_capacity_kwh
        self.p_ch_max = max_charge_kw
        self.p_dis_max = max_discharge_kw
        self.p_imp_max = p_max_import_kw
        self.p_exp_max = p_max_export_kw
        self.target_soc = target_soc
        self.w_wear = wear_cost_weight
        self.w_soc_target = soc_target_weight

    def plan(
        self,
        load_forecast: np.ndarray,
        pv_forecast: np.ndarray,
        price_import: np.ndarray,
        price_export: np.ndarray,
        soc0: float,
        state_dict: Optional[Dict] = None,
        grid_status: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, Dict]:
        """
        Compute the optimal battery power plan.

        Returns (p_batt_0, full_plan, solver_log).
        """

        load = np.asarray(load_forecast)
        pv = np.asarray(pv_forecast)
        pi = np.asarray(price_import)
        pe = np.asarray(price_export)

        if grid_status is None:
            grid_status = np.ones_like(load)
        grid_status = np.asarray(grid_status)

        for vec in (load, pv, pi, pe, grid_status):
            if vec.shape[0] != self.h:
                raise ValueError("All input vectors must have length == horizon")

        x0 = np.zeros(self.h)  # initial guess: idle battery

        bounds = [(-self.p_ch_max, self.p_dis_max) for _ in range(self.h)]

        def objective(p_vec: np.ndarray) -> float:
            return self._simulate_cost(
                p_vec=p_vec,
                load=load,
                pv=pv,
                pi=pi,
                pe=pe,
                soc0=soc0,
                state_dict=state_dict,
                grid_status=grid_status,
            )

        res = minimize(
            fun=objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-4, "disp": False},
        )

        plan = res.x
        log = {"success": res.success, "status": res.message, "cost": res.fun}

        return float(plan[0]), plan, log

    def _simulate_cost(
        self,
        p_vec: np.ndarray,
        load: np.ndarray,
        pv: np.ndarray,
        pi: np.ndarray,
        pe: np.ndarray,
        soc0: float,
        state_dict: Optional[Dict],
        grid_status: np.ndarray,
    ) -> float:
        tm = copy.deepcopy(self.tm)

        soc = soc0
        soe = soc0
        state = dict(state_dict or {})
        state.setdefault("soc", soc)
        state.setdefault("current_charge", soc * self.e_max)

        total_cost = 0.0

        for k in range(self.h):
            p_batt = float(p_vec[k])

            penalty = 0.0
            if p_batt > self.p_dis_max:
                penalty += (p_batt - self.p_dis_max) ** 2 * 1e3
            if -p_batt > self.p_ch_max:
                penalty += (-p_batt - self.p_ch_max) ** 2 * 1e3

            # transition expects energy over the step (positive = charge)
            external_energy_change = -p_batt * self.dt
            tm.transition(
                external_energy_change=external_energy_change,
                min_capacity=self.e_min,
                max_capacity=self.e_max,
                max_charge=self.p_ch_max,
                max_discharge=self.p_dis_max,
                efficiency=1.0,
                battery_cost_cycle=0.0,
                current_step=k,
                state_dict=state,
                state_update=True,
            )

            soc = tm.soc
            soe = tm.soe
            state["soc"] = soc
            state["current_charge"] = soe * self.e_max

            residual = load[k] - pv[k] - p_batt
            imp = max(0.0, residual)
            exp = max(0.0, -residual)

            if grid_status[k] < 0.5:
                penalty += (imp + exp) ** 2 * 1e4
                imp = 0.0
                exp = 0.0
            else:
                if imp > self.p_imp_max:
                    penalty += (imp - self.p_imp_max) ** 2 * 1e3
                if exp > self.p_exp_max:
                    penalty += (exp - self.p_exp_max) ** 2 * 1e3

            energy_import = imp * self.dt
            energy_export = exp * self.dt

            wear = tm.get_wear_cost(
                soc_previous=soc,
                power_kw=p_batt,
                delta_t_hours=self.dt,
            )

            stage_cost = energy_import * pi[k] - energy_export * pe[k] + self.w_wear * wear
            total_cost += stage_cost + penalty

        if self.target_soc is not None:
            total_cost += self.w_soc_target * (soc - self.target_soc) ** 2

        return float(total_cost)
