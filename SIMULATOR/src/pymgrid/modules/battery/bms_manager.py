import copy
import math
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class BatteryBMSManager:
    """Battery-side estimator/manager decoupled from the physical transition model.

    The transition model is used only for physical state updates (Voc, R0, current,
    voltage, SoC/SoE). This manager handles state estimation and bookkeeping:
    SOH evolution, wear cost, replacement flag and BMS history.
    """

    def __init__(self, transition_model):
        self._estimation_model = copy.deepcopy(transition_model)

        self.chemistry = str(getattr(transition_model, "chemistry", "UNKNOWN")).upper()
        self.np_batt = float(getattr(transition_model, "np_batt", 1.0) or 1.0)
        self.delta_t_hours = float(getattr(transition_model, "delta_t_hours", 1.0) or 1.0)
        self.temperature_c = float(getattr(transition_model, "temperature_c", 25.0) or 25.0)

        # Wear parameters are intentionally managed here (not inside transition model).
        self.wear_a = getattr(transition_model, "wear_a", None)
        self.wear_b = getattr(transition_model, "wear_b", None)
        self.wear_c = getattr(transition_model, "wear_c", None)
        self.wear_B = getattr(transition_model, "wear_B", None)
        self.wear_use_temperature = bool(getattr(transition_model, "wear_use_temperature", False))
        self.wear_temp_coeff = float(getattr(transition_model, "wear_temp_coeff", 0.0035) or 0.0035)
        self.wear_temp_ref_c = float(getattr(transition_model, "wear_temp_ref_c", 25.0) or 25.0)

        self.replacement_threshold = 0.699

        self.reset()

    def reset(self, current_step=0, soc=None, soh=1.0):
        soc_value = 0.0 if soc is None else float(soc)
        self.soe = soc_value
        self.soc = soc_value
        self.soh = float(soh)
        self.last_wear_cost = 0.0
        self.last_dynamic_efficiency = None
        self.last_current_a = 0.0
        self.last_voltage_v = None
        self.last_voc_v = None
        self.estimated_voc_v = None
        self.estimated_r0_ohm = None
        self.last_temperature_c = self.temperature_c
        self.battery_replaced = False
        self._soh_07_logged = False
        self._transition_history = []

        self._sync_model_state(self._estimation_model, soc=soc_value, soh=float(soh), current_step=current_step)

    def _sync_model_state(self, transition_model, soc, soh, current_step):
        try:
            transition_model.reset(current_step=current_step, soc=soc, soh=soh)
        except TypeError:
            try:
                transition_model.reset(current_step=current_step, soc=soc)
            except Exception:
                pass
        except Exception:
            pass

        if hasattr(transition_model, "soc"):
            transition_model.soc = soc
        if hasattr(transition_model, "soe"):
            transition_model.soe = soc
        if hasattr(transition_model, "soh"):
            transition_model.soh = soh
        if hasattr(transition_model, "v_prev") and hasattr(transition_model, "_interp_voc_r0"):
            try:
                temp_c = float(getattr(transition_model, "temperature_c", self.temperature_c))
                voc, _ = transition_model._interp_voc_r0(soc, temp_c, soh)
                transition_model.v_prev = voc
            except Exception:
                pass

    def sync_with_transition_model(self, transition_model, *, current_step=None, state_dict=None, sync_estimator=False):
        """Single synchronization point between BMS and transition models.

        - Always aligns runtime model SOH with the BMS SOH.
        - Optionally synchronizes the BMS estimator copy to runtime state.
        """
        state_dict = state_dict or {}
        runtime_soc = getattr(transition_model, "soc", state_dict.get("soc", self.soc))
        runtime_soc = float(runtime_soc if runtime_soc is not None else self.soc)
        runtime_soh = float(state_dict.get("bms_soh", state_dict.get("soh", self.soh)) or self.soh)

        if hasattr(transition_model, "soh"):
            transition_model.soh = runtime_soh

        if sync_estimator:
            step = 0 if current_step is None else int(current_step)
            self._sync_model_state(
                self._estimation_model,
                soc=runtime_soc,
                soh=runtime_soh,
                current_step=step,
            )
            if hasattr(self._estimation_model, "v_prev") and hasattr(transition_model, "v_prev"):
                runtime_v_prev = getattr(transition_model, "v_prev", None)
                if runtime_v_prev is not None:
                    self._estimation_model.v_prev = runtime_v_prev

        return {"soc": runtime_soc, "soh": runtime_soh}

    def _non_state_transition(self,
                              transition_model,
                              *,
                              external_energy_change,
                              min_capacity,
                              max_capacity,
                              max_charge,
                              max_discharge,
                              min_soc,
                              max_soc,
                              efficiency,
                              battery_cost_cycle,
                              current_step,
                              state_dict,
                              preview_transition,
                              limit_transition):
        model = self._estimation_model
        temp_state = dict(state_dict or {})
        temp_state["bms_soh"] = float(self.soh)

        sync_state = self.sync_with_transition_model(
            transition_model=transition_model,
            current_step=current_step,
            state_dict=temp_state,
            sync_estimator=True,
        )

        # Fallback for non-custom transition models.
        if not (hasattr(model, "_interp_voc_r0") and hasattr(model, "_dynamic_efficiency")):
            return model(
                external_energy_change=external_energy_change,
                min_capacity=min_capacity,
                max_capacity=max_capacity,
                max_charge=max_charge,
                max_discharge=max_discharge,
                min_soc=min_soc,
                max_soc=max_soc,
                efficiency=efficiency,
                battery_cost_cycle=battery_cost_cycle,
                current_step=current_step,
                state_dict=temp_state,
                state_update=False,
                preview_transition=bool(preview_transition),
                limit_transition=bool(limit_transition),
            )

        temperature_c = float(temp_state.get("temperature_c", getattr(model, "temperature_c", self.temperature_c)))
        delta_t = max(float(getattr(model, "delta_t_hours", self.delta_t_hours) or self.delta_t_hours), 1e-9)
        soh_for_step = float(sync_state["soh"])
        soc = float(getattr(model, "soc", temp_state.get("soc", sync_state["soc"])))
        soe = float(getattr(model, "soe", temp_state.get("soc", self.soe)))

        voc, r0 = model._interp_voc_r0(soc, temperature_c, soh_for_step)
        v_prev = getattr(model, "v_prev", None)
        if v_prev is None:
            v_prev = voc

        power_kw = -external_energy_change / delta_t
        current_a = 1000.0 * power_kw / max(v_prev, 1e-9)
        v_batt = max(voc - r0 * current_a, 1e-6)
        if current_step == 0:
            dyn_eta = efficiency
        else:
            dyn_eta = max(1e-9, model._dynamic_efficiency(current_a, voc, r0, v_batt))

        if preview_transition:
            soe_new = soe - (current_a * voc * delta_t / 1000.0) / max(max_capacity, 1e-9)
            return (soe_new - soe) * max_capacity

        if limit_transition:
            if external_energy_change >= 0:
                return external_energy_change * dyn_eta
            return external_energy_change / dyn_eta

        raise ValueError("BMS non-state transition requires preview_transition or limit_transition.")

    def preview_transition(self,
                           transition_model,
                           *,
                           external_energy_change,
                           min_capacity,
                           max_capacity,
                           max_charge,
                           max_discharge,
                           min_soc,
                           max_soc,
                           efficiency,
                           battery_cost_cycle,
                           current_step,
                           state_dict):
        """Preview transition using the BMS copy model, without mutating live physical state."""
        return self._non_state_transition(
            transition_model=transition_model,
            external_energy_change=external_energy_change,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            max_charge=max_charge,
            max_discharge=max_discharge,
            min_soc=min_soc,
            max_soc=max_soc,
            efficiency=efficiency,
            battery_cost_cycle=battery_cost_cycle,
            current_step=current_step,
            state_dict=state_dict,
            preview_transition=True,
            limit_transition=False,
        )

    def limit_transition(self,
                         transition_model,
                         *,
                         external_energy_change,
                         min_capacity,
                         max_capacity,
                         max_charge,
                         max_discharge,
                         min_soc,
                         max_soc,
                         efficiency,
                         battery_cost_cycle,
                         current_step,
                         state_dict):
        """Limit-transition evaluation using the BMS copy model."""
        return self._non_state_transition(
            transition_model=transition_model,
            external_energy_change=external_energy_change,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            max_charge=max_charge,
            max_discharge=max_discharge,
            min_soc=min_soc,
            max_soc=max_soc,
            efficiency=efficiency,
            battery_cost_cycle=battery_cost_cycle,
            current_step=current_step,
            state_dict=state_dict,
            preview_transition=False,
            limit_transition=True,
        )

    def _update_after_real_transition(self, module, *, external_energy_change, soe_previous):
        self.update_after_transition(
            module.battery_transition_model,
            current_step=getattr(module, "_current_step", 0),
            external_energy_change=external_energy_change,
            soe_previous=soe_previous,
            min_soc=module.min_soc,
            max_soc=module.max_soc,
            state_dict=module.transition_kwargs().get("state_dict", {}),
        )

    def estimate_dynamic_external_limits(self, module):
        """Compute admissible external limits without clipping preview physics."""
        available_discharge = max(0.0, module._current_charge - module.operative_min_capacity)
        available_charge = max(0.0, module.operative_max_capacity - module._current_charge)

        internal_discharge_allowed = min(float(module._internal_max_discharge), available_discharge)
        internal_charge_allowed = min(float(module._internal_max_charge), available_charge)

        max_production = float(module.model_transition(internal_discharge_allowed, limit_transition=True))
        max_consumption = float(-module.model_transition(-internal_charge_allowed, limit_transition=True))
        return max_consumption, max_production

    def _get_real_soe_tolerance_band(self):
        """Return (soft, hard) SoE tolerances for real-transition checks."""
        soft_tol = float(os.environ.get("PYMGRID_BATTERY_REAL_SOE_SOFT_TOL", "6e-3"))
        hard_tol = float(os.environ.get("PYMGRID_BATTERY_REAL_SOE_HARD_TOL", "2e-2"))
        return soft_tol, hard_tol

    def _get_post_transition_charge_tolerances_kwh(self, module):
        """Project SoE tolerances to charge-domain tolerances [kWh]."""
        soe_soft_tol, soe_hard_tol = self._get_real_soe_tolerance_band()
        effective_capacity = max(float(module.effective_capacity), 1e-9)
        return float(soe_soft_tol * effective_capacity), float(soe_hard_tol * effective_capacity)

    def _handle_post_transition_charge_violation(
        self,
        module,
        *,
        side,
        delta_kwh,
        soft_tol_kwh,
        hard_tol_kwh,
        energy_change,
    ):
        """Log post-transition charge limit violation and raise only on hard excursions."""
        transition_model = getattr(module, "_battery_transition_model", None)
        tm_soc = getattr(transition_model, "soc", None) if transition_model is not None else None
        tm_soe = getattr(transition_model, "soe", None) if transition_model is not None else None
        tm_soh = getattr(transition_model, "soh", None) if transition_model is not None else None
        current_step = getattr(module, "_current_step", None)

        is_lower = side == "lower"
        severity = "hard" if delta_kwh > hard_tol_kwh else "soft"
        title = "charge below operative_min" if is_lower else "charge above operative_max"
        bound_label = "operative_min" if is_lower else "operative_max"
        delta_label = "under_min_kwh" if is_lower else "over_max_kwh"
        bound_value = module.operative_min_capacity if is_lower else module.operative_max_capacity
        no_clamp_msg = "below operative_min" if is_lower else "above operative_max"

        items = [
            ("step", current_step),
            ("current_charge", module._current_charge),
            (bound_label, bound_value),
            (delta_label, delta_kwh),
            ("soft_tol_kwh", soft_tol_kwh),
            ("hard_tol_kwh", hard_tol_kwh),
            ("tolerance_severity", severity),
            ("energy_change", energy_change),
            ("effective_capacity", module.effective_capacity),
            ("max_capacity", module.max_capacity),
            ("min_soc", module.min_soc),
            ("max_soc", module.max_soc),
            ("soh_model", tm_soh),
            ("tm_soc", tm_soc),
            ("tm_soe", tm_soe),
            ("last_soh_before", getattr(module, "_debug_last_soh_before", None)),
            ("last_soh_after", getattr(module, "_debug_last_soh_after", None)),
            ("last_external_request", getattr(module, "_debug_last_external_request", None)),
            ("last_executed_external", getattr(module, "_debug_last_executed_external", None)),
            ("last_internal_energy_change", getattr(module, "_debug_last_internal_energy_change", None)),
        ]
        if not is_lower:
            items.extend(
                [
                    ("prev_current_charge", getattr(module, "_debug_last_prev_current_charge", None)),
                    ("prev_soc", getattr(module, "_debug_last_prev_soc", None)),
                ]
            )

        module._debug_block(
            f"[BatteryModule._update_state] {title}",
            items,
            color="red",
            force=True,
        )
        module._debug_message(
            f"[BatteryModule._update_state] {title} at step={current_step} "
            f"current_charge={module._current_charge} {bound_label}={bound_value} "
            f"{delta_label}={delta_kwh} tolerance_severity={severity} "
            f"energy_change={energy_change} effective_capacity={module.effective_capacity} "
            f"max_capacity={module.max_capacity} min_soc={module.min_soc} max_soc={module.max_soc} "
            f"soh_model={tm_soh} tm_soc={tm_soc} tm_soe={tm_soe} "
            f"last_soh_before={getattr(module, '_debug_last_soh_before', None)} "
            f"last_soh_after={getattr(module, '_debug_last_soh_after', None)} "
            f"last_external_request={getattr(module, '_debug_last_external_request', None)} "
            f"last_executed_external={getattr(module, '_debug_last_executed_external', None)} "
            f"last_internal_energy_change={getattr(module, '_debug_last_internal_energy_change', None)} "
            f"prev_current_charge={getattr(module, '_debug_last_prev_current_charge', None)} "
            f"prev_soc={getattr(module, '_debug_last_prev_soc', None)}",
            force=True,
        )
        module._debug_message(
            f"[BatteryModule._update_state] no clamp applied ({no_clamp_msg}).",
            force=True,
        )

        if delta_kwh > hard_tol_kwh:
            relation = "below" if is_lower else "above"
            raise ValueError(
                "[BatteryModule._update_state] hard limit violation: "
                f"current_charge={module._current_charge} is {relation} {bound_label}={bound_value} "
                f"by {delta_kwh} kWh (> hard_tol={hard_tol_kwh} kWh)."
            )

    def _update_module_state(self, module, energy_change):
        soft_tol, hard_tol = self._get_post_transition_charge_tolerances_kwh(module)
        module._current_charge += energy_change

        if module._current_charge < module.operative_min_capacity - soft_tol:
            under_min_kwh = float(module.operative_min_capacity - module._current_charge)
            self._handle_post_transition_charge_violation(
                module,
                side="lower",
                delta_kwh=under_min_kwh,
                soft_tol_kwh=soft_tol,
                hard_tol_kwh=hard_tol,
                energy_change=energy_change,
            )

        if module._current_charge > module.operative_max_capacity + soft_tol:
            over_max_kwh = float(module._current_charge - module.operative_max_capacity)
            self._handle_post_transition_charge_violation(
                module,
                side="upper",
                delta_kwh=over_max_kwh,
                soft_tol_kwh=soft_tol,
                hard_tol_kwh=hard_tol,
                energy_change=energy_change,
            )

        module._soc = module._current_charge / module.effective_capacity

    def _apply_preview_internal_limit(
        self,
        module,
        *,
        executed_external,
        direction,
        internal_limit,
        safety_tol,
        mode_name,
        step_msg,
    ):
        """Apply preview-time internal (power/current) limit clipping."""
        internal_energy_change = module.model_transition(direction * executed_external, preview_transition=True)
        internal_energy_change_requested = float(internal_energy_change)
        clipped_internal_limit = False

        internal_magnitude = direction * internal_energy_change
        overshoot = internal_magnitude - internal_limit
        if overshoot > safety_tol:
            scale = internal_limit / internal_magnitude if internal_magnitude != 0 else 0.0
            module._debug_block(
                f"[BatteryModule.update{step_msg}] {mode_name} overshoot",
                [
                    ("internal_energy_change", internal_energy_change),
                    ("internal_limit", internal_limit),
                    ("scale", scale),
                    ("executed_external", executed_external),
                ],
                color="yellow",
            )
            clipped_internal_limit = True
            executed_external *= max(0.0, min(1.0, scale))
            module.num_overshoots += 1
            internal_energy_change = direction * internal_limit

        return executed_external, internal_energy_change, internal_energy_change_requested, clipped_internal_limit

    def _apply_preview_soe_limit(
        self,
        module,
        *,
        executed_external,
        direction,
        as_source,
        safety_tol,
        mode_name,
        step_msg,
    ):
        """Apply preview-time SoE window clipping."""
        internal_energy_change = module.model_transition(direction * executed_external, preview_transition=True)
        next_charge = module._current_charge + internal_energy_change
        clipped_soe_limit = False

        if as_source:
            soe_violated = next_charge < module.operative_min_capacity - safety_tol
            available = max(0.0, module._current_charge - module.operative_min_capacity)
            limit_violation = max(0.0, module.operative_min_capacity - next_charge)
            limit_label = "operative_min"
            violation_label = "shortage"
        else:
            soe_violated = next_charge > module.operative_max_capacity + safety_tol
            available = max(0.0, module.operative_max_capacity - module._current_charge)
            limit_violation = max(0.0, next_charge - module.operative_max_capacity)
            limit_label = "operative_max"
            violation_label = "surplus"

        if soe_violated:
            module._debug_block(
                f"[BatteryModule.update{step_msg}] {mode_name} next_charge out of limits",
                [
                    ("next_charge", next_charge),
                    (limit_label, module.operative_min_capacity if as_source else module.operative_max_capacity),
                    ("available", available),
                    (violation_label, limit_violation),
                    ("current_charge", module._current_charge),
                ],
                color="yellow",
            )
            clipped_soe_limit = True
            module.num_overshoots += 1
            if available <= safety_tol:
                executed_external = 0.0
                internal_energy_change = 0.0
            else:
                internal_magnitude = direction * internal_energy_change
                scale = available / internal_magnitude if internal_magnitude != 0 else 0.0
                executed_external *= max(0.0, min(1.0, scale))
                internal_energy_change = direction * available

        return executed_external, internal_energy_change, clipped_soe_limit, limit_violation

    def update_module(self, module, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, "Must act as either source or sink but not both or neither."

        current_step = getattr(module, "_current_step", None)
        step_msg = f" [step {current_step}]" if current_step is not None else ""
        prev_effective_capacity = module.effective_capacity
        prev_operative_min_capacity = module.operative_min_capacity
        prev_operative_max_capacity = module.operative_max_capacity
        prev_current_charge = module._current_charge
        prev_soc = module._soc
        overshoots_before = int(module.num_overshoots)
        safety_tol = 1e-6

        module._debug_last_external_request = external_energy_change
        module._debug_last_as_source = as_source
        module._debug_last_as_sink = as_sink

        module._internal_max_charge, module._internal_max_discharge = module._internal_bounds()

        if as_source:
            mode_name = "discharge"
            info_key = "provided_energy"
            direction = -1.0
            internal_limit = module._internal_max_discharge
        else:
            mode_name = "charge"
            info_key = "absorbed_energy"
            direction = 1.0
            internal_limit = module._internal_max_charge

        requested_external_signed = float(external_energy_change if as_source else -external_energy_change)
        executed_external = float(external_energy_change)
        clipped_internal_limit = False
        clipped_soe_limit = False
        transition_model = getattr(module, "battery_transition_model", None)
        true_soc = getattr(transition_model, "soc", None) if transition_model is not None else None
        true_soe = getattr(transition_model, "soe", None) if transition_model is not None else None

        module._debug_block(
            f"[BatteryModule.update{step_msg}] start",
            [
                ("external_energy_change", external_energy_change),
                ("as_source", as_source),
                ("as_sink", as_sink),
                ("current_charge", module._current_charge),
                ("soc", module._soc),
                ("true_soc", true_soc),
                ("true_soe", true_soe),
                ("effective_capacity", module.effective_capacity),
                ("operative_min", module.operative_min_capacity),
                ("operative_max", module.operative_max_capacity),
            ],
            color="cyan",
        )

        self.sync_with_transition_model(module.battery_transition_model)
        soh = float(getattr(self, "soh", 1.0) or 1.0)
        soe = float(getattr(self, "soe", module._soc) or module._soc)
        module._debug_last_soh_before = soh

        module.effective_capacity = module.max_capacity * soh
        module._current_charge = soe * module.effective_capacity
        module.operative_max_capacity = module.effective_capacity * module.max_soc
        module.operative_min_capacity = module.effective_capacity * module.min_soc
        if (
            not np.isclose(prev_effective_capacity, module.effective_capacity)
            or not np.isclose(prev_operative_max_capacity, module.operative_max_capacity)
            or not np.isclose(prev_operative_min_capacity, module.operative_min_capacity)
        ):
            module._debug_block(
                f"[BatteryModule.update{step_msg}] capacities updated",
                [
                    ("soh", soh),
                    ("effective_capacity", f"{prev_effective_capacity} -> {module.effective_capacity}"),
                    ("operative_min", f"{prev_operative_min_capacity} -> {module.operative_min_capacity}"),
                    ("operative_max", f"{prev_operative_max_capacity} -> {module.operative_max_capacity}"),
                    ("current_charge", module._current_charge),
                ],
                color="yellow",
            )
        if (
            module._current_charge > module.operative_max_capacity + safety_tol
            or module._current_charge < module.operative_min_capacity - safety_tol
        ):
            module._debug_block(
                f"[BatteryModule.update{step_msg}] pre-step out-of-bounds after SOH update",
                [
                    ("current_charge", module._current_charge),
                    ("operative_min", module.operative_min_capacity),
                    ("operative_max", module.operative_max_capacity),
                    ("prev_current_charge", prev_current_charge),
                    ("prev_effective_capacity", prev_effective_capacity),
                    ("prev_operative_min", prev_operative_min_capacity),
                    ("prev_operative_max", prev_operative_max_capacity),
                ],
                color="yellow",
                force=True,
            )
            module._debug_message(
                f"[BatteryModule.update{step_msg}] pre-step charge out of bounds after SOH update",
                force=True,
            )

        pre_step_max_external_discharge = float(module.max_production)
        pre_step_max_external_charge = float(module.max_consumption)

        (
            executed_external,
            internal_energy_change,
            internal_energy_change_requested,
            clipped_internal_limit,
        ) = self._apply_preview_internal_limit(
            module,
            executed_external=executed_external,
            direction=direction,
            internal_limit=internal_limit,
            safety_tol=safety_tol,
            mode_name=mode_name,
            step_msg=step_msg,
        )
        (
            executed_external,
            internal_energy_change,
            clipped_soe_limit,
            limit_violation,
        ) = self._apply_preview_soe_limit(
            module,
            executed_external=executed_external,
            direction=direction,
            as_source=as_source,
            safety_tol=safety_tol,
            mode_name=mode_name,
            step_msg=step_msg,
        )

        internal_energy_change = module.model_transition(direction * executed_external, state_update=True)
        self._update_after_real_transition(
            module,
            external_energy_change=direction * executed_external,
            soe_previous=soe,
        )

        replaced = module._handle_battery_replacement()
        if replaced:
            internal_energy_change = 0.0
            executed_external = 0.0

        soh_after = float(getattr(self, "soh", 1.0) or 1.0)
        module._debug_last_soh_after = soh_after
        if not np.isclose(soh_after, soh):
            module._debug_block(
                f"[BatteryModule.update{step_msg}] soh changed during {mode_name}",
                [
                    ("soh_before", soh),
                    ("soh_after", soh_after),
                ],
                color="cyan",
            )

        if as_source:
            assert internal_energy_change <= 0 and (
                -internal_energy_change <= module._internal_max_discharge
                or np.isclose(-internal_energy_change, module._internal_max_discharge)
            )
        else:
            assert internal_energy_change >= 0 and (
                internal_energy_change <= module._internal_max_charge
                or np.isclose(internal_energy_change, module._internal_max_charge)
            )

        module._debug_last_internal_energy_change = internal_energy_change
        module._debug_last_executed_external = executed_external
        module._debug_last_effective_capacity = module.effective_capacity
        module._debug_last_operative_min_capacity = module.operative_min_capacity
        module._debug_last_operative_max_capacity = module.operative_max_capacity
        module._debug_last_prev_current_charge = prev_current_charge
        module._debug_last_prev_soc = prev_soc

        self._update_module_state(module, internal_energy_change)
        reward = -1.0 * module.get_cost(internal_energy_change)
        wear_cost = float(getattr(self, "last_wear_cost", 0.0) or 0.0)
        cycle_cost = float(np.abs(internal_energy_change) * module.battery_cost_cycle)
        executed_external_signed = float(executed_external if as_source else -executed_external)
        transition_model_after = getattr(module, "battery_transition_model", None)
        true_soc_after = float(getattr(transition_model_after, "soc", np.nan))
        true_soe_after = float(getattr(transition_model_after, "soe", np.nan))
        info = {
            info_key: float(executed_external),
            "requested_external_signed": requested_external_signed,
            "executed_external_signed": executed_external_signed,
            "requested_external_abs": float(external_energy_change),
            "executed_external_abs": float(executed_external),
            "internal_energy_change_requested": float(internal_energy_change_requested),
            "internal_energy_change_executed": float(internal_energy_change),
            "clip_internal_limit": int(bool(clipped_internal_limit)),
            "clip_soe_limit": int(bool(clipped_soe_limit)),
            "soe_violation": float(bool(clipped_soe_limit)),
            "soe_violation_kwh": float(limit_violation if clipped_soe_limit else 0.0),
            "step_overshoots": int(module.num_overshoots - overshoots_before),
            "overshoots_total": int(module.num_overshoots),
            "current_charge_before": float(prev_current_charge),
            "current_charge_after": float(module._current_charge),
            "soc_before": float(prev_soc),
            "soc_after": float(module._soc),
            "true_soc": true_soc_after,
            "true_soe": true_soe_after,
            "effective_capacity": float(module.effective_capacity),
            "operative_min_capacity": float(module.operative_min_capacity),
            "operative_max_capacity": float(module.operative_max_capacity),
            "internal_max_charge": float(module._internal_max_charge),
            "internal_max_discharge": float(module._internal_max_discharge),
            "max_external_charge_pre": float(pre_step_max_external_charge),
            "max_external_discharge_pre": float(pre_step_max_external_discharge),
            "battery_step_reward": float(reward),
            "battery_step_cost": float(-reward),
            "cycle_cost_step": cycle_cost,
            "wear_cost_step": wear_cost,
            "transition_model": module.battery_transition_model.__class__.__name__,
        }
        return reward, False, info

    def _acc(self, dod):
        dod_clamped = max(dod, 1e-6)
        return self.wear_a * (dod_clamped ** (-self.wear_b)) * math.exp(-self.wear_c * dod_clamped)

    def _alpha_temp(self, temperature_c):
        temp_c = self.temperature_c if temperature_c is None else float(temperature_c)
        return math.exp(self.wear_temp_coeff * abs(temp_c - self.wear_temp_ref_c))

    def _compute_wear_cost(self, soe_previous, soe_current, temperature_c=None):
        if None in (self.wear_a, self.wear_b, self.wear_c, self.wear_B):
            return 0.0

        eps = 1e-6
        dod_prev = max(1.0 - float(soe_previous), eps)
        dod_curr = max(1.0 - float(soe_current), eps)

        acc_prev = self._acc(dod_prev)
        acc_curr = self._acc(dod_curr)

        inv_prev = 1.0 / max(acc_prev, eps)
        inv_curr = 1.0 / max(acc_curr, eps)

        alpha = self._alpha_temp(temperature_c) if self.wear_use_temperature else 1.0
        return alpha * (self.wear_B / 2.0) * abs(inv_curr - inv_prev)

    def _validate_limits(self, *, soc, soe, min_soc, max_soc, current_step, external_energy_change):
        if min_soc is None or max_soc is None:
            return

        soft_tol, hard_tol = self._get_real_soe_tolerance_band()

        if not np.isfinite(soc) or not np.isfinite(soe):
            raise ValueError(
                f"[BatteryBMSManager] Non-finite battery state at step={current_step}: "
                f"soc={soc}, soe={soe}, external_energy_change={external_energy_change}"
            )

        under_min = max(0.0, float(min_soc) - float(soe))
        over_max = max(0.0, float(soe) - float(max_soc))
        violation = max(under_min, over_max)

        if violation > hard_tol:
            raise ValueError(
                f"[BatteryBMSManager] Real transition exceeded limits at step={current_step}: "
                f"soe={soe} not in [{min_soc}, {max_soc}] "
                f"(violation={violation}, hard_tol={hard_tol}); "
                f"external_energy_change={external_energy_change}"
            )

        if violation > soft_tol and os.environ.get("PYMGRID_BATTERY_DEBUG"):
            print(
                "[BatteryBMSManager] soft SoE limit excursion "
                f"at step={current_step}: soe={soe}, range=[{min_soc}, {max_soc}], "
                f"violation={violation}, soft_tol={soft_tol}, hard_tol={hard_tol}, "
                f"external_energy_change={external_energy_change}"
            )

    def update_after_transition(self,
                                transition_model,
                                *,
                                current_step,
                                external_energy_change,
                                soe_previous,
                                min_soc=None,
                                max_soc=None,
                                state_dict=None):
        # Default: no replacement trigger on this step.
        self.battery_replaced = False

        state_dict = state_dict or {}
        delta_t = max(float(getattr(transition_model, "delta_t_hours", self.delta_t_hours) or self.delta_t_hours), 1e-9)
        self.delta_t_hours = delta_t

        self.last_temperature_c = float(
            state_dict.get(
                "temperature_c",
                getattr(transition_model, "last_temperature_c", getattr(transition_model, "temperature_c", self.temperature_c)),
            )
        )
        self.temperature_c = self.last_temperature_c

        self.last_current_a = float(getattr(transition_model, "current_a", 0.0) or 0.0)
        self.last_voc_v = getattr(transition_model, "voc_v", None)
        self.last_voltage_v = getattr(transition_model, "voltage_v", None)
        self.last_dynamic_efficiency = float(
            getattr(
                transition_model,
                "last_dynamic_efficiency",
                getattr(transition_model, "dyn_eta", 1.0),
            )
            or 1.0
        )

        soc_live = float(getattr(transition_model, "soc", self.soc) or self.soc)
        soe_live = float(getattr(transition_model, "soe", self.soe) or self.soe)
        self._validate_limits(
            soc=soc_live,
            soe=soe_live,
            min_soc=min_soc,
            max_soc=max_soc,
            current_step=current_step,
            external_energy_change=external_energy_change,
        )
        self.soc = soc_live
        self.soe = soe_live

        # SOH is computed by the real transition model.
        self.soh = float(getattr(transition_model, "soh", self.soh) or self.soh)

        if self.soh <= self.replacement_threshold:
            self.battery_replaced = True
            self.soh = 1.0

        # BMS estimator uses a copy of the physical model for internal estimates.
        if hasattr(self._estimation_model, "soc"):
            self._estimation_model.soc = self.soc
        if hasattr(self._estimation_model, "soe"):
            self._estimation_model.soe = self.soe
        if hasattr(self._estimation_model, "soh"):
            self._estimation_model.soh = self.soh
        if hasattr(self._estimation_model, "_interp_voc_r0"):
            try:
                voc_est, r0_est = self._estimation_model._interp_voc_r0(
                    self.soc,
                    self.last_temperature_c,
                    self.soh,
                )
                self.estimated_voc_v = float(voc_est)
                self.estimated_r0_ohm = float(r0_est)
            except Exception:
                self.estimated_voc_v = None
                self.estimated_r0_ohm = None

        self.last_wear_cost = self._compute_wear_cost(
            soe_previous=soe_previous,
            soe_current=self.soe,
            temperature_c=self.last_temperature_c,
        )

        power_kw = -float(external_energy_change) / delta_t
        self._transition_history.append(
            {
                "time_hours": float(
                    current_step * self.delta_t_hours
                    if current_step is not None
                    else len(self._transition_history) * self.delta_t_hours
                ),
                "soc": float(self.soc),
                "soe": float(self.soe),
                "soh": float(self.soh),
                "current_a": float(self.last_current_a),
                "voc_v": float(self.last_voc_v) if self.last_voc_v is not None else None,
                "voc_est_v": float(self.estimated_voc_v) if self.estimated_voc_v is not None else None,
                "r0_est_ohm": float(self.estimated_r0_ohm) if self.estimated_r0_ohm is not None else None,
                "voltage_v": float(self.last_voltage_v) if self.last_voltage_v is not None else None,
                "power_kw": float(-power_kw),
                "efficiency": float(self.last_dynamic_efficiency),
                "wear_cost": float(self.last_wear_cost),
                "replaced": int(bool(self.battery_replaced)),
            }
        )

        if os.environ.get("PYMGRID_BATTERY_DEBUG") and (not self._soh_07_logged) and self.soh <= 0.7:
            time_hours = (
                current_step * self.delta_t_hours
                if current_step is not None
                else len(self._transition_history) * self.delta_t_hours
            )
            days = time_hours / 24.0
            print(f"SOH=0.7 reached after {days:.2f} days")
            self._soh_07_logged = True

    def clear_replacement_flag(self):
        self.battery_replaced = False

    def get_transition_history(self):
        return list(self._transition_history)

    def save_transition_history(self, history_path: str):
        """Persist BMS transition history to a JSON file."""
        import json

        with open(history_path, "w", encoding="utf-8") as fp:
            json.dump(self._transition_history, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def load_transition_history(history_path: str):
        """Load a BMS transition history previously saved with save_transition_history."""
        import json

        with open(history_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def plot_transition_history(self, save_path: str = None, show: bool = True, history=None):
        """Plot BMS-estimated and measured battery metrics over time."""
        history_to_plot = history if history is not None else self._transition_history

        if not history_to_plot:
            raise ValueError("No transition history to plot. Run transitions before plotting.")

        def _valid_series(key):
            time_axis, values = [], []
            for entry in history_to_plot:
                value = entry.get(key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                time_axis.append(entry["time_hours"])
                values.append(value)
            return time_axis, values

        metric_specs = {
            "soc": ("State of charge", "State of charge [0-1]", "tab:blue"),
            "soe": ("State of energy", "State of energy [0-1]", "tab:green"),
            "soh": ("State of health", "State of health [0-1]", "tab:brown"),
            "voc_v": ("Measured open-circuit voltage", "Voc [V]", "tab:olive"),
            "voc_est_v": ("Estimated open-circuit voltage", "Voc est [V]", "tab:cyan"),
            "r0_est_ohm": ("Estimated internal resistance", "R0 est [ohm]", "tab:pink"),
            "voltage_v": ("Battery terminal voltage", "Voltage [V]", "tab:red"),
            "current_a": ("Battery current", "Current [A]", "tab:gray"),
            "power_kw": ("Battery power", "Power [kW]", "tab:purple"),
            "efficiency": ("Dynamic efficiency", "Efficiency [0-1]", "tab:orange"),
            "wear_cost": ("Wear cost", "Wear cost [-]", "tab:green"),
            "replaced": ("Battery replacement flag", "Replacement [0/1]", "tab:red"),
        }

        figures = {}

        def _build_save_path(base_path: str, suffix: str) -> str:
            path = Path(base_path)
            if path.suffix:
                return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))
            return f"{base_path}{suffix}"

        for key, (title, ylabel, color) in metric_specs.items():
            time_axis, values = _valid_series(key)
            if not values:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, values, linestyle="-", color=color)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{self.__class__.__name__} {title} over time")
            ax.grid(True)
            fig.tight_layout()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(_build_save_path(save_path, f"_{key}"), bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

            figures[key] = (fig, ax)

        if not figures:
            raise ValueError("No plottable metrics found in transition history.")

        return figures
