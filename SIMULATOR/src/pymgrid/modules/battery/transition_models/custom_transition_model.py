import os
from typing import Tuple
import csv
import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import yaml
import builtins
from pathlib import Path

from .transition_model import BatteryTransitionModel


def _setup_print_tee():
    if not os.environ.get("PYMGRID_BATTERY_DEBUG"):
        return
    if getattr(builtins, "_pymgrid_print_tee_enabled", False):
        return
    log_path = os.environ.get("PYMGRID_BATTERY_LOG")
    if not log_path:
        log_path = str(Path.cwd() / "outputs" / "battery_debug.log")
    log_path = str(Path(log_path))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    original_print = builtins.print

    def _pick_color(text):
        lowered = text.lower()
        if any(word in lowered for word in ("error", "assert", "exceeds", "below", "out of bounds")):
            return "\x1b[31m"
        if "battery replaced" in lowered:
            return "\x1b[35m"
        if "clipped" in lowered:
            return "\x1b[33m"
        if "soh" in lowered:
            return "\x1b[36m"
        if "pre-step" in lowered:
            return "\x1b[33m"
        return ""

    def tee_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args) + end
        pretty = os.environ.get("PYMGRID_BATTERY_PRETTY", "1") == "1"
        out_text = text
        if pretty and "\x1b[" not in text:
            color = _pick_color(text)
            if color:
                out_text = f"{color}{out_text}\x1b[0m"
            if not out_text.startswith("\n"):
                out_text = "\n" + out_text
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(out_text)
        except OSError:
            pass
        original_print(out_text, end="")

    builtins.print = tee_print
    builtins._pymgrid_print_tee_enabled = True
    builtins._pymgrid_print_tee_path = log_path
    original_print(f"[pymgrid debug] logging prints to: {log_path}")


_setup_print_tee()


class CustomChemistryTransitionModel(BatteryTransitionModel):
    """Base class for UNIPI chemistry-aware battery transition models.

    This reproduces the Voc/R0 interpolation and dynamic efficiency logic of
    the "ALTRO SIMULATORE" ESS_UNIPI_* classes while keeping the standard
    :class:`BatteryTransitionModel` API. The model uses lookup tables loaded
    from MATLAB ``.mat`` files to compute the open-circuit voltage (Voc) and
    internal resistance (R0) as a function of state-of-charge (SoC) and
    temperature, then derives a step-level round-trip efficiency accordingly.

    Parameters
    ----------
    parameters_mat : str
        File name of the MATLAB parameter table to load (placed in the local
        ``data`` directory next to this module).
    reference_cell_capacity_ah : float
        Nominal capacity (in Ah) of the reference cell for which the R0 table
        was generated. Used to scale R0 when modelling packs with a different
        nominal capacity ``c_n``.
    nominal_cell_voltage : float
        Nominal voltage (in V) of the reference cell.
    ns_batt : int, default 1
        Cells in series.
    np_batt : int, default 1
        Cells in parallel.
    c_n : float, default None
        Nominal cell capacity (Ah) of the target pack. If None, defaults to
        ``reference_cell_capacity_ah``.
    temperature_c : float, default 25.0
        Pack temperature used for interpolation (clipped to the tabulated
        range).
    eta_inverter : float, default 1.0
        Inverter efficiency multiplier applied to the dynamic efficiency.
    delta_t_hours : float, default 1.0
        Duration (in hours) represented by each transition call. External
        energy is divided by ``delta_t_hours`` to obtain the requested power.
    debug_energy : bool, default False
        If True, prints detailed intermediate values used to compute
        ``internal_energy_change`` for both the ``dyn_eta`` and ``soe_new``
        calculation paths.
    disable_soh_degradation : bool, default False
        If True, disables SoH degradation and keeps SoH fixed to ``1.0``.
    wear_* parameters
        Legacy parameters retained for backwards compatibility. They are now
        consumed by the battery BMS manager, not by this physical transition
        model.
    """

    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 parameters_mat: str,
                 reference_cell_capacity_ah: float,
                 nominal_cell_voltage: float,
                 ns_batt: int = 87,
                 np_batt: int = 10,
                 c_n: float = None,
                 temperature_c: float = 25.0,
                 eta_inverter: float = 0.9,
                 delta_t_hours: float = 0.25, 
                 debug_energy: bool = True,
                 disable_soh_degradation: bool = False,
                 wear_a: float = None,
                 wear_b: float = None,
                 wear_c: float = None,
                 wear_B: float = None,
                 wear_use_temperature: bool = False,
                 wear_temp_coeff: float = 0.0035,
                 wear_temp_ref_c: float = 25.0,
                 soh: float = 1.0):
        
        self.parameters_mat = parameters_mat
        self.reference_cell_capacity_ah = reference_cell_capacity_ah
        self.nominal_cell_voltage = nominal_cell_voltage
        self.ns_batt = ns_batt
        self.np_batt = np_batt
        self.c_n = c_n if c_n is not None else reference_cell_capacity_ah
        self.temperature_c = temperature_c
        self.eta_inverter = eta_inverter
        self.delta_t_hours = delta_t_hours
        self.debug_energy = debug_energy
        self.disable_soh_degradation = bool(disable_soh_degradation)
        self.wear_a = wear_a
        self.wear_b = wear_b
        self.wear_c = wear_c
        self.wear_B = wear_B
        self.wear_use_temperature = wear_use_temperature
        self.wear_temp_coeff = wear_temp_coeff
        self.wear_temp_ref_c = wear_temp_ref_c
        self.dyn_eta = None
        self.debug_log_path = Path(__file__).resolve().parent / "debug_log.csv"

         # Determine chemistry from filename
        mat_basename = os.path.splitext(self.parameters_mat)[0].lower()
        if 'lfp' in mat_basename:
            self.chemistry = 'LFP'
        elif 'nca' in mat_basename:
            self.chemistry = 'NCA'
        elif 'nmc' in mat_basename:
            self.chemistry = 'NMC'
        else:
            self.chemistry = 'UNKNOWN'

        if self.disable_soh_degradation:
            soh = 1.0
        
        # LFP and NCA only support SOH=1.0
        if self.chemistry in ('LFP', 'NCA') and not np.isclose(soh, 1.0):
            raise ValueError(f"{self.chemistry} chemistry only supports SOH=1.0. Got SOH={soh}")
        

        self.soh = float(soh)  # State of Health

        self.nominal_energy_kwh = (
            self.c_n * self.nominal_cell_voltage * self.ns_batt * self.np_batt / 1000.0
        )

        self._last_voltage = None
        self._soe = None
        self.soc = None
        self.soe = None
        self.v_prev = None
        self.current_a = 0.0
        self.voc_v = None
        self.voltage_v = None
        self.last_r0_ohm = None
        self.last_temperature_c = None
        self.last_wear_cost = 0.0  # kept for backward compatibility; managed by BMS.
        self.last_dynamic_efficiency = None
        self._load_tables()
        self.soh_ah_interpolator = None
        self.soh_ah_thresholds = None
        self.soh_ah_values = None
        if self.chemistry == "NMC" and not self.disable_soh_degradation:
            self._load_soh_curve()
        self.cumulative_ah_throughput = 0.0
        self.last_soh = float(self.soh)

    def reset(self, current_step=0, soc=None, soh=1.0):
        # Reset internal state for a new episode.
        soc_value = 0.0 if soc is None else float(soc)
        soh_value = 1.0 if self.disable_soh_degradation else float(soh)
        self.soc = soc_value
        self.soe = soc_value
        self._soe = None
        self.soh = soh_value
        self.v_prev = None
        self._last_voltage = None
        self.current_a = 0.0
        self.voc_v = None
        self.voltage_v = None
        self.last_r0_ohm = None
        self.last_temperature_c = None
        self.last_wear_cost = 0.0
        self.last_dynamic_efficiency = None
        self.dyn_eta = None
        self.cumulative_ah_throughput = 0.0
        self.last_soh = soh_value
        self._pack_energy_mismatch_logged = False

    def _debug_internal_energy_change(self,
                                       *,
                                       context: str,
                                       current_step: int,
                                       external_energy_change: float,
                                       temperature_c: float,
                                       delta_t: float,
                                       voc: float,
                                       R0: float,
                                       v_prev: float,
                                       v_batt: float,
                                       power_kw: float,
                                       current_a: float,
                                       soc_previous: float,
                                       soc_unbounded: float,
                                       soc_new: float,
                                       dyn_eta: float,
                                       soe_previous: float,
                                       soe_new: float,
                                       internal_energy_change: float):
        if not self.debug_energy:
            return

        log_entry = {
            "context": context,
            "current_step": current_step,
            "temperature_c": temperature_c,
            "delta_t_hours": delta_t,
            "voc_v": voc,
            "R0_ohm": R0,
            "v_prev_v": v_prev,
            "v_batt_v": v_batt,
            "external_energy_change_kwh": external_energy_change,
            "power_kw": power_kw,
            "current_a": current_a,
            "soc_previous": soc_previous,
            "soc_unbounded": soc_unbounded,
            "soc_new": soc_new,
            "dyn_eta": dyn_eta,
            "soe_previous": soe_previous,
            "soe_new": soe_new,
            "internal_energy_change_kwh": internal_energy_change,
        }

        file_exists = self.debug_log_path.exists()

        with self.debug_log_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

    def _debug_block(self, title: str, items, color: str = "cyan"):
        if not self.debug_energy:
            return
        colors = {
            "red": "\x1b[31m",
            "yellow": "\x1b[33m",
            "cyan": "\x1b[36m",
            "magenta": "\x1b[35m",
            "reset": "\x1b[0m",
        }
        prefix = colors.get(color, "")
        reset = colors["reset"]
        lines = [f"{prefix}{title}{reset}"]
        for key, value in items:
            lines.append(f"  {key}: {value}")
        print("\n" + "\n".join(lines) + "\n")

    def _load_tables(self):
        """Load battery parameters from .mat file.
        
        File structure depends on chemistry:
        - LFP/NCA: 21 rows x 6 columns (SOH=1.0 only)
        - NMC: 105 rows x 6 columns (5 blocks of 21 rows each for SOH=[1.0, 0.863, 0.835, 0.82, 0.799])
        
        Each row contains: [SOC, R0@20C, R0@40C, SOC_dup, Voc@20C, Voc@40C]
        """
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        data_path = os.path.join(base_dir, self.parameters_mat)
        parameters = sio.loadmat(data_path)[os.path.splitext(self.parameters_mat)[0]]
        
        num_rows = len(parameters)
        num_soc_points = 21  # Standard SOC grid size (0-1 at 21 points)
        
        # Determine SOH grid based on chemistry and file size
        if self.chemistry in ('LFP', 'NCA'):
            # LFP/NCA: only SOH=1.0 (21 rows total)
            if num_rows != num_soc_points:
                raise ValueError(f"{self.chemistry} file should have exactly 21 rows, got {num_rows}")
            self.soh_grid = np.array([1.0])
            num_soh_points = 1
        elif self.chemistry == 'NMC':
            # NMC: 5 SOH points (105 rows = 5*21)
            if num_rows != 105:
                raise ValueError(f"NMC file should have exactly 105 rows (5x21), got {num_rows}")
            self.soh_grid = np.array([1.0, 0.863, 0.835, 0.82, 0.799])
            num_soh_points = 5
        else:
            raise ValueError(f"Unknown chemistry: {self.chemistry}")
        
        # Extract SOC grid from first block (rows 0:21)
        soc_from_file = parameters[:num_soc_points, 0]
        self.soc_grid = soc_from_file
        self.temperature_grid = np.array([20, 40])
        
        # Build 3D data arrays: (num_soc, num_temp, num_soh)
        voc_data_3d = np.zeros((num_soc_points, 2, num_soh_points))
        r0_data_3d = np.zeros((num_soc_points, 2, num_soh_points))
        
        # Fill in data from each SOH block
        for soh_idx in range(num_soh_points):
            row_start = soh_idx * num_soc_points
            row_end = row_start + num_soc_points
            
            voc_data_3d[:, :, soh_idx] = parameters[row_start:row_end, 4:6] * self.ns_batt
            r0_data_3d[:, :, soh_idx] = parameters[row_start:row_end, 1:3]
        
        # Scale R0 based on cell configuration
        scaling = (self.ns_batt / self.np_batt) * (self.reference_cell_capacity_ah / self.c_n)
        r0_data_3d *= scaling
        
        # Create interpolators (handles 1D or 3D SOH depending on chemistry)
        if num_soh_points == 1:
            # For LFP/NCA: squeeze out SOH dimension for 2D interpolation
            self.voc_interpolator = RegularGridInterpolator(
                (self.soc_grid, self.temperature_grid, self.soh_grid),
                voc_data_3d
            )
            self.r0_interpolator = RegularGridInterpolator(
                (self.soc_grid, self.temperature_grid, self.soh_grid),
                r0_data_3d
            )
        else:
            # For NMC: full 3D interpolation
            self.voc_interpolator = RegularGridInterpolator(
                (self.soc_grid, self.temperature_grid, self.soh_grid),
                voc_data_3d
            )
            self.r0_interpolator = RegularGridInterpolator(
                (self.soc_grid, self.temperature_grid, self.soh_grid),
                r0_data_3d
            )

    def _interp_voc_r0(self, soc: float, temperature_c: float, soh: float = None) -> Tuple[float, float]:
        """Interpolate Voc and R0 using trilinear interpolation (SOC, Temperature, SOH).

        Parameters
        ----------
        soc : float
            State of charge (0-1).
        temperature_c : float
            Temperature in Celsius.
        soh : float, optional
            State of health to use for interpolation. If None, uses ``self.soh``.
        """
        soc_clipped = float(np.clip(soc, self.soc_grid[0], self.soc_grid[-1]))
        temp_clipped = float(np.clip(temperature_c, self.temperature_grid[0], self.temperature_grid[-1]))
        soh_to_use = self.soh if soh is None else soh
        soh_clipped = float(np.clip(soh_to_use, self.soh_grid[0], self.soh_grid[-1]))
        
        voc = float(self.voc_interpolator((soc_clipped, temp_clipped, soh_clipped)))
        r0 = float(self.r0_interpolator((soc_clipped, temp_clipped, soh_clipped)))
        return voc, r0

    def _load_soh_curve(self):
        try:
            import pandas as pd
            from scipy.interpolate import interp1d
        except ImportError:
            return

        base_dir = os.path.join(os.path.dirname(__file__), "data")
        excel_path = os.path.join(base_dir, "NMC-SOHAh.xlsx")
        if not os.path.exists(excel_path):
            return

        df = pd.read_excel(excel_path)
        self.soh_ah_thresholds = df.iloc[:, 0].values
        self.soh_ah_values = df.iloc[:, 1].values / 100.0
        self.soh_ah_interpolator = interp1d(
            self.soh_ah_thresholds,
            self.soh_ah_values,
            kind="linear",
            bounds_error=False,
            fill_value=(self.soh_ah_values[0], self.soh_ah_values[-1]),
        )

    def _update_soh_from_ah(self, delta_ah: float) -> float:
        if self.disable_soh_degradation:
            self.last_soh = 1.0
            return 1.0

        if self.chemistry != "NMC" or self.soh_ah_interpolator is None:
            return self.soh

        self.cumulative_ah_throughput += abs(delta_ah)
        if self.cumulative_ah_throughput <= 0:
            soh_updated = 1.0
        elif self.cumulative_ah_throughput < self.soh_ah_thresholds[0]:
            ah_ratio = self.cumulative_ah_throughput / self.soh_ah_thresholds[0]
            soh_updated = 1.0 + ah_ratio * (self.soh_ah_values[0] - 1.0)
        else:
            soh_updated = float(self.soh_ah_interpolator(self.cumulative_ah_throughput))

        soh_updated = min(soh_updated, self.last_soh)
        self.last_soh = soh_updated
        return soh_updated

    def _dynamic_efficiency(self, current_a: float, voc: float, R0: float, v_batt: float) -> float:
        if np.isclose(current_a, 0.0):
            return 1.0 * self.eta_inverter

        if current_a > 0:
            base = self.eta_inverter * (1 - (R0 * current_a ** 2) / (current_a * max(voc, 1e-9)))
        else:
            base = self.eta_inverter * (1 - (R0 * current_a ** 2) / (-current_a * max(v_batt, 1e-9)))

        return max(0.0, min(1.0, base))

    def transition(self,
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
                   state_update: bool = False,
                   preview_transition: bool = False,
                   limit_transition: bool = False):
        if preview_transition or limit_transition:
            raise RuntimeError(
                "CustomChemistryTransitionModel supports only real state updates. "
                "Use BatteryBMSManager for preview/limit transitions."
            )

        if not state_update:
            raise RuntimeError(
                "CustomChemistryTransitionModel requires state_update=True. "
                "Non-state transitions are handled by BatteryBMSManager."
            )

        return self.transition_with_state(
            external_energy_change,
            min_capacity,
            max_capacity,
            min_soc,
            max_soc,
            efficiency,
            current_step,
            state_dict,
        )
        
    def transition_with_state(self, 
                          external_energy_change,
                          min_capacity,
                          max_capacity,
                          min_soc,
                          max_soc,
                          efficiency,
                          current_step,
                          state_dict):
        temperature_c = float(state_dict.get("temperature_c", self.temperature_c))
        delta_t = max(self.delta_t_hours, 1e-9)
        soh_for_step = float(state_dict.get("bms_soh", state_dict.get("soh", self.soh)))
        if self.disable_soh_degradation:
            soh_for_step = 1.0
        self.soh = soh_for_step

        if current_step == 0 or self.soc is None or self.soe is None:
            soc = float(state_dict.get("soc", 0.0))
            soe = float(state_dict.get("soc", 0.0))
            voc, R0 = self._interp_voc_r0(soc, temperature_c, soh_for_step)
            v_prev = voc
        else:
            soc = float(self.soc)
            soe = float(self.soe)
            voc, R0 = self._interp_voc_r0(soc, temperature_c, soh_for_step)
            v_prev = self.v_prev if self.v_prev is not None else voc

        power_kw = -external_energy_change / delta_t
        current_a = 1000.0 * power_kw / max(v_prev, 1e-9)
        v_batt = max(voc - R0 * current_a, 1e-6)

        battery_pack_charge = max(soh_for_step * self.c_n * self.np_batt, 1e-9)
        delta_ah = current_a * delta_t
        delta_ah_per_cell = delta_ah / max(self.np_batt, 1e-9)
        if self.disable_soh_degradation:
            self.soh = 1.0
            self.last_soh = 1.0
        else:
            self.soh = float(self._update_soh_from_ah(delta_ah_per_cell))

        soc_unbounded = soc - (current_a * delta_t) / battery_pack_charge
        soc_new = float(soc_unbounded)

        if current_step == 0:
            dyn_eta = efficiency
        else:
            dyn_eta = max(1e-9, self._dynamic_efficiency(current_a, voc, R0, v_batt))

        soe_unbounded = soe - (current_a * voc * delta_t / 1000) / max(max_capacity, 1e-9)
        soe_new = float(soe_unbounded)
        internal_energy_change = (soe_new - soe) * max_capacity

        self.soc = soc_new
        self.soe = soe_new
        self.v_prev = v_batt
        self.current_a = float(current_a)
        self.voc_v = float(voc)
        self.voltage_v = float(v_batt)
        self.last_r0_ohm = float(R0)
        self.last_temperature_c = float(temperature_c)
        self.dyn_eta = float(dyn_eta)
        self.last_dynamic_efficiency = float(dyn_eta)
        self.last_wear_cost = 0.0

        return internal_energy_change
