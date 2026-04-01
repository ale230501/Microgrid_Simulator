

import numpy as np


class Rule_Based_EMS:
    def __init__(self, microgrid):

        self.microgrid = microgrid

    @staticmethod
    def _compute_battery_limits(battery):
        soc_min = float(np.clip(getattr(battery, "min_soc", 0.0), 0.0, 1.0))
        soc_max = float(np.clip(getattr(battery, "max_soc", 1.0), soc_min, 1.0))

        max_capacity = float(getattr(battery, "max_capacity", 0.0))
        current_charge = float(getattr(battery, "current_charge", 0.0))
        e_min = soc_min * max_capacity
        e_max = soc_max * max_capacity

        discharge_internal = min(float(getattr(battery, "max_discharge", 0.0)), max(0.0, current_charge - e_min))
        charge_internal = min(float(getattr(battery, "max_charge", 0.0)), max(0.0, e_max - current_charge))

        if hasattr(battery, "model_transition"):
            discharge_soc_limit = max(0.0, float(battery.model_transition(discharge_internal)))
            charge_soc_limit = max(0.0, float(-battery.model_transition(-charge_internal)))
        else:
            discharge_soc_limit = max(0.0, discharge_internal)
            charge_soc_limit = max(0.0, charge_internal)

        max_discharge = max(
            0.0,
            min(float(getattr(battery, "max_production", discharge_soc_limit)), discharge_soc_limit),
        )
        max_charge = max(
            0.0,
            min(float(getattr(battery, "max_consumption", charge_soc_limit)), charge_soc_limit),
        )
        return max_discharge, max_charge

    def control(self, load_kwh, pv_kwh, band=None):
        """Controllo greedy che decide quanta energia usare da batteria e rete nello step corrente."""
        battery = self.microgrid.battery[0]
        e_grid = 0.0
        e_batt = 0.0

        tolerance = 0   # 1e-6   # Evita oscillazioni dovute alle approssimazioni floating point.
        max_discharge, max_charge = self._compute_battery_limits(battery)

        if load_kwh > pv_kwh + tolerance:
            # Carico maggiore della produzione FV: scarica la batteria finché possibile e importa il resto.
            deficit = load_kwh - pv_kwh
            discharge = min(deficit, max_discharge)     # Limita la scarica della batteria al massimo consentito
            e_batt = discharge
            e_grid = max(deficit - discharge, 0.0)      # Importa il resto dalla rete, se necessario
        elif pv_kwh > load_kwh + tolerance:
            # Surplus FV: carica la batteria entro i limiti e riversa l'eccesso verso la rete.
            surplus = pv_kwh - load_kwh
            charge = min(surplus, max_charge)          # Limita la carica della batteria al massimo consentito
            e_batt = -charge
            e_grid = -max(surplus - charge, 0.0)       # Esporta il resto alla rete, se necessario

        return e_batt, e_grid
