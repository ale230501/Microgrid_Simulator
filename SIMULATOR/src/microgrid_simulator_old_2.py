import numpy as np
import pandas as pd
import pytest

from src.pymgrid import Microgrid
from src.pymgrid.modules import GridModule, LoadModule, RenewableModule, BatteryModule

from pandasgui import show
from src.pymgrid.algos import RuleBasedControl


###############  SIMULATION PARAMETERS ##############


######  BATTERY

soc_min = 0.2
soc_max = 0.8
battery_capacity = 50.2       # [kWh] 
min_capacity = soc_min * battery_capacity # [kWh]
max_capacity = soc_max * battery_capacity # [kWh]
storage_power_max = 70  # [kW]
sample_time = 0.25  # [h]
max_charge_per_step = storage_power_max * 1 * sample_time  # [kWh] <--> maximum_power (in kW) * 1 h * sample_time 
max_discharge_per_step = max_charge_per_step # [kWh]
battery_efficiency = 0.9
init_soc = 0.6 

######

######  GRID

max_grid_export_power = 8 # [kW]
max_grid_import_power = 8 # [kW]

max_grid_export_per_step = max_grid_export_power * sample_time  # defined in [kWh] for each timestep  
max_grid_import_per_step = max_grid_import_power * sample_time  # defined in [kWh] for each timestep  

######  


############################################




battery = BatteryModule(      
                              min_capacity = min_capacity, # [kWh]
                              max_capacity = max_capacity, # [kWh]
                              max_charge = max_charge_per_step, # [kWh]
                              max_discharge = max_discharge_per_step, # [kWh]
                              efficiency = battery_efficiency,
                              init_soc = init_soc      
                                                        )


load_module = LoadModule(
        time_series=None,
        online=True,
        initial_time_series_value=0.0,
    )

pv_module = RenewableModule(
    time_series=None,
    online=True,
    initial_time_series_value=0.0,
)

present_grid_prices = np.array([0.15, 0.05, 0.0, 1.0])

grid_module = GridModule(
                            max_import = max_grid_import_power,
                            max_export = max_grid_export_power, 
                            time_series=None,
                            online=True,
                            initial_time_series_value=present_grid_prices,
                            normalized_action_bounds=( -max_grid_import_power, max_grid_import_power ))


def test_online_real_time_simulation():
    #rng = np.random.default_rng(42)

    steps = 5

    

    

    microgrid = Microgrid(  modules=[
                                    battery,
                                    ('load', load_module), 
                                    ('pv', pv_module), 
                                    grid_module
                                                ]     )
    

    microgrid.reset()

    records = []
    for step in range(steps):
        #load_value = rng.uniform(0.0, 3.0)
        #pv_value = rng.uniform(0.0, 3.0)

        load_value = 2.3
        pv_value = 2

        records.append({'load_consumption': load_value, 'pv_production': pv_value})
        
        data = pd.DataFrame(records)

        
        microgrid.ingest_real_time_data({'load': load_value, 'pv': pv_value, 'grid': [present_grid_prices]})

        measurements = microgrid.fetch_real_time_data(['load', 'pv', 'grid'])

        e_grid = 0

        e_batt = 1

        if pv_value + e_grid <= abs(e_batt):

            e_batt = abs(e_batt) - pv_value + e_grid

        else: e_batt = - (pv_value + e_grid - abs(e_batt))

        control = {"battery" : [e_batt] ,
                   "grid": [e_grid] 
           }

        observations, reward, done, info = microgrid.step(control, normalized = False)


    log = microgrid.log.copy()
    log.columns = ['{}_{}_{}'.format(*col) for col in log.columns]

    microgrid_df = microgrid.log[
        [
            ('load', 0, 'load_met'),
            ('pv', 0, 'renewable_used'),
            ('pv', 0, 'curtailment'),
            ('balancing', 0, 'loss_load'),
            ('battery', 0, 'soc'),
            ('battery', 0, 'current_charge'),
            ('battery', 0, 'discharge_amount'),
            ('battery', 0, 'charge_amount'),
            ('grid', 0, 'grid_import'),
            ('grid', 0, 'grid_export')  # opzionale
        ]
    ]

    microgrid_df['load_consumption'] = data['load_consumption']
    microgrid_df['pv_production'] = data['pv_production']


    show(microgrid_df=microgrid_df)



#def test_rule_based_control():




test_online_real_time_simulation()

