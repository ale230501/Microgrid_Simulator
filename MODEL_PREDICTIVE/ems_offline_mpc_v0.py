import os
import shutil
import sys
import time
import argparse
import warnings
from collections import deque
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import cvxpy as cp
from pandas import MultiIndex
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SIMULATOR_ROOT = PROJECT_ROOT / "SIMULATOR"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from SIMULATOR.microgrid_simulator import MicrogridSimulator
from SIMULATOR.tools import (
    load_config,
    load_pymgrid_scenario_bundle,
    compute_offline_tariff_vectors,
    plot_results,
    add_module_columns,
    print_final_report,
)

from MODEL_PREDICTIVE.mpc_MILP import MPCController


from pandasgui import show

def _resolve_required_dataset_path(path_str: str, arg_name: str) -> Path:
    if not path_str:
        raise ValueError(f"{arg_name} is required")
    dataset_path = Path(path_str)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{arg_name} not found: {dataset_path}")
    return dataset_path


def _prepare_df_for_pandasgui(df: pd.DataFrame) -> pd.DataFrame:
    """Return a GUI-friendly DataFrame with flat string columns."""
    gui_df = df.copy()
    if isinstance(gui_df.columns, pd.MultiIndex):
        gui_df.columns = [
            "__".join(str(part) for part in col if part not in (None, ""))
            for col in gui_df.columns
        ]
    else:
        gui_df.columns = [str(col) for col in gui_df.columns]
    return gui_df.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Offline Model Predictive Control (MPC) simulation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "controllers" / "mpc" / "params_OPSD.yml"),
        help="Path to YAML config file (e.g. configs/controllers/mpc/params_OPSD.yml).",
    )
    parser.add_argument(
        "--MPC_inference_dataset_path",
        default=None,
        help=(
            "Path al dataset CSV per inference MPC. "
            "Se omesso usa ems.inference_dataset_path o scenario.inference_dataset_path dal config."
        ),
    )
    parser.add_argument(
        "--MPC_forecast_dataset_path",
        default=None,
        help=(
            "Path al dataset CSV per forecast MPC. "
            "Se omesso usa ems.forecast_dataset_path o scenario.forecast_dataset_path dal config "
            "(fallback finale al dataset di inference)."
        ),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start timestep index in the CSV (inclusive).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="End timestep index in the CSV (exclusive). If omitted, uses start_step + steps.",
    )
    args = parser.parse_args()

    #os.environ.setdefault("PYMGRID_BATTERY_DEBUG", "1")
    os.environ.setdefault("PYMGRID_BATTERY_LOG", str(PROJECT_ROOT / "outputs" / "battery_debug.log"))
    _base_dir = Path(__file__).resolve().parents[1]
    _log_path = _base_dir / "outputs" / "battery_debug.log"
    os.environ["PYMGRID_BATTERY_LOG"] = str(_log_path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _log_path.touch(exist_ok=True)
    except OSError:
        pass

    CONTROL_STRATEGY_TAG = "MPC"
    BASE_OUTPUT_DIR = Path("outputs") / CONTROL_STRATEGY_TAG
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_OUTPUT_DIR / f"MPC_data_{run_timestamp}"
    if run_dir.exists():
        suffix = 1
        while (BASE_OUTPUT_DIR / f"MPC_data_{run_timestamp}_{suffix}").exists():
            suffix += 1
        run_dir = BASE_OUTPUT_DIR / f"MPC_data_{run_timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)


    ###### LOAD CONFIGURATION FROM YAML 

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)   # Carica configurazione da params.yml
    try:
        shutil.copy2(config_path, run_dir / config_path.name)
    except OSError:
        pass

    price_config = config['price_bands']               # Configurazione fasce prezzi
    simulation_steps = config['steps']                 # Numero di step di simulazione da eseguire
    start_step = int(args.start_step)
    end_step_cfg = args.end_step

    timezone_str = config['timezone']   # Configura timezone per timestamp 

    ##########  TIME SERIES DATASET   ###############

    scenario_dataset_mode = str(config.get("scenario_dataset_mode") or "").strip().lower()
    use_pymgrid_bundle = (
        args.MPC_inference_dataset_path is None
        and scenario_dataset_mode == "pymgrid_bundle"
        and bool(config.get("scenario_load_dataset_path"))
        and bool(config.get("scenario_pv_dataset_path"))
        and bool(config.get("scenario_grid_dataset_path"))
    )

    battery_cmd = []
    grid_cmd = []
    forecast_log = []
    bounds_log = []

    if use_pymgrid_bundle:
        if end_step_cfg is None:
            end_step_candidate = start_step + int(simulation_steps)
        else:
            end_step_candidate = int(end_step_cfg)

        bundle = load_pymgrid_scenario_bundle(
            load_dataset_path=str(config.get("scenario_load_dataset_path")),
            pv_dataset_path=str(config.get("scenario_pv_dataset_path")),
            grid_dataset_path=str(config.get("scenario_grid_dataset_path")),
            start_step=start_step,
            end_step=end_step_candidate,
            start_datetime_utc=str(config.get("scenario_start_datetime_utc") or "2020-01-01T00:00:00Z"),
            base_dir=PROJECT_ROOT,
        )

        time_series = bundle["time_series"].copy().reset_index(drop=True)
        simulation_steps = int(len(time_series))
        end_step = start_step + simulation_steps
        print(
            f"[INFO] MPC scenario dataset (pymgrid bundle): "
            f"start_step={start_step} end_step={end_step} steps={simulation_steps}"
        )

        pv_time_series = pd.Series(bundle["pv_series"])
        load_time_series = pd.Series(bundle["load_series"])
        timestamps = time_series["datetime"]
        price_buy_time_series = np.asarray(bundle["price_buy"], dtype=float)
        price_sell_time_series = np.asarray(bundle["price_sell"], dtype=float)
        grid_time_series = np.asarray(bundle["grid_series"], dtype=float)

        # The MPC core still consumes a single forecast CSV path.
        # If not explicitly provided, synthesize one from the scenario bundle.
        forecast_dataset_path_value = (
            args.MPC_forecast_dataset_path
            or config.get("forecast_dataset_path")
        )
        if forecast_dataset_path_value:
            forecast_dataset_path = _resolve_required_dataset_path(
                forecast_dataset_path_value,
                "MPC_forecast_dataset_path",
            )
        else:
            forecast_dataset_path = run_dir / "mpc_forecast_bundle.csv"
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(timestamps, utc=True, errors="coerce")
                    .dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "solar": np.asarray(pv_time_series, dtype=float),
                    "load": np.asarray(load_time_series, dtype=float),
                    "price_buy": price_buy_time_series,
                    "price_sell": price_sell_time_series,
                }
            ).to_csv(forecast_dataset_path, index=False)
    else:
        inference_dataset_path_value = (
            args.MPC_inference_dataset_path
            or config.get("inference_dataset_path")
        )
        if not inference_dataset_path_value:
            raise ValueError(
                "Missing MPC inference dataset path: pass --MPC_inference_dataset_path or set "
                "ems.inference_dataset_path/scenario.inference_dataset_path in config."
            )

        forecast_dataset_path_value = (
            args.MPC_forecast_dataset_path
            or config.get("forecast_dataset_path")
            or inference_dataset_path_value
        )

        inference_dataset_path = _resolve_required_dataset_path(
            inference_dataset_path_value,
            "MPC_inference_dataset_path",
        )
        forecast_dataset_path = _resolve_required_dataset_path(
            forecast_dataset_path_value,
            "MPC_forecast_dataset_path",
        )

        time_series = pd.read_csv(inference_dataset_path)   # Carica il dataset CSV con i dati di carico e PV
        time_series = time_series[["datetime","solar", "load"]].interpolate()   # interpolate() for removing NaNs
        time_series["datetime"] = pd.to_datetime(time_series["datetime"], utc=True, errors="coerce")

        dataset_len = int(len(time_series))
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        if start_step >= dataset_len:
            raise ValueError(f"start_step {start_step} is beyond dataset length {dataset_len}")

        if end_step_cfg is None:
            end_step = start_step + int(simulation_steps)
        else:
            end_step = int(end_step_cfg)

        if end_step <= start_step:
            raise ValueError(f"end_step must be > start_step, got start_step={start_step} end_step={end_step}")
        if end_step > dataset_len:
            print(f"[WARN] end_step {end_step} exceeds dataset length {dataset_len}. Using end_step={dataset_len}.")
            end_step = dataset_len

        time_series = time_series.iloc[start_step:end_step].reset_index(drop=True)
        simulation_steps = int(len(time_series))
        print(f"[INFO] MPC offline window: start_step={start_step} end_step={end_step} steps={simulation_steps}")

        pv_time_series = time_series['solar']
        pv_time_series = pv_time_series.clip(lower=0)

        load_time_series = time_series['load']

        timestamps = time_series['datetime']

        price_buy_time_series, price_sell_time_series = compute_offline_tariff_vectors(
            ts_series=timestamps,
            local_timezone=timezone_str,
            price_config=price_config
        )
        price_buy_time_series = price_buy_time_series
        price_sell_time_series = price_sell_time_series

        # time series containing values for cost of carbon dioxide emissions (not accounted for this task, so put to zero)
        emissions_time_series = np.zeros(len(price_buy_time_series))

        grid_time_series = np.stack([price_buy_time_series, price_sell_time_series, emissions_time_series], axis=1)


    ############################################


    ###### INSTANTIATE MICROGRID SIMULATOR WITH MODULES


    print("Inizializzazione microgrid...")

    simulator = MicrogridSimulator(
        config_path=str(config_path),
        online=False,
        load_time_series = load_time_series, 
        pv_time_series = pv_time_series, 
        grid_time_series = grid_time_series
    )

    microgrid = simulator.build_microgrid()  # Costruisce la microgrid con i moduli specificati nel file di configurazione 

    load_module = microgrid.modules['load'][0]         # Modulo load
    pv_module = microgrid.modules['pv'][0]             # Modulo PV
    battery_module = microgrid.battery[0]              # Modulo batteria

    microgrid.reset()  # Porta la microgrid in uno stato noto prima di iniziare la simulazione.

    ###### INSTANTIATE MODEL PREDICTIVE CONTROL AND RUN SIMULATION
    mpc = MPCController(
        microgrid,
        config_path=config_path,
        data_path=forecast_dataset_path,
        start_step=start_step,
        end_step=end_step,
        log_path=run_dir / "mpc_milp_log.csv",
    )
    print(mpc._solver)                          

    for step in tqdm(
        range(1, simulation_steps + 1),
        desc="Loading...",
        unit="step",
        ascii=True,
    ):         # Loop principale per il numero di step specificato

        load_kwh = load_module.current_load
        pv_kwh = pv_module.current_renewable

        """load_forecast, pv_forecast, grid_status, import_price, export_price, \
        e_max, e_min, p_max_charge, p_max_discharge, grid_max_import, grid_max_export, \
        soc_0, genset_max, cost_co2, grid_co2, genset_co2 = mpc._get_modular_state_values()

        forecast_log.append({
            "step": step,
            "timestamp": timestamps.iloc[step - 1],
            "load_forecast": load_forecast.tolist(),
            "pv_forecast": pv_forecast.tolist(),
            "import_price_forecast": import_price.tolist(),
            "export_price_forecast": export_price.tolist(),
            "soc_0": soc_0,
            # aggiungi altri campi se ti servono
        })
        """

        control = mpc.get_action(verbose=0)   # verbose=2 serve per vedere i dettagli di calcolo all'interno dell'MPC

        obs, reward, done, info = microgrid.step(control, normalized=False)


    microgrid_df, log = simulator.get_simulation_log(microgrid)     # Ottiene il log della simulazione come DataFrame pandas con tutti gli step

    log.to_csv(run_dir / "microgrid_log.csv", index=True)            # Salva il log della simulazione su file CSV


    # forecast_df = pd.DataFrame(forecast_log)
    # forecast_df.to_csv(OUTPUT_DIR / "mpc_forecast_log.csv", index=False)
    # forecast_df.to_csv(
    #     OUTPUT_DIR / "mpc_forecast_log_excel.csv",
    #     index=False,
    #     sep=";",
    #     decimal=",",
    # )

    bms_manager = getattr(battery_module, "battery_bms_manager", None)
    transition_model = battery_module.battery_transition_model      # Modello fisico (usato solo come fallback)

    if bms_manager is not None and hasattr(bms_manager, "get_transition_history"):
        history = bms_manager.get_transition_history()              # Cronologia transizioni dal BMS
    elif transition_model is not None and hasattr(transition_model, "get_transition_history"):
        history = transition_model.get_transition_history()         # Fallback legacy
    else:
        history = []
    eta = [entry.get('efficiency', np.nan) for entry in history]    # Estrae l'efficienza dinamica dalla cronologia
    soh = [entry.get('soh', np.nan) for entry in history]           # Estrae lo stato di salute (SOH) dalla cronologia

    voc_lut = np.asarray([entry.get("voc_v", np.nan) for entry in history], dtype=float)
    if np.isfinite(voc_lut).any():
        n = min(len(voc_lut), len(time_series))
        plt.figure(figsize=(12, 4))
        plt.plot(time_series["datetime"].to_numpy()[:n], voc_lut[:n], label="Voc (lookup table)")
        plt.xlabel("Time")
        plt.ylabel("Voc [V]")
        plt.title("Open-circuit voltage (Voc) from lookup table over time")
        plt.grid(True)
        plt.tight_layout()
        voc_plot_path = run_dir / f"voc_lut_{simulator.battery_chemistry}.png"
        plt.savefig(voc_plot_path, bbox_inches="tight")
        plt.show()

    additional_columns = {
        ('datetime', 0, 'timestamp'): time_series['datetime'].to_numpy()[:len(microgrid_df)],
        ('pv', 0, 'pv_prod_input'): pv_time_series.to_numpy()[:len(microgrid_df)],                   # Aggiunge la produzione PV gi?? clippata al DataFrame della microgrid
        ('load', 0, 'consumption_input'): time_series['load'].to_numpy()[:len(microgrid_df)],        # Aggiunge il consumo in input come colonna al DataFrame della microgrid
        ('battery', 0, 'eta'): np.asarray(eta),                                                      # Aggiunge l'efficienza dinamica al DataFrame della microgrid
        ('battery', 0, 'soh'): np.asarray(soh),                                                        # Aggiunge lo stato di salute (SOH) al DataFrame della microgrid
        ('price', 0, 'price_buy'): price_buy_time_series[: len(microgrid_df)],                       # Aggiunge la colonna price_buy al DataFrame della microgrid
        ('price', 0, 'price_sell'): price_sell_time_series[: len(microgrid_df)]                      # Aggiunge la colonna price_sell al DataFrame della microgrid
    }
        
    microgrid_df = add_module_columns(microgrid_df, additional_columns)

    print(transition_model)

    summary_buffer = StringIO()
    with redirect_stdout(summary_buffer):
        print_final_report(
            microgrid_df,
            control_strategy=CONTROL_STRATEGY_TAG,
            battery_chemistry=simulator.battery_chemistry,
            soh_degradation_enabled=not bool(getattr(simulator, "disable_soh_degradation", False)),
        )
    summary_text = summary_buffer.getvalue()
    print(summary_text, end="")
    (run_dir / "final_report.txt").write_text(summary_text, encoding="utf-8")

    history_owner = bms_manager if bms_manager is not None else transition_model
    if history_owner is not None and hasattr(history_owner, "plot_transition_history"):
        history_owner.plot_transition_history(
            save_path=str(PROJECT_ROOT / "battery_simulation_data" / "mpc" / f"transitions_{simulator.battery_chemistry}.png"),
            show=True,
        )
    if history_owner is not None and hasattr(history_owner, "save_transition_history"):
        history_owner.save_transition_history(
            history_path=str(PROJECT_ROOT / "battery_simulation_data" / "mpc" / f"transitions_{simulator.battery_chemistry}.json"),
        )


    print("Number of battery overshoots:", int(getattr(battery_module, "num_overshoots", 0)))

    #print(microgrid.log.columns)

    gui_data = {
        "time_series": time_series.reset_index(drop=True),
        "microgrid_df": _prepare_df_for_pandasgui(microgrid_df),
    }
    if forecast_log:
        gui_data["forecast_log"] = pd.DataFrame(forecast_log)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Series.__getitem__ treating keys as positions is deprecated.*",
            category=FutureWarning,
        )
        show(**gui_data)



    ###### PLOT FINAL RESULTS #######

    csv_name = f"{CONTROL_STRATEGY_TAG}_ems_results_{run_timestamp}.csv"      
    csv_path = run_dir / csv_name                                                  # Directory di output per file CSV e grafici
    battery_chemistry = str(simulator.battery_chemistry)
    safe_chemistry = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-"
        for ch in battery_chemistry.lower()
    ).strip("-_")
    if not safe_chemistry:
        safe_chemistry = "battery"
    base_name = (run_dir / f"{csv_name.replace('.csv', '')}_{safe_chemistry}")

    plot_paths = plot_results(microgrid_df, str(base_name), timezone_str)       # Genera e salva i grafici, ottenendo i percorsi dei file

    print("\nApertura grafici...")
    for label, path in plot_paths.items():              # Tenta di aprire automaticamente i file grafici generati
        try:
            os.startfile(os.path.abspath(path))
        except OSError:
            print(f"  Impossibile aprire automaticamente {path}")


if __name__ == "__main__":
    main()
