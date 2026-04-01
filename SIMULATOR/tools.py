
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, MultipleLocator
import yaml
from typing import Dict, List, Optional, Tuple, Any
import pytz
import pandas as pd


def _resolve_local_path(path_value: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = base_dir or Path(__file__).resolve().parents[1]
    return root / path


def _load_numeric_array(path: Path) -> np.ndarray:
    df = pd.read_csv(path, compression="infer")
    if df.empty:
        raise ValueError(f"Dataset vuoto: {path}")

    first_col = str(df.columns[0]) if len(df.columns) else ""
    if first_col.startswith("Unnamed:") or first_col == "":
        df = df.iloc[:, 1:]

    if df.empty:
        raise ValueError(f"Dataset senza colonne utili: {path}")

    arr = df.to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_pymgrid_scenario_bundle(
    *,
    load_dataset_path: str,
    pv_dataset_path: str,
    grid_dataset_path: str,
    start_step: int = 0,
    end_step: Optional[int] = None,
    start_datetime_utc: str = "2020-01-01T00:00:00Z",
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    load_path = _resolve_local_path(load_dataset_path, base_dir=base_dir)
    pv_path = _resolve_local_path(pv_dataset_path, base_dir=base_dir)
    grid_path = _resolve_local_path(grid_dataset_path, base_dir=base_dir)

    for p in (load_path, pv_path, grid_path):
        if not p.is_file():
            raise FileNotFoundError(f"Dataset non trovato: {p}")

    load_arr = _load_numeric_array(load_path).reshape(-1)
    pv_arr = _load_numeric_array(pv_path).reshape(-1)
    grid_arr = _load_numeric_array(grid_path)

    total_len = min(load_arr.shape[0], pv_arr.shape[0], grid_arr.shape[0])
    if total_len <= 0:
        raise ValueError("Nessun dato disponibile nel bundle scenario pymgrid.")

    if start_step < 0:
        raise ValueError(f"start_step deve essere >= 0, ricevuto {start_step}")
    if start_step >= total_len:
        raise ValueError(f"start_step={start_step} oltre la lunghezza disponibile {total_len}")

    if end_step is None:
        end_step = total_len
    else:
        end_step = int(end_step)

    if end_step <= start_step:
        raise ValueError(f"end_step deve essere > start_step (start={start_step}, end={end_step})")
    if end_step > total_len:
        end_step = total_len

    load_slice = np.abs(load_arr[start_step:end_step])
    pv_slice = np.clip(pv_arr[start_step:end_step], 0.0, None)
    grid_slice = grid_arr[start_step:end_step]

    if grid_slice.shape[1] < 2:
        raise ValueError(f"Grid dataset deve avere almeno 2 colonne (buy/sell). Trovate: {grid_slice.shape[1]}")
    if grid_slice.shape[1] == 2:
        zeros = np.zeros((grid_slice.shape[0], 1), dtype=float)
        grid_slice = np.concatenate([grid_slice, zeros], axis=1)
    elif grid_slice.shape[1] > 4:
        grid_slice = grid_slice[:, :4]

    start_ts = pd.Timestamp(start_datetime_utc)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    timestamps = pd.date_range(start=start_ts, periods=end_step - start_step, freq="h", tz="UTC")

    time_series = pd.DataFrame(
        {
            "datetime": timestamps,
            "solar": pv_slice,
            "load": load_slice,
        }
    )

    price_buy = np.asarray(grid_slice[:, 0], dtype=float)
    price_sell = np.asarray(grid_slice[:, 1], dtype=float)

    return {
        "time_series": time_series,
        "pv_series": np.asarray(pv_slice, dtype=float),
        "load_series": np.asarray(load_slice, dtype=float),
        "timestamps": pd.Series(timestamps),
        "price_buy": price_buy,
        "price_sell": price_sell,
        "grid_series": np.asarray(grid_slice, dtype=float),
    }




def get_online_grid_prices(timestamp: datetime, price_config: dict):
    """Determina la fascia oraria del timestamp e restituisce il vettore prezzi associato."""
    hour = timestamp.hour  # Confrontiamo solo l'ora perché le fasce sono espresse in intervalli orari.

    def hour_in_ranges(hr, ranges):
        """True se l'ora `hr` rientra in uno dei range dichiarati per la fascia (peak o standard), altrimenti False."""
        for rng in ranges or []:
            start, end = rng
            if start <= hr <= end:
                return True
        return False

    # Ricerca prioritaria: prima le fasce più costose (peak) poi quelle standard.
    for band_key in ('peak', 'standard'):
        band_cfg = price_config.get(band_key)
        if band_cfg and hour_in_ranges(hour, band_cfg.get('ranges')):
            return (
                np.array([band_cfg.get('buy', 0.0), band_cfg.get('sell', 0.0), 0.0, 1.0]),
                band_key.upper(),
            )

    # In assenza di match utilizziamo la fascia di fallback (offpeak o la prima definita).
    fallback_key = 'offpeak' if 'offpeak' in price_config else next(iter(price_config))
    fallback_cfg = price_config.get(fallback_key, {})
    return (
        np.array([fallback_cfg.get('buy', 0.0), fallback_cfg.get('sell', 0.0), 0.0, 1.0]),
        fallback_key.upper(),
    )



def init_live_battery_display(initial_soc_pct, timestamp):
    """Crea la finestra matplotlib con l'icona della batteria aggiornata ad ogni step."""
    try:
        plt.ion()  # Modalità interattiva per aggiornare il disegno senza bloccare l'esecuzione.
        fig, ax = plt.subplots(figsize=(3, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.3)
        ax.axis('off')  # Nasconde gli assi per mostrare solo l'icona stilizzata.

        # Corpo e cappuccio della batteria (solo contorni).
        body = patches.Rectangle((0.2, 0.15), 0.6, 1.0, linewidth=2.5, edgecolor='black', facecolor='none', joinstyle='round')
        cap = patches.Rectangle((0.4, 1.15), 0.2, 0.08, linewidth=2.0, edgecolor='black', facecolor='lightgray')

        # Riempimento proporzionale allo stato di carica iniziale.
        fill = patches.Rectangle(
            (0.2, 0.15),
            0.6,
            max(0.0, min(1.0, initial_soc_pct / 100.0)) * 1.0,
            facecolor='#32CD32',
        )

        ax.add_patch(fill)
        ax.add_patch(body)
        ax.add_patch(cap)

        # Testi dinamici per SOC e timestamp corrente.
        soc_text = ax.text(0.5, 0.05, f"{initial_soc_pct:5.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        time_text = ax.text(0.5, 1.28, str(timestamp), ha='center', va='bottom', fontsize=10)

        fig.canvas.draw()
        fig.canvas.flush_events()
        return {'fig': fig, 'fill': fill, 'soc_text': soc_text, 'time_text': time_text}
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] impossibile inizializzare la batteria live: {exc}")
        return None

def update_live_battery_display(display, soc_pct, timestamp):
    """Aggiorna colore, altezza del riempimento e testi della batteria live."""
    if not display:
        return
    soc_norm = max(0.0, min(1.0, soc_pct / 100.0))
    display['fill'].set_height(soc_norm * 1.0)

    # Colori intuitivi in base allo stato di carica.
    if soc_pct <= 20:
        display['fill'].set_color('#CC2936')  # rosso
    elif soc_pct <= 40:
        display['fill'].set_color('#FFA500')  # arancione
    else:
        display['fill'].set_color('#32CD32')  # verde

    display['soc_text'].set_text(f"{soc_pct:5.1f}%")
    display['time_text'].set_text(str(timestamp))
    display['fig'].canvas.draw()
    display['fig'].canvas.flush_events()


def load_config(path=None):
    """Legge la sezione `ems` dal file YAML e valida i campi necessari all'esecuzione."""
    if path is None:
        base_dir = Path(__file__).resolve().parents[1]
        path = base_dir / "RULE_BASED" / "params_OPSD.yml"
    path = Path(path)
    with open(path, 'r') as cfg_file:
        full_config = yaml.safe_load(cfg_file)

    ems_cfg = full_config.get('ems')
    scenario_cfg = full_config.get('scenario') or {}
    if not ems_cfg:
        raise KeyError(f"Sezione 'ems' mancante in {path}")

    # Verifica presenza chiavi richieste per evitare errori silenziosi più avanti.
    required_keys = ('buffer_size', 'timezone', 'steps', 'price_bands')
    missing_keys = [key for key in required_keys if key not in ems_cfg]
    if missing_keys:
        raise KeyError(f"Mancano le chiavi {missing_keys} nella sezione 'ems' di {path}")

    try:
        buffer_size = int(ems_cfg['buffer_size'])
        steps = int(ems_cfg['steps'])
    except (TypeError, ValueError) as exc:
        raise ValueError("I campi 'buffer_size' e 'steps' devono essere interi.") from exc

    start_step = ems_cfg.get('start_step', 0)
    end_step = ems_cfg.get('end_step', None)
    try:
        start_step = int(start_step) if start_step is not None else 0
    except (TypeError, ValueError) as exc:
        raise ValueError("Il campo opzionale 'start_step' deve essere un intero.") from exc
    if end_step is not None:
        try:
            end_step = int(end_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("Il campo opzionale 'end_step' deve essere un intero o null.") from exc

    inference_dataset_path = (
        ems_cfg.get('inference_dataset_path')
        or scenario_cfg.get('inference_dataset_path')
        or scenario_cfg.get('dataset_path')
    )
    forecast_dataset_path = (
        ems_cfg.get('forecast_dataset_path')
        or scenario_cfg.get('forecast_dataset_path')
        or scenario_cfg.get('dataset_path')
    )

    return {
        'buffer_size': buffer_size,
        'timezone': ems_cfg['timezone'],
        'steps': steps,
        'start_step': start_step,
        'end_step': end_step,
        'price_bands': ems_cfg['price_bands'],
        'inference_dataset_path': inference_dataset_path,
        'forecast_dataset_path': forecast_dataset_path,
        'scenario_dataset_mode': scenario_cfg.get('dataset_mode'),
        'scenario_load_dataset_path': scenario_cfg.get('load_dataset_path'),
        'scenario_pv_dataset_path': scenario_cfg.get('pv_dataset_path'),
        'scenario_grid_dataset_path': scenario_cfg.get('grid_dataset_path'),
        'scenario_start_datetime_utc': scenario_cfg.get('start_datetime_utc', '2020-01-01T00:00:00Z'),
    }


def print_step_report(step_idx, timestamp, band, input_load, input_pv, load_kwh, pv_kwh,
                      battery_info, grid_info, energy_metrics, control, prices, economics):
    """
    Stampa un report leggibile per ogni step con dati input vs simulator, controllo applicato,
    stato batteria, scambi rete, prezzi ed economia, facilitando il debug live.
    """
    header = f"\n{'=' * 120}\nSTEP {step_idx} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({band})\n{'=' * 120}"    # Intestazione step
    print(header)

    print(f"Input Load/PV        : load={input_load:6.3f} kWh | pv={input_pv:6.3f} kWh")
    print(f"Microgrid Load/PV    : load={load_kwh:6.3f} kWh | pv={pv_kwh:6.3f} kWh")
    print(f"Controllo applicato  : battery={control['battery']:6.3f} kWh | grid={control['grid']:6.3f} kWh")

    print("\nEnergia Microgrid:")
    print(f"  Load met          : {energy_metrics['load_met']:6.3f} kWh")
    print(f"  Renewable used    : {energy_metrics['renewable_used']:6.3f} kWh")
    print(f"  Curtailment       : {energy_metrics['curtailment']:6.3f} kWh")
    print(f"  Loss of load      : {energy_metrics['loss_load']:6.3f} kWh")

    print("\nBatteria:")
    print(f"  SOC              : {battery_info['soc_pct']:6.2f}%")
    print(f"  Current charge   : {battery_info['current_charge']:6.3f} kWh")
    print(f"  Charge/Discharge : +{battery_info['charge_amount']:6.3f} kWh | -{battery_info['discharge_amount']:6.3f} kWh")

    print("\nRete:")
    print(f"  Import/Export    : +{grid_info['import']:6.3f} kWh | -{grid_info['export']:6.3f} kWh")
    print(f"  Prezzi           : buy={prices['buy']:5.2f} EUR/kWh | sell={prices['sell']:5.2f} EUR/kWh")

    print("\nEconomia:")
    print(f"  Cost Import      : {economics['cost']:7.4f} EUR")
    print(f"  Revenue Export   : {economics['revenue']:7.4f} EUR")
    print(f"  Balance          : {economics['balance']:7.4f} EUR")
    print(f"  Reward (approx)  : {economics['reward']:7.4f}")


def print_final_report(
    microgrid_df: pd.DataFrame,
    control_strategy: Optional[str] = None,
    battery_chemistry: Optional[str] = None,
    soh_degradation_enabled: Optional[bool] = None,
):
    """
    Stampa un resoconto finale riassumendo energia, batteria ed economia a partire dal microgrid_df finale.
    Se il DataFrame è vuoto o None viene emesso un avviso e la funzione termina silenziosamente.
    """
    if microgrid_df is None or microgrid_df.empty:
        print("[WARN] impossibile generare il resoconto finale: microgrid_df è vuoto.")
        return

    df = microgrid_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if x not in (None, ""))
            for col in df.columns
        ]

    def normalize_label(label):
        if label is None:
            return None
        if isinstance(label, tuple):
            return "_".join(str(x) for x in label if x not in (None, ""))
        return str(label)

    def get_series(*labels, numeric=True):
        for label in labels:
            normalized = normalize_label(label)
            if not normalized:
                continue
            if normalized in df.columns:
                series = df[normalized]
                return pd.to_numeric(series, errors="coerce") if numeric else series
        return None

    def sum_series(*labels):
        series = get_series(*labels)
        if series is None:
            return None
        return float(series.fillna(0.0).sum())

    def last_series_value(*labels):
        series = get_series(*labels)
        if series is None:
            return None
        valid = series.dropna()
        if valid.empty:
            return None
        return float(valid.iloc[-1])

    def min_series_value(*labels):
        series = get_series(*labels)
        if series is None:
            return None
        valid = series.dropna()
        if valid.empty:
            return None
        return float(valid.min())

    def max_series_value(*labels):
        series = get_series(*labels)
        if series is None:
            return None
        valid = series.dropna()
        if valid.empty:
            return None
        return float(valid.max())

    def sum_product(series_a, series_b):
        if series_a is None or series_b is None:
            return None
        return float((series_a.fillna(0.0) * series_b.fillna(0.0)).sum())

    def fmt_value(value, unit="", digits=3):
        if value is None:
            return "n.d."
        return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"

    def fmt_percent(value):
        if value is None:
            return "n.d."
        return f"{value * 100:.1f}%"

    steps = len(df)
    timestamp_series = get_series(('datetime', 0, 'timestamp'), numeric=False)
    if timestamp_series is not None:
        timestamps = pd.to_datetime(timestamp_series, errors="coerce").dropna()
        start_ts = timestamps.iloc[0] if not timestamps.empty else None
        end_ts = timestamps.iloc[-1] if not timestamps.empty else None
    else:
        start_ts = end_ts = None

    load_input = sum_series(('load', 0, 'consumption_input'))
    load_met = sum_series(('load', 0, 'load_met'))
    if load_input is None:
        load_input = load_met

    pv_input = sum_series(('pv', 0, 'pv_prod_input'))
    renewable_used = sum_series(('pv', 0, 'renewable_used'))
    if pv_input is None:
        pv_input = renewable_used

    curtailment = sum_series(('pv', 0, 'curtailment'))
    loss_load = sum_series(('balancing', 0, 'loss_load'))
    overgeneration = sum_series(('balancing', 0, 'overgeneration'))

    grid_import_total = sum_series(('grid', 0, 'grid_import'))
    grid_export_total = sum_series(('grid', 0, 'grid_export'))
    battery_charge_total = sum_series(('battery', 0, 'charge_amount'))
    battery_discharge_total = sum_series(('battery', 0, 'discharge_amount'))

    soc_final = last_series_value(('battery', 0, 'soc'))
    soc_min = min_series_value(('battery', 0, 'soc'))
    soc_max = max_series_value(('battery', 0, 'soc'))
    soh_final = last_series_value(('battery', 0, 'soh'))
    resid_charge = last_series_value(('battery', 0, 'current_charge'))

    grid_import_series = get_series(('grid', 0, 'grid_import'))
    grid_export_series = get_series(('grid', 0, 'grid_export'))
    price_buy_series = get_series(('price', 0, 'price_buy'), ('grid', 0, 'import_price_current'))
    price_sell_series = get_series(('price', 0, 'price_sell'), ('grid', 0, 'export_price_current'))

    cost_import = sum_product(grid_import_series, price_buy_series)
    revenue_export = sum_product(grid_export_series, price_sell_series)

    battery_reward_total = sum_series(('battery', 0, 'reward'))
    battery_wear_cost = -battery_reward_total if battery_reward_total is not None else None
    grid_reward_total = sum_series(('grid', 0, 'reward'))
    balance_reward_total = sum_series(('balance', 0, 'reward'))

    line = "=" * 110
    print(f"\n{line}\nRESOCONTO FINALE MICROGRID\n{line}")
    if control_strategy:
        print(f"Tipo simulazione     : {control_strategy}")
    if battery_chemistry:
        print(f"Chimica batteria     : {battery_chemistry}")
    if soh_degradation_enabled is not None:
        soh_mode = "ON" if bool(soh_degradation_enabled) else "OFF"
        print(f"Degradazione SoH     : {soh_mode}")
    if start_ts is not None and end_ts is not None:
        start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Periodo simulato      : {start_str}  ->  {end_str}")
    print(f"Step simulati         : {steps}")

    print("\nEnergia [kWh]")
    print(f"  Domanda totale (input)   : {fmt_value(load_input, 'kWh')}")
    print(f"  Domanda servita          : {fmt_value(load_met, 'kWh')}")
    print(f"  Produzione PV (input)    : {fmt_value(pv_input, 'kWh')}")
    print(f"  Rinnovabile consumata    : {fmt_value(renewable_used, 'kWh')}")
    print(f"  Import dalla rete        : {fmt_value(grid_import_total, 'kWh')}")
    print(f"  Export verso rete        : {fmt_value(grid_export_total, 'kWh')}")
    print(f"  Curtailment              : {fmt_value(curtailment, 'kWh')}")
    print(f"  Loss of load             : {fmt_value(loss_load, 'kWh')}")
    print(f"  Overgeneration           : {fmt_value(overgeneration, 'kWh')}")

    print("\nBatteria")
    print(f"  Energia caricata         : {fmt_value(battery_charge_total, 'kWh')}")
    print(f"  Energia scaricata        : {fmt_value(battery_discharge_total, 'kWh')}")
    print(f"  SOC finale / min / max   : {fmt_percent(soc_final)} | {fmt_percent(soc_min)} | {fmt_percent(soc_max)}")
    print(f"  SOH finale               : {fmt_percent(soh_final)}")
    print(f"  Carica residua           : {fmt_value(resid_charge, 'kWh')}")

    print("\nEconomia")
    print(f"  Costo energia import     : {fmt_value(cost_import, 'EUR', digits=2)}")
    print(f"  Ricavo export            : {fmt_value(revenue_export, 'EUR', digits=2)}")
    print(f"  Usura batteria (stimata) : {fmt_value(battery_wear_cost, 'EUR', digits=2)}")
    print(f"  Reward grid module       : {fmt_value(grid_reward_total, 'EUR', digits=2)}")
    print(f"  Reward complessivo       : {fmt_value(balance_reward_total, 'EUR', digits=2)}")
    print(line)


def plot_results(df: pd.DataFrame, base_name: str, timezone: Optional[str] = None):
    """
    Duplica il DataFrame finale, ordina per timestamp e genera cinque grafici (potenze, energia rete step+cumulata, 
    prezzi/bande TOU con fill_between, SOC e flussi batteria con doppio asse, metriche economiche con bilancio cumulativo).
    Ogni figura viene salvata a 160 dpi con nome basato su base_name, poi chiusa per liberare memoria; 
    la funzione restituisce i percorsi dei file generati per uso successivo.
    """
    df = df.copy()                                         # Duplica DataFrame per evitare modifiche all'originale

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if x not in (None, ""))
            for col in df.columns
        ]

    column_mapping = {
        "datetime_0_timestamp": "timestamp",
        "pv_0_pv_prod_input": "pv_prod",
        "load_0_consumption_input": "consumption",
        "price_0_price_buy": "price_buy",
        "price_0_price_sell": "price_sell",
        "grid_0_grid_import": "grid_import",
        "grid_0_grid_export": "grid_export",
        "battery_0_soc": "soe",
        "battery_0_true_soc": "true_soc",
        "battery_0_true_soe": "true_soe",
        "battery_0_current_charge": "current_charge",
        "battery_0_charge_amount": "charge_amount",
        "battery_0_discharge_amount": "discharge_amount",
        "battery_0_reward": "wear_cost_battery",
        "balance_0_reward": "economic_balance_eur",
    }
    df.rename(
        columns={old: new for old, new in column_mapping.items() if old in df.columns},
        inplace=True,
    )

    if "cost_import_eur" not in df.columns and {"grid_import", "price_buy"}.issubset(df.columns):
        df["cost_import_eur"] = df["grid_import"] * df["price_buy"]

    if "revenue_export_eur" not in df.columns and {"grid_export", "price_sell"}.issubset(df.columns):
        df["revenue_export_eur"] = df["grid_export"] * df["price_sell"]

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")  # Converte la colonna timestamp in datetime

    # Normalizza il timezone: se i timestamp sono tz-aware, convertili nel timezone configurato (se fornito)
    if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
        if timezone:
            tz = pytz.timezone(timezone)
            timestamps = timestamps.dt.tz_convert(tz)
        timestamps = timestamps.dt.tz_localize(None)       # Porta i timestamp a naive per matplotlib
    df["timestamp"] = timestamps
    df.dropna(subset=["timestamp"], inplace=True)          # Rimuove eventuali righe senza timestamp valido
    df.sort_values("timestamp", inplace=True)              # Ordina per timestamp
    df.sort_values("timestamp", inplace=True)
    df["timestamp_original"] = df["timestamp"]
    df.set_index("timestamp", inplace=True)                # Imposta timestamp come indice

    # Rebase timeline to a regular range starting from the first timestamp.
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None and len(df.index) > 1:
        deltas = df.index.to_series().diff().dropna()
        if not deltas.empty:
            inferred_freq = deltas.median()
    if inferred_freq is None:
        inferred_freq = pd.Timedelta(minutes=15)
    aligned_index = pd.date_range(
        start=df.index.min(),
        periods=len(df.index),
        freq=inferred_freq
    )
    df.index = aligned_index

    # 1) Energie istantanee
    fig, ax = plt.subplots(figsize=(12, 5))
    control_batt = df["discharge_amount"] - df["charge_amount"] 
    control_grid = df["grid_import"] - df["grid_export"]
    ax.plot(df.index, df["consumption"], label="Load (kWh)", linewidth=2)
    ax.plot(df.index, df["pv_prod"], label="PV (kWh)", linewidth=2)
    ax.plot(df.index, control_batt, label="Battery Net Flow (kWh)", linewidth=1.8, linestyle="-.")
    ax.plot(df.index, control_grid, label="Grid Energy (kWh)", linewidth=1.8, linestyle="--")
    ax.set_title("Energy Flows per Step")
    ax.set_ylabel("Energy [kWh]")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    energy_path = f"{base_name}_energy.png"
    fig.tight_layout()
    fig.savefig(energy_path, dpi=160)
    plt.close(fig)

    # 2) Energia rete cumulativa e per step
    cumulative_import = df["grid_import"].cumsum()
    cumulative_export = df["grid_export"].cumsum()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax1.bar(df.index, df["grid_import"], label="Import step (kWh)", color="tab:blue", alpha=0.45, width=0.025)
    ax1.bar(df.index, -df["grid_export"], label="Export step (kWh)", color="tab:orange", alpha=0.45, width=0.025)
    ax1.set_ylabel("Energy per step [kWh]")
    ax1.set_xlabel("Time")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax2.plot(df.index, cumulative_import, label="Cumulative Import (kWh)", color="tab:blue", linewidth=2.2)
    ax2.plot(df.index, cumulative_export, label="Cumulative Export (kWh)", color="tab:orange", linewidth=2.2)
    ax2.set_ylabel("Cumulative Energy [kWh]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Grid Energy Exchange (Step & Cumulative)")
    grid_path = f"{base_name}_grid.png"
    fig.tight_layout()
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)

    # 3) Prezzi
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.plot(df.index, df["price_buy"], label="Price Buy (eur/kWh)", linewidth=2)
    ax.plot(df.index, df["price_sell"], label="Price Sell (eur/kWh)", linewidth=2)
    
    ax.set_title("Grid Prices Over Time")
    ax.set_ylabel("eur/kWh")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    prices_path = f"{base_name}_prices.png"
    fig.tight_layout()
    fig.savefig(prices_path, dpi=160)
    plt.close(fig)

    # 4) Batteria: SoE e SoC veri (se disponibili) + flussi
    if "soe" not in df.columns and "soc" in df.columns:
        df["soe"] = df["soc"]
    if "true_soe" in df.columns:
        df["true_soe"] = pd.to_numeric(df["true_soe"], errors="coerce")
    if "true_soc" in df.columns:
        df["true_soc"] = pd.to_numeric(df["true_soc"], errors="coerce")

    if "true_soe" in df.columns and df["true_soe"].notna().any():
        battery_soe_pct = df["true_soe"] * 100
        soe_label = "True SoE (%)"
    else:
        battery_soe_pct = df["soe"] * 100
        soe_label = "SoE (%)"

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax1.plot(df.index, battery_soe_pct, color="tab:blue", label=soe_label, linewidth=2.2)
    if "true_soc" in df.columns and df["true_soc"].notna().any():
        ax1.plot(
            df.index,
            df["true_soc"] * 100,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label="True SoC (%)",
        )
    ax1.set_ylabel("State [%]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax2.bar(df.index, df["charge_amount"], label="Charge (kWh)", color="tab:green", alpha=0.4, width=0.025)
    ax2.bar(df.index, -df["discharge_amount"], label="Discharge (kWh)", color="tab:red", alpha=0.4, width=0.025)
    ax2.set_ylabel("Charge / Discharge [kWh]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Battery State (SoE/SoC) and Flows")

    battery_path = f"{base_name}_battery.png"
    fig.tight_layout()
    fig.savefig(battery_path, dpi=160)
    plt.close(fig)

    # 4b) Animazione batteria a forma di icona
    # 5) Indicatori economici per step e cumulativi
    wear_cost_battery = -df["wear_cost_battery"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax1.bar(df.index, df["cost_import_eur"], label="Import Cost (eur/step)", color="tab:blue", alpha=0.45, width=0.025)
    ax1.bar(df.index, df["revenue_export_eur"], label="Export Revenue (eur/step)", color="tab:orange", alpha=0.45, width=0.025)
    ax1.bar(df.index, wear_cost_battery, label="Wear cost battery (eur/step)", color="tab:green", alpha=0.45, width=0.025)

    ax1.set_ylabel("eur/step")
    ax1.set_xlabel("Time")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax2.plot(
        df.index,
        df["economic_balance_eur"].cumsum(),
        label="Cumulative Balance (EUR)",
        color="tab:purple",
        linewidth=2.3,
    )
    ax2.set_ylabel("Cumulative Balance [EUR]", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Economic Performance")

    economics_path = f"{base_name}_economics.png"
    fig.tight_layout()
    fig.savefig(economics_path, dpi=160)
    plt.close(fig)

    # 6) Reward cumulativo per modulo (come vista "module_name,module_number,field")
    reward_series_specs = [
        ("(battery, 0, reward)", ["battery_0_reward", "wear_cost_battery"]),
        ("(grid, 0, reward)", ["grid_0_reward"]),
        ("(load, 0, reward)", ["load_0_reward"]),
        ("(pv, 0, reward)", ["pv_0_reward"]),
        ("(unbalanced_energy, 0, reward)", ["unbalanced_energy_0_reward", "balancing_0_reward"]),
        ("(balance, 0, reward)", ["balance_0_reward", "economic_balance_eur"]),
    ]
    cumulative_reward_series = []
    for label, candidates in reward_series_specs:
        source_col = next((col for col in candidates if col in df.columns), None)
        if source_col is None:
            continue
        series = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0).cumsum()
        cumulative_reward_series.append((label, series))

    module_rewards_path = None
    if cumulative_reward_series:
        fig, ax = plt.subplots(figsize=(12, 5))
        step_idx = np.arange(len(df), dtype=int)
        for label, series in cumulative_reward_series:
            ax.plot(step_idx, series.to_numpy(dtype=float), label=label, linewidth=2)

        ax.set_title("Cumulative Module Rewards")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative reward")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="module_name,module_number,field", loc="lower left")

        module_rewards_path = f"{base_name}_module_rewards.png"
        fig.tight_layout()
        fig.savefig(module_rewards_path, dpi=160)
        plt.close(fig)

    plot_paths = {                             # Restituisce i percorsi dei file generati
        "energy": energy_path,
        "grid": grid_path,
        "prices": prices_path,
        "battery": battery_path,
        "economics": economics_path,
    }
    if module_rewards_path is not None:
        plot_paths["module_rewards"] = module_rewards_path

    return plot_paths



def compute_offline_tariff_vectors(ts_series, local_timezone, price_config):

    hr = ts_series.dt.tz_convert(local_timezone).dt.hour

    # List di condizioni e valori risultanti
    condlist = []
    buy_choices = []
    sell_choices = []

    # Itera sulle fasce definite nel file YAML
    for band_name, band_data in price_config.items():
        buy_val = float(band_data['buy'])
        sell_val = float(band_data['sell'])

        # Bande con ranges
        if 'ranges' in band_data and band_data['ranges'] is not None:
            band_conditions = None

            # Ogni range è [start_hour, end_hour]
            for start, end in band_data['ranges']:
                # Se l’utente usa in YAML range inclusivi (es. 18–20)
                # li interpretiamo come ore intere: start ≤ hr ≤ end
                condition = hr.between(start, end)
                band_conditions = condition if band_conditions is None else (band_conditions | condition)

            condlist.append(band_conditions)

        else:
            # Bande senza ranges → si assume valida per tutte le ore non coperte da altre fasce
            # Per evitare comportamenti non deterministici, creiamo una condizione placeholder;
            # verrà assegnata *solo se nessuna delle fasce precedenti è valida* dopo np.select.
            condlist.append(np.full(len(hr), True, dtype=bool))

        buy_choices.append(buy_val)
        sell_choices.append(sell_val)

    # np.select valuta i condlist in ordine: la prima condizione vera viene assegnata
    price_buy_vec = np.select(condlist, buy_choices).astype(float)
    price_sell_vec = np.select(condlist, sell_choices).astype(float)

    return price_buy_vec, price_sell_vec


def add_module_columns(df, mapping):
    """
    Aggiunge colonne extra al DataFrame preservandone la MultiIndex e l'ordine dei moduli.

    I nuovi campi vengono raggruppati accanto alle colonne del relativo modulo,
    così l'ispezione resta ordinata anche dopo l'aggiunta di grandezze derivate.
    Se una serie ha lunghezza diversa dal DataFrame:
    - viene troncata se più lunga;
    - viene completata con NaN/NaT se più corta (incluso il caso vuoto).
    """
    def _align_values(values, target_len):
        arr = np.asarray(values)

        if arr.ndim == 0:
            return np.repeat(arr, target_len)

        arr = arr.reshape(-1)
        current_len = int(arr.shape[0])
        if current_len == target_len:
            return arr

        take = min(current_len, target_len)

        if np.issubdtype(arr.dtype, np.number):
            out = np.full(target_len, np.nan, dtype=float)
            if take > 0:
                out[:take] = arr[:take].astype(float, copy=False)
            return out

        if np.issubdtype(arr.dtype, np.datetime64):
            out = np.full(target_len, np.datetime64("NaT"), dtype="datetime64[ns]")
            if take > 0:
                out[:take] = arr[:take].astype("datetime64[ns]", copy=False)
            return out

        out = np.full(target_len, pd.NA, dtype=object)
        if take > 0:
            out[:take] = arr[:take]
        return out

    target_len = len(df)

    if not isinstance(df.columns, pd.MultiIndex):
        for column_key, values in mapping.items():
            df[column_key] = _align_values(values, target_len)
        return df

    module_order = list(dict.fromkeys(df.columns.get_level_values(0)))

    for column_key, values in mapping.items():
        df.loc[:, column_key] = _align_values(values, target_len)
        module_name = column_key[0] if isinstance(column_key, tuple) else column_key
        if module_name not in module_order:
            module_order.append(module_name)

    ordered_cols = []
    for module_name in module_order:
        ordered_cols.extend([col for col in df.columns if col[0] == module_name])

    df = df.loc[:, ordered_cols]
    return df
