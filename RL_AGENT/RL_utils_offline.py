from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from SIMULATOR.tools import compute_offline_tariff_vectors, load_pymgrid_scenario_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "./data/scenario_0_timeseries_hourly.csv"


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _is_pymgrid_bundle_mode(config: Dict[str, Any]) -> bool:
    scenario_cfg = config.get("scenario") or {}
    mode_value = (
        scenario_cfg.get("dataset_mode")
        or config.get("scenario_dataset_mode")
        or ""
    )
    return str(mode_value).strip().lower() == "pymgrid_bundle"


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_from_pymgrid_bundle(config: Dict[str, Any]) -> Dict[str, Any]:
    scenario_cfg = config.get("scenario") or {}
    ems_cfg = config.get("ems") or {}

    load_dataset_path = scenario_cfg.get("load_dataset_path") or config.get("scenario_load_dataset_path")
    pv_dataset_path = scenario_cfg.get("pv_dataset_path") or config.get("scenario_pv_dataset_path")
    grid_dataset_path = scenario_cfg.get("grid_dataset_path") or config.get("scenario_grid_dataset_path")
    if not (load_dataset_path and pv_dataset_path and grid_dataset_path):
        raise ValueError(
            "scenario.dataset_mode=pymgrid_bundle richiede load_dataset_path, pv_dataset_path e grid_dataset_path."
        )

    start_step = _as_optional_int(ems_cfg.get("start_step"))
    if start_step is None:
        start_step = _as_optional_int(scenario_cfg.get("initial_step"))
    if start_step is None:
        start_step = 0

    end_step = _as_optional_int(ems_cfg.get("end_step"))
    if end_step is None:
        final_step = _as_optional_int(scenario_cfg.get("final_step"))
        if final_step is not None:
            # scenario.final_step in pymgrid YAML is inclusive.
            end_step = final_step + 1

    start_datetime_utc = (
        scenario_cfg.get("start_datetime_utc")
        or config.get("scenario_start_datetime_utc")
        or "2020-01-01T00:00:00Z"
    )

    bundle = load_pymgrid_scenario_bundle(
        load_dataset_path=str(load_dataset_path),
        pv_dataset_path=str(pv_dataset_path),
        grid_dataset_path=str(grid_dataset_path),
        start_step=int(start_step),
        end_step=end_step,
        start_datetime_utc=str(start_datetime_utc),
        base_dir=PROJECT_ROOT,
    )

    return {
        "time_series": bundle["time_series"].copy().reset_index(drop=True),
        "pv_series": np.asarray(bundle["pv_series"], dtype=float),
        "load_series": np.asarray(bundle["load_series"], dtype=float),
        "timestamps": pd.Series(bundle["timestamps"]).reset_index(drop=True),
        "price_buy": np.asarray(bundle["price_buy"], dtype=float),
        "price_sell": np.asarray(bundle["price_sell"], dtype=float),
        "grid_series": np.asarray(bundle["grid_series"], dtype=float),
        "dataset_source": "pymgrid_bundle",
    }


def load_offline_timeseries(config: Dict[str, Any]) -> Dict[str, Any]:
    if _is_pymgrid_bundle_mode(config):
        return _load_from_pymgrid_bundle(config)

    rl_cfg = config.get("rl", {}) or {}
    scenario_cfg = config.get("scenario", {}) or {}
    dataset_path = (
        rl_cfg.get("dataset_path")
        or rl_cfg.get("dataset_path_train")
        or scenario_cfg.get("inference_dataset_path")
        or DEFAULT_DATASET_PATH
    )
    data_path = _resolve_project_path(str(dataset_path))
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    time_series = pd.read_csv(data_path)
    required_cols = {"datetime", "solar", "load"}
    missing_cols = required_cols - set(time_series.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {sorted(missing_cols)}")

    keep_columns = ["datetime", "solar", "load"]
    if "price_buy" in time_series.columns:
        keep_columns.append("price_buy")
    if "price_sell" in time_series.columns:
        keep_columns.append("price_sell")
    time_series = time_series[keep_columns].copy()

    time_series[["solar", "load"]] = (
        time_series[["solar", "load"]]
        .infer_objects(copy=False)
        .interpolate(limit_direction="both")
    )
    if time_series[["solar", "load"]].isna().to_numpy().any():
        raise ValueError("Dataset RL contiene valori NaN non interpolabili in solar/load.")

    time_series["datetime"] = pd.to_datetime(time_series["datetime"], utc=True, errors="coerce")

    pv_series = time_series["solar"].clip(lower=0).astype(float)
    load_series = time_series["load"].astype(float)
    timestamps = time_series["datetime"]

    ems_cfg = config.get("ems", {}) or {}
    timezone = str(ems_cfg.get("timezone", "UTC"))
    price_bands = ems_cfg.get("price_bands")

    if {"price_buy", "price_sell"}.issubset(time_series.columns):
        price_buy = pd.to_numeric(time_series["price_buy"], errors="coerce").to_numpy(dtype=float)
        price_sell = pd.to_numeric(time_series["price_sell"], errors="coerce").to_numpy(dtype=float)

        if np.isnan(price_buy).any() or np.isnan(price_sell).any():
            if not price_bands:
                raise ValueError(
                    "Dataset RL contiene NaN in price_buy/price_sell e ems.price_bands non e' disponibile."
                )
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
        if not price_bands:
            raise ValueError(
                "Dataset RL senza colonne price_buy/price_sell: serve ems.price_bands nel config."
            )
        price_buy, price_sell = compute_offline_tariff_vectors(
            ts_series=timestamps,
            local_timezone=timezone,
            price_config=price_bands,
        )

    emissions = np.zeros(len(price_buy), dtype=float)
    grid_series = np.stack([price_buy, price_sell, emissions], axis=1)

    return {
        "time_series": time_series,
        "pv_series": pv_series.to_numpy(),
        "load_series": load_series.to_numpy(),
        "timestamps": timestamps,
        "price_buy": price_buy,
        "price_sell": price_sell,
        "grid_series": grid_series,
        "dataset_source": "csv",
    }
