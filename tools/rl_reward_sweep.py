from __future__ import annotations

import argparse
import hashlib
import csv
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "RL_AGENT" / "ems_offline_RL_agent.py"


def _parse_number(text: str) -> Any:
    text = str(text).strip()
    if not text:
        raise ValueError("Empty numeric value")
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_csv_list(text: str) -> List[Any]:
    items = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(_parse_number(part))
    if not items:
        raise ValueError(f"No values parsed from: {text!r}")
    return items


def _set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _format_value_for_name(value: Any) -> str:
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        s = f"{value:.6g}"
        s = s.replace("-", "neg").replace(".", "p")
        return s
    s = str(value)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")
    return s or "x"


def _short_param_name(dotted_key: str) -> str:
    last = dotted_key.split(".")[-1]
    mapping = {
        "coeff_economic": "ce",
        "coeff_wear_cost": "cwear",
        "coeff_SSR": "cssr",
        "coeff_action_violation": "cact",
        "coeff_soc_violation": "csoc",
        "coeff_bad_logic": "cblog",
        "coeff_cyclic_aging": "ccyc",
        "coeff_soh_calendar": "ccal",
        "coeff_action_smoothness": "cas",
        "coeff_micro_throughput": "cmt",
        "micro_throughput_kwh": "mtk",
    }
    return mapping.get(last, last)


def _shorten_run_name_parts(pieces: Sequence[str], max_len: int = 50) -> str:
    run_name = "__".join(pieces)
    if len(run_name) <= max_len:
        return run_name

    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:8]
    prefix = pieces[0] if pieces else "run"
    seed_part = ""
    if pieces and pieces[-1].startswith("seed"):
        seed_part = pieces[-1]

    keep_parts = [prefix]
    for part in pieces[1:]:
        if part == seed_part:
            continue
        keep_parts.append(part)
        candidate = "__".join([p for p in keep_parts if p] + ([seed_part] if seed_part else []) + [digest])
        if len(candidate) > max_len:
            keep_parts.pop()
            break

    trimmed = "__".join([p for p in keep_parts if p] + ([seed_part] if seed_part else []) + [digest])
    if len(trimmed) <= max_len:
        return trimmed

    base = "__".join([prefix] + ([seed_part] if seed_part else []) + [digest])
    if len(base) <= max_len:
        return base

    max_prefix = max(6, max_len - (len(digest) + 2))
    return f"{prefix[:max_prefix]}_{digest}"


def _resolve_latest_run_dir(base_output: Path, mode: str, run_name: str) -> Optional[Path]:
    mode_dir = base_output / mode
    if not mode_dir.exists():
        return None
    candidates = [p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith(run_name + "_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _start_tensorboard(
    *,
    logdir: Path,
    host: str,
    port: int,
    open_browser_flag: bool,
    log_path: Path,
    cwd: Path,
    env: Dict[str, str],
) -> Optional[subprocess.Popen]:
    logdir.mkdir(parents=True, exist_ok=True)

    url = f"http://{host}:{port}"
    if _is_port_open(host, port):
        print(f"[TensorBoard] Already running at {url} (not starting a new one).")
        if open_browser_flag:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return None

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = log_path.open("w", encoding="utf-8")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
    except OSError:
        handle = None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=handle if handle is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        print(f"[WARN] Unable to start TensorBoard: {exc}")
        print(f"[INFO] You can run it manually with: {' '.join(cmd)}")
        return None

    time.sleep(0.5)
    print(f"[TensorBoard] Running at {url} (logdir={logdir})")
    if open_browser_flag:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return proc


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _parse_final_report(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}

    def find_float(pattern: str) -> Optional[float]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            return None
        raw = match.group(1).strip()
        if raw.lower().startswith("n"):
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def find_percent(pattern: str) -> Optional[float]:
        value = find_float(pattern)
        if value is None:
            return None
        return float(value) / 100.0

    return {
        "kpi_cost_import_eur": find_float(r"Costo energia import\s*:\s*([-\d.]+)"),
        "kpi_revenue_export_eur": find_float(r"Ricavo export\s*:\s*([-\d.]+)"),
        "kpi_battery_wear_cost_eur": find_float(r"Usura batteria .*:\s*([-\d.]+)"),
        "kpi_total_reward_eur": find_float(r"Reward complessivo\s*:\s*([-\d.]+)"),
        "kpi_soc_final": find_percent(r"SOC finale .*:\s*([-\d.]+)%"),
        "kpi_soh_final": find_percent(r"SOH finale\s*:\s*([-\d.]+)%"),
    }


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    seed: Optional[int]
    params: Dict[str, Any]
    config_path: Path


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _build_runs(
    *,
    base_config: Path,
    out_configs_dir: Path,
    run_prefix: str,
    seeds: Sequence[Optional[int]],
    param_values: Dict[str, Sequence[Any]],
) -> List[RunSpec]:
    base_cfg = _load_yaml(base_config)
    keys = list(param_values.keys())
    values = [list(param_values[k]) for k in keys]

    runs: List[RunSpec] = []
    for seed in seeds:
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            pieces = [run_prefix]
            for k in keys:
                pieces.append(f"{_short_param_name(k)}{_format_value_for_name(params[k])}")
            if seed is not None:
                pieces.append(f"seed{seed}")
            run_name = _shorten_run_name_parts(pieces)

            cfg = json.loads(json.dumps(base_cfg))
            if seed is not None:
                _set_dotted(cfg, "rl.seed", int(seed))
            _set_dotted(cfg, "rl.run_name", run_name)
            _set_dotted(cfg, "rl.tensorboard_auto_open", False)
            for k, v in params.items():
                _set_dotted(cfg, k, v)

            cfg_path = out_configs_dir / f"{run_name}.yml"
            _dump_yaml(cfg, cfg_path)
            runs.append(RunSpec(run_name=run_name, seed=seed if seed is None else int(seed), params=params, config_path=cfg_path))
    return runs


def _ensure_script_exists() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")


def _run_subprocess(
    *,
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)
        return int(proc.wait())


def _load_existing_run_names(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    done = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row or {}).get("run_name")
            status = (row or {}).get("status")
            if name and status in {"ok", "failed"}:
                done.add(str(name))
    return done


def _append_summary(summary_csv: Path, row: Dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    fieldnames = list(row.keys())
    if exists:
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header:
            fieldnames = header

    with summary_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reward-coefficients sweep runner for RL_AGENT/ems_offline_RL_agent.py")
    parser.add_argument("--base-config", type=str, default="", help="Base YAML config (e.g. configs/controllers/rl/opsd/...).")
    parser.add_argument("--sweep-spec", type=str, default="", help="Optional YAML spec file (see example in tools/).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Base output directory for runs (e.g. outputs/RL_REWARD_SWEEP_...). "
            "If omitted, defaults to outputs/RL_REWARD_SWEEP_<dataset>_<horizon> inferred from base config name."
        ),
    )
    parser.add_argument("--run-prefix", type=str, default="", help="Prefix for run names (default: base config stem).")
    parser.add_argument("--param", action="append", default=[], help="Repeatable: dotted.path=comma,separated,values")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds (or empty to use sweep-spec / default).")
    parser.add_argument("--max-parallel", type=int, default=1, help="Parallel subprocesses (be careful on CPU).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda (passed to ems_offline_RL_agent.py).")
    parser.add_argument("--omp-threads", type=int, default=0, help="Set OMP_NUM_THREADS for each run (0=leave as-is).")
    parser.add_argument("--mkl-threads", type=int, default=0, help="Set MKL_NUM_THREADS for each run (0=leave as-is).")
    parser.add_argument("--tensorboard", action="store_true", help="Start a single TensorBoard for the sweep.")
    parser.add_argument("--tensorboard-host", type=str, default="localhost", help="TensorBoard host.")
    parser.add_argument("--tensorboard-port", type=int, default=6006, help="TensorBoard port.")
    parser.add_argument("--tensorboard-open", action="store_true", help="Open TensorBoard in a browser.")
    parser.add_argument("--tensorboard-keep-alive", action="store_true", help="Do not stop TensorBoard when sweep ends.")
    parser.add_argument("--train-episodes", type=int, default=None, help="Override training episodes.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override training steps per episode.")
    parser.add_argument("--eval", action="store_true", help="Run eval after training.")
    parser.add_argument("--test", action="store_true", help="Run test after training (writes final_report.txt).")
    parser.add_argument("--test-start-step", type=int, default=0, help="Test start step.")
    parser.add_argument("--test-steps", type=int, default=6000, help="Test episode steps.")
    parser.add_argument("--resume", action="store_true", help="Skip runs already present in summary CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    _ensure_script_exists()

    sweep_spec: Dict[str, Any] = {}
    if args.sweep_spec:
        sweep_spec = _load_yaml(Path(args.sweep_spec))

    base_config_str = args.base_config or sweep_spec.get("base_config") or ""
    if not base_config_str:
        raise ValueError("Provide --base-config or sweep-spec.base_config")
    base_config = Path(base_config_str)
    if not base_config.is_absolute():
        base_config = (PROJECT_ROOT / base_config).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    output_dir_str = args.output_dir or sweep_spec.get("output_dir") or ""
    if not output_dir_str:
        match = re.match(r"^params_RL_agent_(?P<tag>[^_]+)_(?P<horizon>.+)$", base_config.stem)
        if match:
            inferred_name = f"RL_REWARD_SWEEP_{match.group('tag')}_{match.group('horizon')}"
        else:
            inferred_name = f"RL_REWARD_SWEEP_{base_config.stem}"
        output_dir = (PROJECT_ROOT / "outputs" / inferred_name).resolve()
        print(f"[INFO] output_dir not provided; using inferred: {output_dir}")
    else:
        output_dir = Path(output_dir_str)
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
    meta_dir = output_dir / "sweep_meta"
    configs_dir = meta_dir / "configs"
    logs_dir = meta_dir / "logs"
    summary_csv = meta_dir / "summary.csv"

    run_prefix = args.run_prefix or str(sweep_spec.get("run_prefix") or "") or base_config.stem

    seeds: List[Optional[int]]
    seeds_str = str(args.seeds or "").strip()
    spec_seeds = sweep_spec.get("seeds", None)
    if seeds_str:
        seeds = [int(x) for x in _parse_csv_list(seeds_str)]
    elif isinstance(spec_seeds, list) and spec_seeds:
        seeds = [int(x) for x in spec_seeds]
    elif isinstance(spec_seeds, str) and str(spec_seeds).strip():
        seeds = [int(x) for x in _parse_csv_list(str(spec_seeds).strip())]
    else:
        seeds = [42]

    param_values: Dict[str, Sequence[Any]] = {}
    spec_params = (sweep_spec.get("params") or {}) if isinstance(sweep_spec.get("params"), dict) else {}
    for k, v in spec_params.items():
        if isinstance(v, list):
            param_values[str(k)] = v

    for item in args.param:
        if "=" not in item:
            raise ValueError(f"Invalid --param {item!r}, expected dotted.path=v1,v2,...")
        k, raw = item.split("=", 1)
        k = k.strip()
        param_values[k] = _parse_csv_list(raw)

    if not param_values:
        raise ValueError("No parameters to sweep: provide sweep-spec.params or at least one --param.")

    tb_spec = sweep_spec.get("tensorboard", None)
    if args.tensorboard:
        do_tensorboard = True
        tb_host = str(args.tensorboard_host)
        tb_port = int(args.tensorboard_port)
        tb_open = bool(args.tensorboard_open)
        tb_keep_alive = bool(args.tensorboard_keep_alive)
    else:
        do_tensorboard = bool(tb_spec is True or (isinstance(tb_spec, dict) and tb_spec.get("enabled", True)))
        tb_host = str((tb_spec.get("host") if isinstance(tb_spec, dict) else None) or "localhost")
        tb_port = int((tb_spec.get("port") if isinstance(tb_spec, dict) else None) or 6006)
        tb_open = bool((tb_spec.get("open_browser") if isinstance(tb_spec, dict) else None) or False)
        tb_keep_alive = bool((tb_spec.get("keep_alive") if isinstance(tb_spec, dict) else None) or False)

    do_eval = bool(args.eval or bool(sweep_spec.get("eval", False)))
    do_test = bool(args.test or bool(sweep_spec.get("test", False)))
    test_start_step = int(sweep_spec.get("test_start_step", args.test_start_step))
    test_steps = int(sweep_spec.get("test_steps", args.test_steps))

    runs = _build_runs(
        base_config=base_config,
        out_configs_dir=configs_dir,
        run_prefix=run_prefix,
        seeds=seeds,
        param_values=param_values,
    )

    if args.dry_run:
        print(f"Planned runs: {len(runs)}")
        for spec in runs[:50]:
            print(spec.run_name)
        if len(runs) > 50:
            print("...")
        return 0

    done_run_names = _load_existing_run_names(summary_csv) if args.resume else set()

    train_base = output_dir

    env_base = dict(os.environ)
    env_base.setdefault("MPLBACKEND", "Agg")
    env_base.setdefault("PYTHONUTF8", "1")
    if args.omp_threads and int(args.omp_threads) > 0:
        env_base["OMP_NUM_THREADS"] = str(int(args.omp_threads))
    if args.mkl_threads and int(args.mkl_threads) > 0:
        env_base["MKL_NUM_THREADS"] = str(int(args.mkl_threads))

    tb_proc: Optional[subprocess.Popen] = None
    if do_tensorboard:
        tb_proc = _start_tensorboard(
            logdir=output_dir / "train",
            host=tb_host,
            port=tb_port,
            open_browser_flag=tb_open,
            log_path=logs_dir / "tensorboard.log",
            cwd=PROJECT_ROOT,
            env=dict(env_base),
        )

    max_parallel = max(1, int(args.max_parallel))
    queue = list(runs)

    active: List[Tuple[RunSpec, subprocess.Popen, Path, Any, List[str]]] = []

    def start_one(spec: RunSpec) -> None:
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--mode", "train", "--config", str(spec.config_path), "--run-name", spec.run_name, "--output-dir", str(train_base), "--device", str(args.device)]
        if args.train_episodes is not None:
            cmd += ["--episodes", str(int(args.train_episodes))]
        if args.train_steps is not None:
            cmd += ["--train-steps", str(int(args.train_steps))]

        env = dict(env_base)
        log_path = logs_dir / f"{spec.run_name}__train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        active.append((spec, proc, log_path, handle, cmd))

    def finish_one(item: Tuple[RunSpec, subprocess.Popen, Path, Any, List[str]]) -> None:
        spec, proc, train_log_path, handle, train_cmd = item
        try:
            rc = int(proc.wait())
        finally:
            try:
                handle.close()
            except Exception:
                pass

        train_dir = _resolve_latest_run_dir(output_dir, "train", spec.run_name)
        model_path = None
        if train_dir is not None:
            candidate = train_dir / "model_final.zip"
            if candidate.exists():
                model_path = candidate

        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": spec.run_name,
            "status": "ok" if rc == 0 and model_path is not None else "failed",
            "train_return_code": rc,
            "train_log": str(train_log_path),
            "config": str(spec.config_path),
            "train_dir": str(train_dir) if train_dir is not None else "",
            "model_path": str(model_path) if model_path is not None else "",
        }
        for k, v in spec.params.items():
            row[k] = v
        if spec.seed is not None:
            row["seed"] = int(spec.seed)

        if row["status"] != "ok":
            _append_summary(summary_csv, row)
            done_run_names.add(spec.run_name)
            return

        if do_eval:
            eval_cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--mode",
                "eval",
                "--config",
                str(spec.config_path),
                "--model-path",
                str(model_path),
                "--run-name",
                spec.run_name,
                "--output-dir",
                str(output_dir),
                "--device",
                str(args.device),
                "--no-random-start",
            ]
            eval_log = logs_dir / f"{spec.run_name}__eval.log"
            eval_rc = _run_subprocess(cmd=eval_cmd, cwd=PROJECT_ROOT, env=dict(env_base), log_path=eval_log)
            eval_dir = _resolve_latest_run_dir(output_dir, "eval", spec.run_name)
            summary = _read_json(eval_dir / "evaluation_summary.json") if eval_dir is not None else {}
            row.update(
                {
                    "eval_return_code": eval_rc,
                    "eval_log": str(eval_log),
                    "eval_dir": str(eval_dir) if eval_dir is not None else "",
                    "eval_mean_reward": summary.get("mean_reward"),
                }
            )

        if do_test:
            test_cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--mode",
                "test",
                "--config",
                str(spec.config_path),
                "--model-path",
                str(model_path),
                "--run-name",
                spec.run_name,
                "--output-dir",
                str(output_dir),
                "--device",
                str(args.device),
                "--no-random-start",
                "--test-start-step",
                str(int(test_start_step)),
                "--test-steps",
                str(int(test_steps)),
            ]
            test_log = logs_dir / f"{spec.run_name}__test.log"
            test_rc = _run_subprocess(cmd=test_cmd, cwd=PROJECT_ROOT, env=dict(env_base), log_path=test_log)
            test_dir = _resolve_latest_run_dir(output_dir, "test", spec.run_name)
            row.update(
                {
                    "test_return_code": test_rc,
                    "test_log": str(test_log),
                    "test_dir": str(test_dir) if test_dir is not None else "",
                }
            )
            if test_dir is not None:
                row.update(_parse_final_report(test_dir / "final_report.txt"))

        _append_summary(summary_csv, row)
        done_run_names.add(spec.run_name)

    while queue or active:
        while queue and len(active) < max_parallel:
            spec = queue.pop(0)
            if args.resume and spec.run_name in done_run_names:
                continue
            start_one(spec)

        still_active: List[Tuple[RunSpec, subprocess.Popen, Path, Any, List[str]]] = []
        progressed = False
        for item in active:
            spec, proc, _, _, _ = item
            if proc.poll() is None:
                still_active.append(item)
                continue
            finish_one(item)
            progressed = True
        active = still_active
        if not progressed and active:
            time.sleep(0.5)

    print(f"Done. Summary: {summary_csv}")
    if tb_proc is not None and tb_proc.poll() is None and not tb_keep_alive:
        try:
            tb_proc.terminate()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
