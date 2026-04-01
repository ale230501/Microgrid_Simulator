from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "RL_AGENT" / "ems_offline_RL_agent.py"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _strip_timestamp(name: str) -> str:
    return re.sub(r"_\d{8}_\d{6}$", "", str(name))


def _resolve_latest_run_dir(base_output: Path, mode: str, run_name: str) -> Optional[Path]:
    mode_dir = base_output / mode
    if not mode_dir.exists():
        return None
    candidates = [p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith(run_name + "_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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


def _find_config_in_run_dir(run_dir: Path, run_name: str) -> Optional[Path]:
    cfgs = sorted(run_dir.glob("*.yml")) + sorted(run_dir.glob("*.yaml"))
    if not cfgs:
        return None
    preferred = run_dir / f"{run_name}.yml"
    if preferred.exists():
        return preferred
    return cfgs[0]


@dataclass(frozen=True)
class EvalTarget:
    run_name: str
    train_dir: Path
    model_path: Path
    config_path: Path


def _discover_targets(train_root: Path) -> List[EvalTarget]:
    targets: List[EvalTarget] = []
    if not train_root.exists():
        return targets
    for run_dir in sorted([p for p in train_root.iterdir() if p.is_dir()]):
        run_name = _strip_timestamp(run_dir.name)
        model_path = run_dir / "model_final.zip"
        if not model_path.exists():
            continue
        cfg_path = _find_config_in_run_dir(run_dir, run_name=run_name)
        if cfg_path is None:
            continue
        targets.append(EvalTarget(run_name=run_name, train_dir=run_dir, model_path=model_path, config_path=cfg_path))
    return targets


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel evaluation runner for many trained RL models.")
    parser.add_argument("--sweep-spec", type=str, default="", help="Optional YAML spec file.")
    parser.add_argument("--base-output", type=str, default="", help="Base output directory containing train/ (required if no spec).")
    parser.add_argument("--train-dir", type=str, default="", help="Override train directory (default: <base-output>/train).")
    parser.add_argument("--max-parallel", type=int, default=None, help="Parallel eval subprocesses.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (passed to ems_offline_RL_agent.py).")
    parser.add_argument("--omp-threads", type=int, default=None, help="Set OMP_NUM_THREADS for each eval (0=leave as-is).")
    parser.add_argument("--mkl-threads", type=int, default=None, help="Set MKL_NUM_THREADS for each eval (0=leave as-is).")
    parser.add_argument("--eval-start-step", type=int, default=None, help="Override rl.eval_start_step for all runs.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override rl.eval_episode_steps for all runs.")
    parser.add_argument(
        "--eval-random-start",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Force eval_random_start for all runs (overrides config).",
    )
    parser.add_argument("--no-random-start", action="store_true", help="Force eval_random_start=false at runtime.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions in eval (passes --stochastic).")
    parser.add_argument("--resume", action="store_true", help="Skip runs already present in summary CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered runs and exit.")
    parser.add_argument(
        "--allow-tensorboard-auto-open",
        action="store_true",
        help="Do not override rl.tensorboard_auto_open (default: force false to avoid many browser opens).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Eval script not found: {EVAL_SCRIPT}")

    spec: Dict[str, Any] = {}
    if args.sweep_spec:
        spec = _load_yaml(Path(args.sweep_spec))

    max_parallel = int(args.max_parallel) if args.max_parallel is not None else int(spec.get("max_parallel", 4))
    device = str(args.device) if args.device is not None else str(spec.get("device", "cpu"))
    omp_threads = int(args.omp_threads) if args.omp_threads is not None else int(spec.get("omp_threads", 0))
    mkl_threads = int(args.mkl_threads) if args.mkl_threads is not None else int(spec.get("mkl_threads", 0))
    eval_start_step = int(args.eval_start_step) if args.eval_start_step is not None else spec.get("eval_start_step", None)
    eval_steps = int(args.eval_steps) if args.eval_steps is not None else spec.get("eval_steps", None)
    eval_random_start = args.eval_random_start if args.eval_random_start is not None else spec.get("eval_random_start", None)
    if isinstance(eval_random_start, bool):
        eval_random_start = "true" if eval_random_start else "false"
    if eval_random_start is not None:
        eval_random_start = str(eval_random_start).strip().lower()
        if eval_random_start not in {"true", "false"}:
            raise ValueError("eval_random_start must be true/false")

    no_random_start = bool(args.no_random_start or bool(spec.get("no_random_start", False)))
    stochastic = bool(args.stochastic or bool(spec.get("stochastic", False)))
    resume = bool(args.resume or bool(spec.get("resume", False)))
    allow_tb_auto_open = bool(args.allow_tensorboard_auto_open or bool(spec.get("allow_tensorboard_auto_open", False)))

    base_output_str = args.base_output or spec.get("base_output") or spec.get("output_dir") or ""
    if not base_output_str:
        raise ValueError("Provide --base-output or sweep-spec.base_output/output_dir")
    base_output = Path(base_output_str)
    if not base_output.is_absolute():
        base_output = (PROJECT_ROOT / base_output).resolve()

    train_dir_str = args.train_dir or spec.get("train_dir") or ""
    train_root = Path(train_dir_str) if train_dir_str else (base_output / "train")
    if not train_root.is_absolute():
        train_root = (PROJECT_ROOT / train_root).resolve()

    meta_dir = base_output / "eval_meta"
    configs_dir = meta_dir / "configs"
    logs_dir = meta_dir / "logs"
    summary_csv = meta_dir / "summary.csv"

    targets = _discover_targets(train_root)
    if args.dry_run:
        print(f"Found {len(targets)} models under: {train_root}")
        for t in targets[:50]:
            print(t.run_name)
        if len(targets) > 50:
            print("...")
        return 0

    done_run_names = _load_existing_run_names(summary_csv) if resume else set()

    env_base = dict(os.environ)
    env_base.setdefault("MPLBACKEND", "Agg")
    env_base.setdefault("PYTHONUTF8", "1")
    if omp_threads and int(omp_threads) > 0:
        env_base["OMP_NUM_THREADS"] = str(int(omp_threads))
    if mkl_threads and int(mkl_threads) > 0:
        env_base["MKL_NUM_THREADS"] = str(int(mkl_threads))

    max_parallel = max(1, int(max_parallel))
    queue = [t for t in targets if not (resume and t.run_name in done_run_names)]

    active: List[Tuple[EvalTarget, subprocess.Popen, Any, Path, Path, List[str]]] = []

    def start_one(target: EvalTarget) -> None:
        cfg_path = target.config_path
        cfg_override_needed = (
            (not allow_tb_auto_open)
            or (eval_start_step is not None)
            or (eval_steps is not None)
            or (eval_random_start is not None)
        )
        if cfg_override_needed:
            cfg = _load_yaml(cfg_path)
            rl_cfg = cfg.setdefault("rl", {})
            if not allow_tb_auto_open:
                rl_cfg["tensorboard_auto_open"] = False
            if eval_start_step is not None:
                rl_cfg["eval_start_step"] = int(eval_start_step)
            if eval_steps is not None:
                rl_cfg["eval_episode_steps"] = int(eval_steps)
            if eval_random_start is not None:
                rl_cfg["eval_random_start"] = (eval_random_start == "true")
            tmp_cfg = configs_dir / f"{target.run_name}.yml"
            _dump_yaml(cfg, tmp_cfg)
            cfg_path = tmp_cfg

        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--mode",
            "eval",
            "--config",
            str(cfg_path),
            "--model-path",
            str(target.model_path),
            "--output-dir",
            str(base_output),
            "--run-name",
            target.run_name,
            "--device",
            str(device),
        ]
        effective_no_random_start = no_random_start
        if eval_random_start is not None:
            effective_no_random_start = (eval_random_start == "false")
        if effective_no_random_start:
            cmd.append("--no-random-start")
        if eval_start_step is not None:
            cmd += ["--eval-start-step", str(int(eval_start_step))]
        if eval_steps is not None:
            cmd += ["--steps", str(int(eval_steps))]
        if stochastic:
            cmd.append("--stochastic")

        log_path = logs_dir / f"{target.run_name}__eval.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=dict(env_base), stdout=handle, stderr=subprocess.STDOUT)
        active.append((target, proc, handle, log_path, cfg_path, cmd))

    def finish_one(item: Tuple[EvalTarget, subprocess.Popen, Any, Path, Path, List[str]]) -> None:
        target, proc, handle, log_path, cfg_path, cmd = item
        try:
            rc = int(proc.wait())
        finally:
            try:
                handle.close()
            except Exception:
                pass

        eval_dir = _resolve_latest_run_dir(base_output, "eval", target.run_name)
        summary = _read_json(eval_dir / "evaluation_summary.json") if eval_dir is not None else {}
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": target.run_name,
            "status": "ok" if rc == 0 else "failed",
            "eval_return_code": rc,
            "eval_log": str(log_path),
            "config": str(cfg_path),
            "train_dir": str(target.train_dir),
            "model_path": str(target.model_path),
            "eval_dir": str(eval_dir) if eval_dir is not None else "",
            "eval_mean_reward": summary.get("mean_reward"),
        }
        _append_summary(summary_csv, row)
        done_run_names.add(target.run_name)

    while queue or active:
        while queue and len(active) < max_parallel:
            start_one(queue.pop(0))

        still_active: List[Tuple[EvalTarget, subprocess.Popen, Any, Path, Path, List[str]]] = []
        progressed = False
        for item in active:
            if item[1].poll() is None:
                still_active.append(item)
                continue
            finish_one(item)
            progressed = True
        active = still_active
        if not progressed and active:
            time.sleep(0.5)

    print(f"Done. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
