import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def _safe_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def _parse_final_report(path: Path) -> dict:
    content = path.read_text(encoding="utf-8", errors="ignore")
    sim_type = _find_single(content, r"Tipo simulazione\s*:\s*(.+)")
    soh = _find_single(content, r"SOH finale\s*:\s*([0-9.,]+)%")
    import_cost = _find_single(content, r"Costo energia import\s*:\s*([0-9.,]+)\s*EUR")
    export_rev = _find_single(content, r"Ricavo export\s*:\s*([0-9.,]+)\s*EUR")
    wear_cost = _find_single(content, r"Usura batteria \(stimata\)\s*:\s*([0-9.,]+)\s*EUR")

    if soh is None or import_cost is None or export_rev is None:
        return {}

    return {
        "sim_type": sim_type or "UNKNOWN",
        "soh_final": _safe_float(soh),
        "import_cost": _safe_float(import_cost),
        "export_revenue": _safe_float(export_rev),
        "wear_cost": _safe_float(wear_cost) if wear_cost is not None else None,
    }


def _find_single(text: str, pattern: str):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _compute_pareto(points):
    # Maximize SOH (x), minimize net cost (y). Sort by SOH descending and keep best cost.
    ordered = sorted(points, key=lambda p: (-p["soh_final"], p["net_cost"]))
    front = []
    best_cost = None
    for p in ordered:
        if best_cost is None or p["net_cost"] < best_cost - 1e-9:
            front.append(p)
            best_cost = p["net_cost"]
    return front


def _compute_pareto_wear(points):
    # Minimize wear cost (x) and net cost (y). Sort by wear ascending and keep best cost.
    ordered = sorted(points, key=lambda p: (p["wear_cost"], p["net_cost"]))
    front = []
    best_cost = None
    for p in ordered:
        if best_cost is None or p["net_cost"] < best_cost - 1e-9:
            front.append(p)
            best_cost = p["net_cost"]
    return front


def _load_reward_coeffs(report_dir: Path) -> dict:
    cfg = next(report_dir.glob("*.yml"), None)
    if cfg is None:
        # Try sibling test_configs using run name (strip timestamp suffix).
        run_name = re.sub(r"_\d{8}_\d{6}$", "", report_dir.name)
        if report_dir.parent.name == "test":
            candidate = report_dir.parent.parent / "test_configs" / f"{run_name}.yml"
            if candidate.exists():
                cfg = candidate
    if cfg is None:
        return {}
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    reward = ((data.get("rl") or {}).get("reward") or {})
    if not isinstance(reward, dict):
        return {}
    return {
        "rl.reward.coeff_economic": reward.get("coeff_economic"),
        "rl.reward.coeff_wear_cost": reward.get("coeff_wear_cost"),
        "rl.reward.coeff_action_violation": reward.get("coeff_action_violation"),
        "rl.reward.coeff_soc_violation": reward.get("coeff_soc_violation"),
        "rl.reward.coeff_bad_logic": reward.get("coeff_bad_logic"),
        "rl.reward.coeff_soh_calendar": reward.get("coeff_soh_calendar"),
        "rl.reward.coeff_cyclic_aging": reward.get("coeff_cyclic_aging"),
        "rl.reward.coeff_SSR": reward.get("coeff_SSR"),
        "rl.reward.coeff_action_smoothness": reward.get("coeff_action_smoothness"),
        "rl.reward.coeff_micro_throughput": reward.get("coeff_micro_throughput"),
        "rl.reward.micro_throughput_kwh": reward.get("micro_throughput_kwh"),
        "rl.reward.sell_discount": reward.get("sell_discount"),
        "rl.reward.bad_logic_discount": reward.get("bad_logic_discount"),
        "rl.reward.bad_logic_aging_weight": reward.get("bad_logic_aging_weight"),
        "rl.reward.action_tolerance": reward.get("action_tolerance"),
        "rl.reward.soc_tolerance": reward.get("soc_tolerance"),
        "rl.reward.calendar_aging_per_step": reward.get("calendar_aging_per_step"),
        "rl.reward.scale_reward_components": reward.get("scale_reward_components"),
        "rl.reward.scale_economic": reward.get("scale_economic"),
        "rl.reward.scale_action_violation": reward.get("scale_action_violation"),
        "rl.reward.scale_wear_cost": reward.get("scale_wear_cost"),
        "rl.reward.scale_ssr": reward.get("scale_ssr"),
    }


def _format_reward_coeffs(coeffs: dict) -> str:
    if not coeffs:
        return "reward_coeffs: n/a"
    short_names = {
        "rl.reward.coeff_economic": "ce",
        "rl.reward.coeff_wear_cost": "cwear",
        "rl.reward.coeff_action_violation": "cact",
        "rl.reward.coeff_soc_violation": "csoc",
        "rl.reward.coeff_bad_logic": "cblog",
        "rl.reward.coeff_soh_calendar": "ccal",
        "rl.reward.coeff_cyclic_aging": "ccyc",
        "rl.reward.coeff_SSR": "cssr",
        "rl.reward.coeff_action_smoothness": "cas",
        "rl.reward.coeff_micro_throughput": "cmt",
        "rl.reward.micro_throughput_kwh": "mtk",
        "rl.reward.sell_discount": "sdisc",
        "rl.reward.bad_logic_discount": "bdisc",
        "rl.reward.bad_logic_aging_weight": "blaw",
        "rl.reward.action_tolerance": "atol",
        "rl.reward.soc_tolerance": "stol",
        "rl.reward.calendar_aging_per_step": "calstep",
        "rl.reward.scale_reward_components": "scale",
        "rl.reward.scale_economic": "sce",
        "rl.reward.scale_action_violation": "sact",
        "rl.reward.scale_wear_cost": "swear",
        "rl.reward.scale_ssr": "sssr",
    }
    parts = []
    for key, value in coeffs.items():
        if value is None:
            continue
        label = short_names.get(key, key)
        parts.append(f"{label}: {value}")
    return "reward_coeffs: " + (", ".join(parts) if parts else "n/a")


def _load_mpc_wear_cost_scale(report_dir: Path):
    cfg = next(report_dir.glob("*.yml"), None)
    if cfg is None:
        return None
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not isinstance(data, dict):
        return None
    mpc = data.get("mpc") or {}
    if not isinstance(mpc, dict):
        return None
    return mpc.get("wear_cost_scale")


def _format_mpc_wear_cost_scale(value) -> str:
    if value is None:
        return "mpc.wear_cost_scale: n/a"
    return f"mpc.wear_cost_scale: {value}"


def _plot_interactive(points, pareto, out_path: Path, x_key: str, x_label: str, title: str, auto_open: bool):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly is not installed; skipping interactive plot.")
        return

    def make_trace(label, color, symbol, pts):
        hover = []
        xs = []
        ys = []
        custom = []
        for p in pts:
            x = p.get(x_key)
            if x is None:
                continue
            xs.append(x)
            ys.append(p["net_cost"])
            custom.append(
                [
                    p.get("soh_final"),
                    p.get("wear_cost"),
                    p.get("hover_details", "reward_coeffs: n/a"),
                ]
            )
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name=label,
            marker=dict(color=color, symbol=symbol, size=9),
            customdata=custom,
            hovertemplate=(
                "SOH=%{customdata[0]:.2f}%<br>"
                "wear_cost=%{customdata[1]:.2f} EUR<br>"
                "net_cost=%{y:.2f} EUR<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            ),
        )

    fig = go.Figure()
    rl_pts = [p for p in points if p["source"] == "RL"]
    mpc_pts = [p for p in points if p["source"] == "MPC"]
    rbc_pts = [p for p in points if p["source"] == "RBC"]
    if rl_pts:
        fig.add_trace(make_trace("RL", "#1f77b4", "circle", rl_pts))
    if mpc_pts:
        fig.add_trace(make_trace("MPC", "#d62728", "diamond", mpc_pts))
    if rbc_pts:
        fig.add_trace(make_trace("RBC", "#9467bd", "square", rbc_pts))

    pareto_sorted = sorted(pareto, key=lambda p: p[x_key])
    fig.add_trace(
        go.Scatter(
            x=[p[x_key] for p in pareto_sorted],
            y=[p["net_cost"] for p in pareto_sorted],
            mode="lines",
            name="Pareto front",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Net operating cost (EUR) = import - export",
        template="plotly_white",
    )
    fig.write_html(str(out_path), auto_open=auto_open)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto front for SOH final vs net operating cost."
    )
    parser.add_argument(
        "--rl-root",
        type=Path,
        default=Path("outputs/RL_REWARD_SWEEP_OPSD_1_WEEK/test"),
        help="Root directory with RL test outputs.",
    )
    parser.add_argument(
        "--mpc-root",
        type=Path,
        default=Path("outputs/MPC"),
        help="Root directory with MPC outputs.",
    )
    parser.add_argument(
        "--rbc-root",
        type=Path,
        default=Path("outputs/RBC"),
        help="Root directory with RBC outputs.",
    )
    parser.add_argument(
        "--exclude-runs",
        type=Path,
        default=None,
        help="Optional text file with one run folder name per line to exclude.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits for the wear-cost plot (Battery wear cost vs Operating cost).",
    )
    parser.add_argument(
        "--soh-xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits for the SOH plot (Final SOH vs Operating cost).",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits.",
    )
    parser.add_argument(
        "--xscale",
        choices=("linear", "log"),
        default="linear",
        help="X-axis scale.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale.",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Do not force axes to start at 0; use Matplotlib autoscaling.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=12.0,
        help="Legend font size (Matplotlib).",
    )
    parser.add_argument(
        "--legend-markerscale",
        type=float,
        default=1.0,
        help="Legend marker scale (Matplotlib).",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=14.0,
        help="Axis label font size (Matplotlib).",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=16.0,
        help="Plot title font size (Matplotlib).",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=float,
        default=11.0,
        help="Tick label font size (Matplotlib).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/pareto_soh_costs.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster DPI for saving PNG/JPG outputs (ignored for vector formats like PDF/SVG).",
    )
    parser.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also save a vector PDF with the same stem as --out.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with run names.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive HTML plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )
    args = parser.parse_args()

    rl_reports = list(args.rl_root.rglob("final_report.txt"))
    mpc_reports = list(args.mpc_root.rglob("final_report.txt"))
    rbc_reports = list(args.rbc_root.rglob("final_report.txt"))
    excluded_runs = set()
    if args.exclude_runs:
        try:
            excluded_runs = {
                line.strip()
                for line in args.exclude_runs.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        except OSError:
            excluded_runs = set()

    points = []
    for report in rl_reports + mpc_reports + rbc_reports:
        parsed = _parse_final_report(report)
        if not parsed:
            continue
        run_name = report.parent.name
        if excluded_runs and run_name in excluded_runs:
            continue
        net_cost = parsed["import_cost"] - parsed["export_revenue"]
        source = "MPC" if report in mpc_reports else ("RBC" if report in rbc_reports else "RL")
        if source == "MPC":
            hover_details = _format_mpc_wear_cost_scale(_load_mpc_wear_cost_scale(report.parent))
        elif source == "RL":
            hover_details = _format_reward_coeffs(_load_reward_coeffs(report.parent))
            hover_details += f"<br>folder: {run_name}"
        else:
            hover_details = "reward_coeffs: n/a"
        points.append(
            {
                "run_name": run_name,
                "source": source,
                "soh_final": parsed["soh_final"],
                "net_cost": net_cost,
                "wear_cost": parsed.get("wear_cost"),
                "hover_details": hover_details,
            }
        )

    if not points:
        raise SystemExit("No valid final_report.txt found to plot.")

    rl_points = [p for p in points if p["source"] == "RL"]
    mpc_points = [p for p in points if p["source"] == "MPC"]
    rbc_points = [p for p in points if p["source"] == "RBC"]
    pareto = _compute_pareto(points)

    fig, ax = plt.subplots(figsize=(9, 6))
    if rl_points:
        ax.scatter(
            [p["soh_final"] for p in rl_points],
            [p["net_cost"] for p in rl_points],
            label="RL",
            s=45,
            alpha=0.85,
            color="#1f77b4",
        )
    if mpc_points:
        ax.scatter(
            [p["soh_final"] for p in mpc_points],
            [p["net_cost"] for p in mpc_points],
            label="MPC",
            s=60,
            alpha=0.9,
            color="#d62728",
            marker="D",
        )
    if rbc_points:
        ax.scatter(
            [p["soh_final"] for p in rbc_points],
            [p["net_cost"] for p in rbc_points],
            label="RBC",
            s=60,
            alpha=0.9,
            color="#9467bd",
            marker="s",
        )

    pareto_sorted = sorted(pareto, key=lambda p: p["soh_final"])
    ax.plot(
        [p["soh_final"] for p in pareto_sorted],
        [p["net_cost"] for p in pareto_sorted],
        color="#2ca02c",
        linewidth=2,
        label="Pareto front",
    )

    if args.annotate:
        for p in points:
            ax.annotate(
                p["run_name"],
                (p["soh_final"], p["net_cost"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

    ax.set_xlabel("Final SOH (%)", fontsize=args.label_fontsize)
    ax.set_ylabel("Net operating cost (EUR) = import - export", fontsize=args.label_fontsize)
    ax.set_title("Pareto: Final SOH vs Operating Cost", fontsize=args.title_fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=args.legend_fontsize, markerscale=args.legend_markerscale)
    ax.tick_params(axis="both", labelsize=args.tick_fontsize)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    if args.soh_xlim:
        ax.set_xlim(args.soh_xlim[0], args.soh_xlim[1])
    if args.ylim:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    if not args.tight and not args.ylim:
        ax.set_ylim(bottom=0)

    wear_points = [p for p in points if p.get("wear_cost") is not None]
    fig_w, ax_w = plt.subplots(figsize=(9, 6))
    if wear_points:
        rl_wear = [p for p in wear_points if p["source"] == "RL"]
        mpc_wear = [p for p in wear_points if p["source"] == "MPC"]
        rbc_wear = [p for p in wear_points if p["source"] == "RBC"]
        if rl_wear:
            ax_w.scatter(
                [p["wear_cost"] for p in rl_wear],
                [p["net_cost"] for p in rl_wear],
                label="RL",
                s=45,
                alpha=0.85,
                color="#1f77b4",
            )
        if mpc_wear:
            ax_w.scatter(
                [p["wear_cost"] for p in mpc_wear],
                [p["net_cost"] for p in mpc_wear],
                label="MPC",
                s=60,
                alpha=0.9,
                color="#d62728",
                marker="D",
            )
        if rbc_wear:
            ax_w.scatter(
                [p["wear_cost"] for p in rbc_wear],
                [p["net_cost"] for p in rbc_wear],
                label="RBC",
                s=60,
                alpha=0.9,
                color="#9467bd",
                marker="s",
            )
        if args.annotate:
            for p in wear_points:
                ax_w.annotate(
                    p["run_name"],
                    (p["wear_cost"], p["net_cost"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                )
        wear_pareto = _compute_pareto_wear(wear_points)
        wear_pareto_sorted = sorted(wear_pareto, key=lambda p: p["wear_cost"])
        ax_w.plot(
            [p["wear_cost"] for p in wear_pareto_sorted],
            [p["net_cost"] for p in wear_pareto_sorted],
            color="#2ca02c",
            linewidth=2,
            label="Pareto front",
        )
        ax_w.set_xlabel("Battery wear cost (EUR)", fontsize=args.label_fontsize)
        ax_w.set_ylabel("Net operating cost (EUR) = import - export", fontsize=args.label_fontsize)
        ax_w.set_title("Costs: Battery Wear vs Operating Cost", fontsize=args.title_fontsize)
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=args.legend_fontsize, markerscale=args.legend_markerscale)
        ax_w.tick_params(axis="both", labelsize=args.tick_fontsize)
        ax_w.set_xscale(args.xscale)
        ax_w.set_yscale(args.yscale)
        if args.xlim:
            ax_w.set_xlim(args.xlim[0], args.xlim[1])
        if args.ylim:
            ax_w.set_ylim(args.ylim[0], args.ylim[1])
        if not args.tight and not args.ylim:
            ax_w.set_ylim(bottom=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    wear_out = args.out.with_name(f"{args.out.stem}_wear_vs_operating{args.out.suffix}")
    fig_w.tight_layout()
    fig_w.savefig(wear_out, dpi=args.dpi, bbox_inches="tight")
    if args.also_pdf and args.out.suffix.lower() != ".pdf":
        pdf_out = args.out.with_suffix(".pdf")
        pdf_wear_out = wear_out.with_suffix(".pdf")
        fig.savefig(pdf_out, bbox_inches="tight")
        fig_w.savefig(pdf_wear_out, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    print(f"Saved plot to {args.out}")
    print(f"Saved plot to {wear_out}")
    if args.also_pdf and args.out.suffix.lower() != ".pdf":
        print(f"Saved plot to {pdf_out}")
        print(f"Saved plot to {pdf_wear_out}")

    if not args.no_interactive:
        interactive_out = args.out.with_name(f"{args.out.stem}_interactive.html")
        wear_interactive_out = args.out.with_name(f"{args.out.stem}_wear_vs_operating_interactive.html")
        _plot_interactive(
            points,
            pareto,
            interactive_out,
            x_key="soh_final",
            x_label="Final SOH (%)",
            title="Pareto: Final SOH vs Operating Cost",
            auto_open=True,
        )
        _plot_interactive(
            wear_points,
            wear_pareto,
            wear_interactive_out,
            x_key="wear_cost",
            x_label="Battery wear cost (EUR)",
            title="Costs: Battery Wear vs Operating Cost",
            auto_open=True,
        )
        print(f"Saved plot to {interactive_out}")
        print(f"Saved plot to {wear_interactive_out}")


if __name__ == "__main__":
    main()
