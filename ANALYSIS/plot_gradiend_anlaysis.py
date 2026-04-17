"""Fetch gradient statistics from W&B and plot vanishing-gradient diagnostics.

Uses metrics logged as grad/norm, grad/min, grad/max (global grad norm and
per-parameter extrema, written in graphgps/train/custom_train.py lines 45-47).
Aligns train/loss from the training stats dict for context.

Usage:
    cd GraphGPS && pixi run python ANALYSIS/plot_gradiend_anlaysis.py

Outputs PNGs under: ANALYSIS/gradient_analysis_plots/   (next to this script)
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb

ENTITY = "znidar-mark-stanford-university"
PROJECTS = ["lhks-pe-2seed", "lhks-pe"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "gradient_analysis_plots")
os.makedirs(OUT_DIR, exist_ok=True)

api = wandb.Api(timeout=120)
SAMPLES = 5000


def fetch_runs():
    runs = []
    for project in PROJECTS:
        try:
            pr = api.runs(f"{ENTITY}/{project}")
            for r in pr:
                runs.append((project, r))
            print(f"Found runs in {project}")
        except Exception as e:
            print(f"Could not access {project}: {e}")
    return runs


def has_grad_metrics(run) -> bool:
    keys = list(run.summary.keys()) if hasattr(run.summary, "keys") else []
    return any(k in keys for k in ("grad/norm", "grad/min", "grad/max"))


def load_grad_history(run) -> pd.DataFrame | None:
    if not has_grad_metrics(run):
        return None
    try:
        g = run.history(
            keys=["grad/norm", "grad/min", "grad/max", "_step"],
            samples=SAMPLES,
            pandas=True,
        )
    except Exception as e:
        print(f"  [warn] grad history: {e}")
        return None
    if g is None or g.empty or "grad/norm" not in g.columns:
        return None
    g = g.sort_values("_step").dropna(subset=["grad/norm"], how="all")
    if g.empty:
        return None

    # Optional: align train loss for context (separate log stream in W&B)
    try:
        l = run.history(keys=["train/loss", "_step"], samples=SAMPLES, pandas=True)
        if l is not None and not l.empty and "train/loss" in l.columns:
            l = l.sort_values("_step")
            g = pd.merge_asof(g, l, on="_step", direction="nearest")
    except Exception:
        pass
    return g


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)[:120]


def vanishing_summary(df: pd.DataFrame) -> str:
    gn = df["grad/norm"].astype(float)
    gm = df["grad/min"].astype(float) if "grad/min" in df.columns else None
    lines = [
        f"grad/norm: min={gn.min():.2e}, median={gn.median():.2e}, max={gn.max():.2e}",
    ]
    if gm is not None:
        frac_tiny = (gm < 1e-6).mean() * 100
        lines.append(
            f"grad/min: min={gm.min():.2e}, median={gm.median():.2e} | "
            f"fraction of steps with grad/min < 1e-6: {frac_tiny:.1f}%"
        )
    return "\n".join(lines)


def plot_run(name: str, project: str, df: pd.DataFrame, run_id: str) -> None:
    step = df["_step"].values
    gn = df["grad/norm"].values.astype(float)
    gm = df["grad/min"].values.astype(float) if "grad/min" in df.columns else None
    gx = df["grad/max"].values.astype(float) if "grad/max" in df.columns else None

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(step, gn, color="#1a237e", linewidth=1.2, label="||g||_2 (epoch)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Global grad norm (log)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"{name}\n[{project}] — gradient norms vs W&B step", fontsize=10)

    if "train/loss" in df.columns and df["train/loss"].notna().any():
        ax2 = axes[0].twinx()
        ax2.plot(step, df["train/loss"], color="#c62828", alpha=0.45, linewidth=1.0, label="train/loss")
        ax2.set_ylabel("train loss", color="#c62828")
        ax2.tick_params(axis="y", labelcolor="#c62828")

    if gm is not None:
        axes[1].plot(step, gm, color="#1565c0", linewidth=1.0, alpha=0.9, label="grad/min")
    if gx is not None:
        axes[1].plot(step, gx, color="#ef6c00", linewidth=1.0, alpha=0.85, label="grad/max")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("W&B step (monotonic training progress)")
    axes[1].set_ylabel("Per-param grad extrema (log)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    summary = vanishing_summary(df)
    fig.text(0.02, 0.01, summary, fontsize=8, family="monospace", va="bottom")

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, f"{safe_filename(name)}_{run_id[:6]}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    print(f"  {summary.replace(chr(10), ' | ')}")


def plot_all_overlay(run_frames: list[tuple[str, str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(run_frames), 1)))
    for i, (name, _proj, df) in enumerate(run_frames):
        label = name.split(".")[-1] if "." in name else name[:40]
        ax.plot(
            df["_step"],
            df["grad/norm"],
            color=cmap[i],
            linewidth=1.1,
            alpha=0.85,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("W&B step")
    ax.set_ylabel("||g||_2 (log)")
    ax.set_title("Global gradient norm — all runs with grad metrics")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "all_runs_grad_norm.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved overview: {out}")


def main():
    print(f"Fetching runs from W&B ({ENTITY})...")
    runs = fetch_runs()
    if not runs:
        print("No runs found.")
        return

    collected: list[tuple[str, str, pd.DataFrame]] = []

    for project, run in runs:
        name = run.name or run.id
        print(f"\n{name} [{project}]")
        df = load_grad_history(run)
        if df is None:
            print("  [skip] No grad history")
            continue
        print(f"  {len(df)} points")
        plot_run(name, project, df, run.id)
        collected.append((name, project, df))

    if collected:
        plot_all_overlay(collected)
    print(f"\nDone. Plots in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
