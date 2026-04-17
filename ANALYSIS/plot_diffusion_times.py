"""Fetch diffusion time trajectories from W&B and plot t_j vs epoch per run.

Uses metrics logged as train/diffusion_time/t{j} (the GraphGym stats logger
prepends "train/"; see graphgps/train/custom_train.py where diff_stats are
written under keys "diffusion_time/t{j}"). Per-t_j initial values match the
log-spaced init from graphgps/encoder/lhks_encoder.py:
    log_times = linspace(log(0.01), log(100.0), K)  ->  t_j = exp(log_times_j)

Usage:
    # Default (ZINC projects):
    python ANALYSIS/plot_diffusion_times.py

    # MLP3 runs in PCQM4Mv2-subset ablation:
    python ANALYSIS/plot_diffusion_times.py \
        --project pcqm4m-subset-lhks-mlp-ablation \
        --filter MLP3 --out-subdir mlp3

Plots saved to: ANALYSIS/diffusion_time_plots[/<out-subdir>]/
"""
import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb

ENTITY  = "znidar-mark-stanford-university"
DEFAULT_PROJECTS = ["lhks-pe-2seed", "lhks-pe"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "diffusion_time_plots")

K = 16  # default K for legend subsampling fallback
TIME_KEYS = [f"train/diffusion_time/t{j}" for j in range(K)]
INIT_TIMES = np.exp(np.linspace(math.log(0.01), math.log(100.0), K))


def init_times_for_k(n: int) -> np.ndarray:
    """Match lhks_encoder: log-spaced in [0.01, 100] for n diffusion times."""
    if n < 1:
        return np.array([])
    return np.exp(np.linspace(math.log(0.01), math.log(100.0), n))

api = wandb.Api(timeout=60)


def fetch_runs(projects, name_filter=None):
    runs = []
    for project in projects:
        try:
            project_runs = list(api.runs(f"{ENTITY}/{project}"))
            kept = 0
            for r in project_runs:
                name = r.name or ""
                if name_filter and name_filter not in name:
                    continue
                if "CRASHED" in name.upper():
                    continue
                runs.append((project, r))
                kept += 1
            if name_filter:
                print(f"Found {len(project_runs)} runs in {project} "
                      f"({kept} match filter '{name_filter}')")
            else:
                print(f"Found {len(project_runs)} runs in {project}")
        except Exception as e:
            print(f"Could not access {project}: {e}")
    return runs


def get_diffusion_history(run):
    """Return dict: {key -> np.array of values} and epochs array."""
    # Check summary to find which t keys actually exist (handles variable K)
    summary_keys = list(run.summary.keys()) if hasattr(run.summary, 'keys') else []
    present_keys = [k for k in summary_keys if k.startswith("train/diffusion_time/t")
                    and k.split("/t")[-1].isdigit()]
    if not present_keys:
        return None, None

    keys_to_fetch = present_keys + ["_step"]
    try:
        history = run.history(keys=keys_to_fetch, samples=10000, pandas=True,
                              x_axis="_step")
    except Exception as e:
        print(f"  [warn] history() error: {e}")
        return None, None

    if history is None or history.empty:
        return None, None

    present_in_df = [k for k in present_keys if k in history.columns]
    if not present_in_df:
        return None, None

    history = history.dropna(subset=present_in_df, how='all')
    if history.empty:
        return None, None

    epochs = history["_step"].values if "_step" in history.columns else np.arange(len(history))
    data = {k: history[k].values for k in present_in_df}
    return epochs, data


def last_logged_diffusion_times(epochs, data, sorted_keys):
    """Linear-scale t_j at the latest W&B step with complete t_* values."""
    order = np.argsort(epochs)
    last_idx = None
    for idx in reversed(order):
        if all(not np.isnan(data[k][idx]) for k in sorted_keys):
            last_idx = int(idx)
            break
    if last_idx is None:
        return None, None
    step_val = float(epochs[last_idx])
    times = [float(data[k][last_idx]) for k in sorted_keys]
    return step_val, times


def format_last_epoch_block(sorted_keys, times_at_last, step_val):
    """Multi-line monospace block: t_j = value (linear scale)."""
    lines = [f"Last W&B _step = {step_val:.0f}  |  t_j = exp(log_times_j)  (linear scale)"]
    ncol = 4
    n = len(times_at_last)
    rows = (n + ncol - 1) // ncol
    buf = []
    for r in range(rows):
        parts = []
        for c in range(ncol):
            j = r + c * rows
            if j < n:
                key = sorted_keys[j]
                jj = int(key.split("/t")[-1])
                parts.append(f"t{jj:2d}={times_at_last[j]:.6g}")
        buf.append("  ".join(parts))
    lines.extend(buf)
    return "\n".join(lines)


def plot_run(run_name, project, epochs, data, run_id):
    """One plot per run: all K diffusion times vs epoch."""
    # Sort keys by index
    sorted_keys = sorted(data.keys(),
                         key=lambda k: int(k.split("/t")[-1]) if k.split("/t")[-1].isdigit() else 0)
    n = len(sorted_keys)
    max_j = max(int(k.split("/t")[-1]) for k in sorted_keys) if sorted_keys else 0
    init_arr = init_times_for_k(max_j + 1)

    fig_h = 7.2 if "K32" in run_name else 5.0
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colors = cm.plasma(np.linspace(0.05, 0.95, max(n, 1)))

    for ci, key in enumerate(sorted_keys):
        j = int(key.split("/t")[-1])
        vals = data[key]
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            continue
        ax.plot(epochs[mask], vals[mask], color=colors[ci],
                linewidth=1.2, alpha=0.85, label=f"t{j}")

    # Init reference lines (encoder: linspace in log-t from 0.01 to 100)
    for ci, key in enumerate(sorted_keys):
        j = int(key.split("/t")[-1])
        if j < len(init_arr):
            ax.axhline(init_arr[j], color=colors[ci], linewidth=0.5,
                       linestyle='--', alpha=0.3)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diffusion time t (log scale; logged values are linear t)')
    ax.set_title(f'{run_name}\n[{project}]', fontsize=10)
    ax.grid(True, alpha=0.25, which='both')

    # Legend subsample for many curves
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        step_leg = max(1, len(handles) // 8)
        idx = list(range(0, len(handles), step_leg))
        ax.legend([handles[i] for i in idx if i < len(handles)],
                  [labels[i] for i in idx if i < len(labels)],
                  fontsize=7, loc='upper left', ncol=2, title='time index')
    else:
        ax.legend(fontsize=7, loc='upper left', ncol=2, title='time index')

    step_val, times_last = last_logged_diffusion_times(epochs, data, sorted_keys)
    if step_val is not None and times_last is not None and "K32" in run_name:
        block = format_last_epoch_block(sorted_keys, times_last, step_val)
        fig.tight_layout(rect=(0, 0.22, 1, 1))
        fig.text(0.02, 0.02, block, fontsize=6.5, family='monospace',
                 va='bottom', ha='left', transform=fig.transFigure,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.88))
        txt_path = os.path.join(OUT_DIR, "LHKS-K32_last_epoch_diffusion_times_linear.txt")
        with open(txt_path, "w") as f:
            f.write(f"# W&B _step (last complete row): {step_val}\n")
            f.write("# t_j in linear scale (as logged)\n")
            for key, tv in zip(sorted_keys, times_last):
                jj = int(key.split("/t")[-1])
                f.write(f"t{jj}\t{tv:.12g}\n")
        print(f"  Wrote {txt_path}")
    else:
        fig.tight_layout()

    safe_name = run_name.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(OUT_DIR, f"{safe_name}_{run_id[:6]}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_all_runs_overlay(all_run_data):
    """One subplot per t_j, all runs overlaid."""
    if not all_run_data:
        return

    # Find max t index across all runs
    max_j = 0
    for _, _, data in all_run_data:
        for k in data:
            idx = k.split("/t")[-1]
            if idx.isdigit():
                max_j = max(max_j, int(idx))
    n_times = max_j + 1
    cols = min(n_times, 8)
    rows = math.ceil(n_times / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), sharey=False)
    axes = np.array(axes).flatten()

    run_colors = cm.tab10(np.linspace(0, 1, len(all_run_data)))
    inits = init_times_for_k(n_times)

    for j in range(n_times):
        ax = axes[j]
        key = f"train/diffusion_time/t{j}"
        if j < len(inits):
            ax.axhline(inits[j], color='gray', linestyle='--',
                       linewidth=1, alpha=0.5, label='init')
        for idx, (run_name, epochs, data) in enumerate(all_run_data):
            if key not in data:
                continue
            vals = data[key]
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                continue
            short = run_name.split('.')[-1] if '.' in run_name else run_name
            ax.plot(epochs[mask], vals[mask],
                    color=run_colors[idx], linewidth=1.2,
                    alpha=0.8, label=short)
        ax.set_yscale('log')
        t_init_str = f"{inits[j]:.3f}" if j < len(inits) else "?"
        ax.set_title(f't{j}  (init={t_init_str})', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, which='both')
        if j % cols == 0:
            ax.set_ylabel('t value (log)', fontsize=7)
        if j >= (rows - 1) * cols:
            ax.set_xlabel('Epoch', fontsize=7)

    # Hide unused subplots
    for j in range(n_times, len(axes)):
        axes[j].set_visible(False)

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(all_run_data) + 1, 6),
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Diffusion time evolution per t_j — all runs', fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = os.path.join(OUT_DIR, "all_runs_per_t.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved overview: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", action="append", default=None,
                   help="W&B project (repeatable). Defaults to "
                        f"{DEFAULT_PROJECTS}.")
    p.add_argument("--filter", dest="name_filter", default=None,
                   help="Only keep runs whose name contains this substring "
                        "(e.g. 'MLP3').")
    p.add_argument("--out-subdir", default=None,
                   help="Optional subdirectory name under diffusion_time_plots/.")
    return p.parse_args()


def main():
    args = parse_args()
    projects = args.project or DEFAULT_PROJECTS

    out_dir = os.path.join(os.path.dirname(__file__), "diffusion_time_plots")
    if args.out_subdir:
        out_dir = os.path.join(out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    # Expose to helpers that reference module-level OUT_DIR.
    global OUT_DIR
    OUT_DIR = out_dir

    filt_msg = f", filter='{args.name_filter}'" if args.name_filter else ""
    print(f"Fetching runs from W&B ({ENTITY}, projects={projects}{filt_msg})...")
    runs = fetch_runs(projects, args.name_filter)

    if not runs:
        print("No runs found. Check entity/project names and wandb login.")
        return

    all_run_data = []  # (name, epochs, data)

    for project, run in runs:
        name = run.name or run.id
        print(f"\nProcessing: {name} [{project}]")

        epochs, data = get_diffusion_history(run)
        if epochs is None:
            print(f"  [skip] No diffusion_time data found")
            continue

        n_times = len(data)
        print(f"  {n_times} time params, {len(epochs)} epochs logged")

        plot_run(name, project, epochs, data, run.id)
        all_run_data.append((name, epochs, data))

    if all_run_data:
        print(f"\nGenerating overview plot across {len(all_run_data)} runs...")
        plot_all_runs_overlay(all_run_data)
    else:
        print("\nNo runs with diffusion_time data found.")

    print(f"\nDone. Plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
