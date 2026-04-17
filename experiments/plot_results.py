"""Generate all figures from experiment results.

Figures saved to figures/ directory:
  figure1_stability.pdf  — prediction variance vs perturbation epsilon
  figure2_gap_scatter.pdf — variance vs eigenvalue gap at epsilon=0.05
  figure3_learned_times.pdf — learned diffusion times across seeds
"""
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_stability():
    """Figure 1: prediction variance vs epsilon, one line per PE type."""
    path = "results/stability_results.pt"
    if not os.path.exists(path):
        print(f"[skip] {path} not found — run experiments/run_stability.py first")
        return

    data = torch.load(path, map_location='cpu')

    fig, ax = plt.subplots()
    colors = {'LHKS': '#2196F3', 'RWSE': '#FF5722'}

    for method, eps_dict in data.items():
        epsilons = sorted(eps_dict.keys())
        means, stds = [], []
        for eps in epsilons:
            variances = [r['variance'] for r in eps_dict[eps]]
            means.append(np.mean(variances))
            stds.append(np.std(variances))
        means, stds = np.array(means), np.array(stds)
        c = colors.get(method, '#666666')
        ax.plot(epsilons, means, 'o-', label=method, color=c, linewidth=2)
        ax.fill_between(epsilons, means - stds, means + stds, alpha=0.15, color=c)

    ax.set_yscale('log')
    ax.set_xlabel('Perturbation ε')
    ax.set_ylabel('Mean prediction variance (log)')
    ax.set_title('Stability under edge perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, "figure1_stability.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_gap_scatter():
    """Figure 2: scatter of variance vs min eigenvalue gap at eps=0.05."""
    path = "results/stability_results.pt"
    if not os.path.exists(path):
        print(f"[skip] {path} not found — run experiments/run_stability.py first")
        return

    data = torch.load(path, map_location='cpu')
    eps_key = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colors = {'RWSE': '#FF5722', 'LHKS': '#2196F3'}

    for ax, method in zip(axes, ['RWSE', 'LHKS']):
        if method not in data or eps_key not in data[method]:
            ax.set_title(f'{method} (no data)')
            continue
        entries = data[method][eps_key]
        gaps = [r['eigval_gap'] for r in entries]
        variances = [r['variance'] for r in entries]
        c = colors.get(method, '#666666')
        ax.scatter(gaps, variances, alpha=0.4, s=12, color=c)
        ax.set_xlabel('Min eigenvalue gap')
        ax.set_title(method)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Prediction variance (log)')
    fig.suptitle(f'Stability vs spectral gap (ε = {eps_key})', y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "figure2_gap_scatter.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _experiment_outbase():
    b = os.environ.get("EXPERIMENT_OUTBASE", "").strip()
    if b:
        return b
    return (
        "results/slurm1_2seed"
        if os.path.isdir("results/slurm1_2seed")
        else "results"
    )


def plot_learned_times():
    """Figure 3: learned diffusion times across seeds."""
    base = _experiment_outbase()
    ckpt_patterns = [
        os.path.join(base, "exp1_lhks", "seed*", "*", "ckpt", "*.ckpt"),
        os.path.join(base, "exp1_lhks", "seed*", "ckpt", "*.ckpt"),
    ]

    ckpt_files = []
    for pat in ckpt_patterns:
        ckpt_files.extend(glob.glob(pat))

    if not ckpt_files:
        print("[skip] No LHKS checkpoints found for figure3")
        return

    import math
    K = None
    init_times = None

    fig, ax = plt.subplots(figsize=(8, 3))
    seed_times = []

    for ckpt_path in sorted(ckpt_files):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state', ckpt)
        log_times_key = None
        for k in state:
            if 'log_times' in k:
                log_times_key = k
                break
        if log_times_key is None:
            continue

        log_t = state[log_times_key]
        times = torch.exp(log_t).numpy()
        seed_times.append(times)

        if K is None:
            K = len(times)
            init_times = np.exp(np.linspace(math.log(0.01), math.log(100.0), K))

    if not seed_times:
        print("[skip] No log_times found in checkpoints")
        plt.close(fig)
        return

    # Plot initialization as gray lines
    for t in init_times:
        ax.axvline(t, color='gray', alpha=0.2, linewidth=0.8)

    # Plot learned times for each seed
    for i, times in enumerate(seed_times):
        y = np.full_like(times, i)
        ax.scatter(times, y, s=40, zorder=5, label=f'Seed {i}' if i < 6 else None)

    ax.set_xscale('log')
    ax.set_xlabel('Diffusion time t (log scale)')
    ax.set_ylabel('Seed')
    ax.set_yticks(range(len(seed_times)))
    ax.set_title('Learned diffusion times across seeds')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    out = os.path.join(FIGURES_DIR, "figure3_learned_times.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    plot_stability()
    plot_gap_scatter()
    plot_learned_times()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
