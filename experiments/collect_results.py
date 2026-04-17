"""Collect results from all experiment directories and print a summary table.

Reads {out_dir}/{seed}/{split}/stats.json, picks the best epoch by val MAE,
and reports test MAE (mean +/- std across seeds).
"""
import json
import os
import sys

import numpy as np


def _is_pcqm4m_subset_outbase(base: str) -> bool:
    b = base.replace("\\", "/")
    return "results_pcqm4m_subset" in b or os.path.isdir(
        os.path.join(base, "exp1_lhks_K32")
    )


def experiment_dirs():
    """Paths under EXPERIMENT_OUTBASE (set by Slurm scripts) or default layout."""
    base = os.environ.get("EXPERIMENT_OUTBASE", "").strip()
    if not base:
        base = (
            "results/slurm1_2seed"
            if os.path.isdir("results/slurm1_2seed")
            else "results"
        )
    # PCQM4Mv2-Subset launcher (run/pcqm4m_subset_run_slurm.sh) layout
    if _is_pcqm4m_subset_outbase(base):
        return [
            ("PCQM subset L-HKS K=32", os.path.join(base, "exp1_lhks_K32")),
            ("PCQM subset L-HKS K=16", os.path.join(base, "exp2_lhks_K16")),
            ("PCQM subset L-HKS K=4", os.path.join(base, "exp3_lhks_K4")),
            ("PCQM subset L-HKS K=5", os.path.join(base, "exp3b_lhks_K5")),
            ("PCQM subset L-HKS K=32 fixed", os.path.join(base, "exp4_lhks_K32_fixed")),
            ("PCQM subset HKdiagSE", os.path.join(base, "exp5_hkdiag")),
        ]
    return [
        ("GPS+RWSE", os.path.join(base, "exp0_rwse")),
        ("GPS+L-HKS (learned)", os.path.join(base, "exp1_lhks")),
        ("GPS+L-HKS (fixed)", os.path.join(base, "exp4a_fixed")),
        ("GPS (no PE)", os.path.join(base, "exp0b_noPE")),
        ("K=4", os.path.join(base, "exp4b_K4")),
        ("K=8", os.path.join(base, "exp4b_K8")),
        ("K=32", os.path.join(base, "exp4b_K32")),
    ]


def read_stats(stats_path):
    """Read a stats.json file (one JSON object per line)."""
    entries = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_best_test_mae(run_dir):
    """Given a single run directory (containing train/val/test subdirs),
    find the best epoch by val MAE and return the corresponding test MAE.
    """
    val_path = os.path.join(run_dir, "val", "stats.json")
    test_path = os.path.join(run_dir, "test", "stats.json")
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        return None

    val_stats = read_stats(val_path)
    test_stats = read_stats(test_path)

    if not val_stats or not test_stats:
        return None

    best_epoch = min(val_stats, key=lambda x: x.get("mae", float("inf")))["epoch"]
    test_at_best = [s for s in test_stats if s["epoch"] == best_epoch]
    if not test_at_best:
        return None
    return test_at_best[0]["mae"]


def collect_experiment(exp_dir):
    """Collect test MAEs across all seeds in an experiment directory."""
    maes = []
    if not os.path.isdir(exp_dir):
        return maes

    for entry in sorted(os.listdir(exp_dir)):
        run_dir = os.path.join(exp_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        # Support both seed{N} dirs (from run scripts) and plain numeric dirs
        inner = run_dir
        subdirs = os.listdir(inner)
        if "val" not in subdirs:
            # Look one level deeper (e.g. seed0/0/val/)
            for sub in sorted(subdirs):
                sub_path = os.path.join(inner, sub)
                if os.path.isdir(sub_path) and os.path.exists(
                    os.path.join(sub_path, "val", "stats.json")
                ):
                    mae = get_best_test_mae(sub_path)
                    if mae is not None:
                        maes.append(mae)
        else:
            mae = get_best_test_mae(inner)
            if mae is not None:
                maes.append(mae)
    return maes


def main():
    summary = {}
    rows = []

    for name, exp_dir in experiment_dirs():
        maes = collect_experiment(exp_dir)
        if len(maes) == 0:
            rows.append((name, None, None, 0))
            continue
        mean = np.mean(maes)
        std = np.std(maes) if len(maes) > 1 else 0.0
        rows.append((name, mean, std, len(maes)))
        summary[name] = {
            "mae_mean": round(float(mean), 5),
            "mae_std": round(float(std), 5),
            "seeds": len(maes),
            "maes": [round(float(m), 5) for m in maes],
        }

    # Print table
    print(f"{'Method':<25} | {'MAE':>15} | {'Seeds':>5}")
    print("-" * 25 + "-+-" + "-" * 15 + "-+-" + "-" * 5)
    for name, mean, std, n_seeds in rows:
        if mean is None:
            print(f"{name:<25} | {'(not found)':>15} | {0:>5}")
        elif n_seeds > 1:
            print(f"{name:<25} | {mean:.3f} +/- {std:.3f} | {n_seeds:>5}")
        else:
            print(f"{name:<25} | {mean:.3f}         | {n_seeds:>5}")

    base = os.environ.get("EXPERIMENT_OUTBASE", "").strip()
    if base and _is_pcqm4m_subset_outbase(base):
        out_summary = os.path.join(base, "summary.json")
    else:
        os.makedirs("results", exist_ok=True)
        out_summary = "results/summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_summary}")


if __name__ == "__main__":
    main()
