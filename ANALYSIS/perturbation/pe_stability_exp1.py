"""Experiment 1 - Non-bridge edge removal, stratified by δ_min.

For each of 6 methods (LapPE, SignNet-MLP, SignNet-DeepSets, L-HKS,
fix-L-HKS, noPE) on PCQM4Mv2, remove k ∈ {1,2,3} non-bridge edges uniformly at
random, recompute the PE for the perturbed graph (noPE has no PE to recompute,
so it isolates pure MPNN sensitivity to topology edits), and measure how much
the model's scalar prediction varies across 20 independent random perturbations.

Results are stratified by each molecule's combinatorial-Laplacian eigenvalue
gap δ_min (computed at k=8 so the x-axis is comparable across methods) to
show that instability of eigenvector-based PEs grows as δ_min shrinks,
while HKS stays flat.

Usage:
    python ANALYSIS/perturbation/pe_stability_exp1.py --method LapPE --run-dirs results_pcqm4m_subset/stability_baselines/lappe/seed1
    python ANALYSIS/perturbation/pe_stability_exp1.py --all

Outputs (ANALYSIS/perturbation/exp1_results/):
    <method>__<seed_tag>.json   raw per-graph std per k_remove
    exp1_main.png               mean std by δ_min bin, per method
    exp1_summary.json           aggregate table
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from numpy.random import default_rng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


# ------------------------------ config ------------------------------------
METHOD_RUN_DIRS: Dict[str, List[str]] = {
    "LapPE":            ["results_pcqm4m_subset/stability_baselines/lappe/seed1"],
    "SignNet-MLP":      ["results_pcqm4m_subset/stability_baselines/snmlp/seed1"],
    "SignNet-DeepSets": ["results_pcqm4m_subset/stability_baselines/snds/seed1"],
    "L-HKS":            ["results_pcqm4m_subset/mlp_ablation/mlp3/seed2"],
    "fix-L-HKS":        ["results_pcqm4m_subset/mlp_ablation/mlp3_fixed/seed5"],
    "noPE":             ["results_pcqm4m_subset/stability_baselines/nope/seed1"],
}
K_REMOVE_LEVELS = (1, 2, 3)
N_PERTURBATIONS = 20
N_TARGET_GRAPHS = 5000
BATCH_SIZE      = 64
OUT_DIR         = Path("ANALYSIS/perturbation/exp1_results")


# -------------------------- per-method experiment -------------------------

def run_one_checkpoint(method: str, run_dir: str, test_graphs: C.PCQMGraphs,
                       sample_ids: np.ndarray, sample_dmin: np.ndarray,
                       device: torch.device, rng_seed: int = 0) -> dict:
    print(f"\n[exp1] === {method} :: {run_dir} ===")
    model, cfg, pe_type = C.load_model_and_cfg(run_dir, device)
    print(f"[exp1] loaded model (pe_type={pe_type})")

    rng = default_rng(rng_seed)
    rows: List[dict] = []
    t0 = time.time()
    for gi_pos, gi in enumerate(sample_ids.tolist()):
        base = test_graphs.get(gi)
        nb   = C.find_non_bridge_edges(base.edge_index, base.num_nodes)
        if not nb:                       # pure trees (skipped upstream anyway)
            continue
        per_k = {}
        for k_rm in K_REMOVE_LEVELS:
            if len(nb) < k_rm:
                per_k[k_rm] = {"n_preds": 0, "std": float("nan")}
                continue
            replicas = []
            for _ in range(N_PERTURBATIONS):
                picks_idx = rng.choice(len(nb), size=k_rm, replace=False)
                picks     = [nb[i] for i in picks_idx]
                d = C.remove_edges(base, picks)
                d = C.recompute_pe(d, cfg, pe_type)
                replicas.append(d)
            preds = C.batched_forward(model, replicas, device, BATCH_SIZE)
            per_k[k_rm] = {"n_preds": int(preds.size),
                           "std":     float(preds.std()),
                           "mean":    float(preds.mean())}
        rows.append({"gi": int(gi),
                     "num_nodes": int(base.num_nodes),
                     "num_nonbridges": int(len(nb)),
                     "delta_min_k8":   float(sample_dmin[gi_pos]),
                     "bin":            C.deltamin_bin(float(sample_dmin[gi_pos])),
                     "per_k": per_k})
        if (gi_pos + 1) % 100 == 0 or gi_pos + 1 == len(sample_ids):
            dt = time.time() - t0
            print(f"[exp1] {gi_pos + 1:5d}/{len(sample_ids)} "
                  f"({dt:6.1f}s, {dt / (gi_pos + 1):5.2f}s/graph)")

    seed_tag = Path(run_dir).name
    out_path = OUT_DIR / f"{method}__{seed_tag}.json"
    C.ensure_output_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump({"method": method, "run_dir": run_dir, "pe_type": pe_type,
                   "rows": rows}, f)
    print(f"[exp1] wrote {out_path} ({len(rows)} graphs)")
    return {"method": method, "run_dir": run_dir, "rows": rows}


# ------------------------------- plotting ---------------------------------

def _aggregate(results_per_method: Dict[str, List[dict]]) -> dict:
    """Average per-graph std across seeds, then mean per (method, bin, k_rm)."""
    agg: Dict[str, dict] = {}
    for method, seeds in results_per_method.items():
        by_gi: Dict[int, dict] = {}       # gi -> {"bin": str, "kr": [stds]}
        for seed in seeds:
            for r in seed["rows"]:
                rec = by_gi.setdefault(r["gi"], {"bin": r["bin"], "kr": {}})
                for k, v in r["per_k"].items():
                    rec["kr"].setdefault(int(k), []).append(v["std"])
        # Average over seeds for each graph; then group by (bin, k).
        per_bin_per_k: Dict[str, Dict[int, list]] = {b: {k: [] for k in K_REMOVE_LEVELS}
                                                     for b in "ABCD"}
        for gi, rec in by_gi.items():
            b = rec["bin"]
            if b not in per_bin_per_k:       # skip N (undefined)
                continue
            for k, stds in rec["kr"].items():
                s = np.nanmean(stds)
                if np.isfinite(s):
                    per_bin_per_k[b][k].append(float(s))
        agg[method] = {b: {k: (float(np.mean(v)) if v else float("nan"),
                                int(len(v)))
                            for k, v in kr.items()}
                       for b, kr in per_bin_per_k.items()}
    return agg


def plot_main(agg: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    bins = ["A", "B", "C", "D"]
    bin_labels = [r"$<10^{-10}$", r"$<0.05$", r"$<0.15$", r"$\geq 0.15$"]
    fig, axes = plt.subplots(1, len(K_REMOVE_LEVELS), figsize=(4.2 * len(K_REMOVE_LEVELS),
                                                                3.8),
                             sharey=True)
    colors = {"LapPE": "#d62728", "SignNet-MLP": "#ff7f0e",
              "SignNet-DeepSets": "#9467bd", "L-HKS": "#1f77b4",
              "fix-L-HKS": "#17becf", "noPE": "#2ca02c"}
    markers = {"LapPE": "o", "SignNet-MLP": "s", "SignNet-DeepSets": "D",
               "L-HKS": "^", "fix-L-HKS": "v", "noPE": "x"}
    for ax, k in zip(axes, K_REMOVE_LEVELS):
        for method, per_bin in agg.items():
            ys = [per_bin[b].get(k, (float("nan"), 0))[0] for b in bins]
            ns = [per_bin[b].get(k, (float("nan"), 0))[1] for b in bins]
            ys_plot = [y if np.isfinite(y) and y > 0 else np.nan for y in ys]
            ax.plot(bins, ys_plot, marker=markers.get(method, "o"),
                    color=colors.get(method, None), label=f"{method}")
            for x, y, n in zip(bins, ys, ns):
                if not np.isfinite(y) or y <= 0: continue
                ax.annotate(f"n={n}", (x, y), fontsize=6, alpha=0.6,
                            xytext=(0, 4), textcoords="offset points", ha="center")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\delta_{\min}$ bin")
        ax.set_xticks(range(4)); ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_title(f"remove {k} non-bridge edge(s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean prediction std (over 20 perturbations)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Exp 1 - prediction instability vs. eigenvalue gap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[exp1] wrote {out_path}")


# ---------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHOD_RUN_DIRS.keys()))
    ap.add_argument("--all", action="store_true",
                    help="Run every method listed in METHOD_RUN_DIRS")
    ap.add_argument("--run-dirs", nargs="*", default=None,
                    help="Override run_dirs for --method (one per seed)")
    ap.add_argument("--n-graphs", type=int, default=N_TARGET_GRAPHS)
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip experiment, only redo aggregation + plot from cache")
    ap.add_argument("--no-plot", action="store_true",
                    help="Run experiment(s) only; skip aggregation/plot (use --plot-only after)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not (args.method or args.all or args.plot_only):
        ap.error("specify --method, --all, or --plot-only")
    if args.plot_only and args.no_plot:
        ap.error("cannot combine --plot-only and --no-plot")

    C.ensure_output_dir(OUT_DIR)
    device = torch.device(args.device)

    if not args.plot_only:
        test_graphs = C.PCQMGraphs.load()
        sample_ids, sample_dmin, counts = C.stratified_subsample(
            test_graphs, n_target=args.n_graphs, bin_k=8,
            seed=0, skip_trees=True, skip_tiny=True)

        methods = list(METHOD_RUN_DIRS) if args.all else [args.method]
        for m in methods:
            dirs = args.run_dirs if (args.method == m and args.run_dirs) \
                   else METHOD_RUN_DIRS[m]
            for rd in dirs:
                run_one_checkpoint(m, rd, test_graphs, sample_ids, sample_dmin,
                                   device, rng_seed=hash((m, rd)) & 0xFFFF)

    # --- aggregation / plot ---
    if args.no_plot:
        print("[exp1] --no-plot: skipping aggregation and figures.")
        return
    results_per_method: Dict[str, List[dict]] = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name in ("exp1_summary.json",): continue
        with open(path) as f:
            payload = json.load(f)
        results_per_method.setdefault(payload["method"], []).append(payload)
    if not results_per_method:
        print("[exp1] no result json files yet."); return
    agg = _aggregate(results_per_method)
    with open(OUT_DIR / "exp1_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[exp1] wrote {OUT_DIR / 'exp1_summary.json'}")
    plot_main(agg, OUT_DIR / "exp1_main.png")


if __name__ == "__main__":
    main()
