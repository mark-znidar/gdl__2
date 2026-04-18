"""Experiment 2 - Eigenspace rotation injection (graph unchanged).

Without touching the graph at all, rotate the Laplacian eigenvectors inside
a near-degenerate (or exactly-degenerate) eigenpair (i, i+1), recompute the
PE from this alternative eigenbasis, and measure how much the model's scalar
prediction moves across 20 random rotations.

- Eigenvector-based PEs (LapPE, SignNet*) should be unstable; the instability
  should scale like O(1/Δλ).
- HKS-based PEs should be almost bit-invariant under *exact* degeneracy
  (Pythagorean identity) and drift by only O(t·Δλ) under near-degeneracy.

Usage:
    python ANALYSIS/perturbation/pe_stability_exp2.py --method L-HKS --run-dirs results_pcqm4m_subset/mlp_ablation/mlp3/seed2 --no-plot
    python ANALYSIS/perturbation/pe_stability_exp2.py --all
    python ANALYSIS/perturbation/pe_stability_exp2.py --plot-only

Outputs (ANALYSIS/perturbation/exp2_results/):
    <method>__<seed_tag>.json   raw per-graph std
    exp2_bar.png                mean std on exact-degeneracy subset (< 1e-10)
    exp2_scatter.png            per-graph std vs pair-gap, log x
    exp2_summary.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.random import default_rng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


METHOD_RUN_DIRS: Dict[str, List[str]] = {
    "LapPE":            ["results_pcqm4m_subset/stability_baselines/lappe/seed1"],
    "SignNet-MLP":      ["results_pcqm4m_subset/stability_baselines/snmlp/seed1"],
    "SignNet-DeepSets": ["results_pcqm4m_subset/stability_baselines/snds/seed1"],
    "L-HKS":            ["results_pcqm4m_subset/mlp_ablation/mlp3/seed2"],
    "fix-L-HKS":        ["results_pcqm4m_subset/mlp_ablation/mlp3_fixed/seed5"],
}
N_ROTATIONS      = 20
N_TARGET_GRAPHS  = 5000
BATCH_SIZE       = 64
EIGENSPACE_K     = 21           # search for degeneracies in λ_1..λ_{k-1}
GAP_EXACT        = 1e-10
GAP_NEARDEG      = 0.05
OUT_DIR          = Path("ANALYSIS/perturbation/exp2_results")


# ------------------- candidate graph selection ---------------------------

def select_candidates(test_graphs: C.PCQMGraphs, n_target: int, seed: int = 0) \
        -> List[Tuple[int, int, float, np.ndarray, np.ndarray]]:
    """Return list of (gi, pair_idx, pair_gap, evals, evecs).  We stratify so
    the returned list is half exact-degeneracy (gap<1e-10) and half
    near-degeneracy (1e-10 ≤ gap < 0.05).  Evals/evecs are cached so we don't
    redo the eigendecomposition later."""
    rng = default_rng(seed)
    exact: List[Tuple[int, int, float, np.ndarray, np.ndarray]] = []
    near:  List[Tuple[int, int, float, np.ndarray, np.ndarray]] = []
    t0 = time.time()
    print(f"[exp2] scanning {len(test_graphs.test_idx):,} test graphs for "
          f"near-degenerate pairs (k={EIGENSPACE_K}) ...")
    for pos, gi in enumerate(test_graphs.test_idx.tolist()):
        d = test_graphs.get(gi)
        n = d.num_nodes
        if n < 3: continue
        evals, evecs, _ = C.eig_and_gap(d.edge_index, n, k=n)    # all evals
        k_use = min(EIGENSPACE_K, n)
        if k_use < 3: continue
        diffs = np.diff(evals[1:k_use])          # gaps on λ_1..λ_{k_use-1}
        if diffs.size == 0: continue
        g_exact_mask = diffs < GAP_EXACT
        g_near_mask  = (diffs >= GAP_EXACT) & (diffs < GAP_NEARDEG)
        if g_exact_mask.any():
            cand = np.where(g_exact_mask)[0]
            j = int(rng.choice(cand))            # relative index into diffs
            pair_idx = 1 + j                     # absolute (column in evecs)
            exact.append((gi, pair_idx, float(diffs[j]), evals, evecs))
        elif g_near_mask.any():
            cand = np.where(g_near_mask)[0]
            j = int(rng.choice(cand))
            pair_idx = 1 + j
            near.append((gi, pair_idx, float(diffs[j]), evals, evecs))
        if (pos + 1) % 5000 == 0:
            print(f"[exp2]   scanned {pos+1:6d}/{len(test_graphs.test_idx)}  "
                  f"exact={len(exact)} near={len(near)}  "
                  f"({time.time()-t0:5.1f}s)")

    per_group = n_target // 2
    if len(exact) > per_group:
        pick = rng.choice(len(exact), size=per_group, replace=False)
        exact = [exact[i] for i in pick]
    if len(near) > per_group:
        pick = rng.choice(len(near), size=per_group, replace=False)
        near = [near[i] for i in pick]
    cands = exact + near
    print(f"[exp2] sampled {len(exact)} exact-degenerate + {len(near)} "
          f"near-degenerate = {len(cands)} graphs")
    return cands


# -------------------- per-method experiment ------------------------------

def rotate_and_recompute(base: "C.Data", evals: np.ndarray, evecs: np.ndarray,
                         pair_idx: int, theta: float, cfg, pe_type: str):
    """Return a Data clone with PE derived from eigenvectors rotated in the
    (pair_idx, pair_idx+1) 2-D subspace by ``theta`` radians."""
    V = evecs.copy()
    i, j = pair_idx, pair_idx + 1
    c, s = np.cos(theta), np.sin(theta)
    a = V[:, i].copy(); b = V[:, j].copy()
    V[:, i] =  c * a + s * b
    V[:, j] = -s * a + c * b
    d = base.clone()
    for attr in ("EigVals", "EigVecs", "eigvals_sn", "eigvecs_sn",
                 "pestat_LHKS_eigvals", "pestat_LHKS_eigvecs_sq"):
        if hasattr(d, attr): delattr(d, attr)
    d = C.apply_pe_from_decomp(d, evals, V, cfg, pe_type)
    return d


def run_one_checkpoint(method: str, run_dir: str, test_graphs: C.PCQMGraphs,
                       cands, device: torch.device, rng_seed: int = 0) -> dict:
    print(f"\n[exp2] === {method} :: {run_dir} ===")
    model, cfg, pe_type = C.load_model_and_cfg(run_dir, device)
    print(f"[exp2] loaded model (pe_type={pe_type})")

    rng = default_rng(rng_seed)
    rows: List[dict] = []
    t0 = time.time()
    for pos, (gi, pair_idx, pair_gap, evals, evecs) in enumerate(cands):
        base = test_graphs.get(gi)
        # Baseline PE (theta=0) so we can also report deviation from baseline.
        thetas = rng.uniform(0.0, 2 * np.pi, size=N_ROTATIONS)
        replicas = [rotate_and_recompute(base, evals, evecs, pair_idx, th,
                                         cfg, pe_type) for th in thetas]
        preds = C.batched_forward(model, replicas, device, BATCH_SIZE)
        rows.append({"gi": int(gi),
                     "num_nodes": int(base.num_nodes),
                     "pair_idx": int(pair_idx),
                     "pair_gap": float(pair_gap),
                     "is_exact": bool(pair_gap < GAP_EXACT),
                     "pred_mean": float(preds.mean()),
                     "pred_std":  float(preds.std()),
                     "pred_min":  float(preds.min()),
                     "pred_max":  float(preds.max())})
        if (pos + 1) % 100 == 0 or pos + 1 == len(cands):
            dt = time.time() - t0
            print(f"[exp2] {pos+1:5d}/{len(cands)} ({dt:6.1f}s, "
                  f"{dt / (pos + 1):5.2f}s/graph)")

    seed_tag = Path(run_dir).name
    out_path = OUT_DIR / f"{method}__{seed_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"method": method, "run_dir": run_dir, "pe_type": pe_type,
                   "rows": rows}, f)
    print(f"[exp2] wrote {out_path} ({len(rows)} graphs)")
    return {"method": method, "run_dir": run_dir, "rows": rows}


# ------------------------------ plots ------------------------------------

def plot_results(method_seeds: Dict[str, List[dict]], out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    # Average per-graph std across seeds.
    per_method = {}           # method -> List[(gap, is_exact, mean_std_over_seeds)]
    for method, seeds in method_seeds.items():
        by_gi: Dict[int, dict] = {}
        for s in seeds:
            for r in s["rows"]:
                rec = by_gi.setdefault(r["gi"], {"gap": r["pair_gap"],
                                                 "exact": r["is_exact"],
                                                 "stds": []})
                rec["stds"].append(r["pred_std"])
        pts = [(v["gap"], v["exact"], float(np.mean(v["stds"])))
               for v in by_gi.values()]
        per_method[method] = pts

    colors = {"LapPE": "#d62728", "SignNet-MLP": "#ff7f0e",
              "SignNet-DeepSets": "#9467bd", "L-HKS": "#1f77b4",
              "fix-L-HKS": "#17becf"}

    # --- exact-degeneracy bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(per_method.keys())
    means, sems, ns = [], [], []
    for m in methods:
        vals = [s for (_, ex, s) in per_method[m] if ex]
        means.append(float(np.mean(vals)) if vals else 0.0)
        sems .append(float(np.std(vals) / max(1, np.sqrt(len(vals)))) if vals else 0.0)
        ns.append(len(vals))
    # log scale requires strictly positive bar heights
    eps = 1e-20
    means_bar = [max(m, eps) for m in means]
    bars = ax.bar(methods, means_bar, yerr=sems, capsize=3,
                  color=[colors.get(m, "gray") for m in methods])
    ax.set_yscale("log")
    ax.set_ylabel(r"mean prediction std on exact-degeneracy subset ($\delta_{\min}<10^{-10}$)")
    ax.set_title("Exp 2a - well-definedness under eigenbasis rotation")
    for b, n in zip(bars, ns):
        ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = out_dir / "exp2_bar.png"
    fig.savefig(p1, dpi=160); plt.close(fig)
    print(f"[exp2] wrote {p1}")

    # --- scatter: per-graph std vs pair gap (log x) ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in methods:
        gaps = np.array([g for (g, _, _)  in per_method[m]])
        stds = np.array([s for (_, _, s)  in per_method[m]])
        # Plot gaps < 1e-12 at a floor for visibility on log axis.
        g_plot = np.clip(gaps, 1e-16, None)
        s_plot = np.clip(stds, 1e-20, None)
        ax.scatter(g_plot, s_plot, s=10, alpha=0.45, color=colors.get(m, None),
                   label=m, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"pair gap $\lambda_{i+1}-\lambda_i$")
    ax.set_ylabel("prediction std (20 SO(2) rotations)")
    ax.set_title("Exp 2b - prediction instability vs eigenpair gap")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p2 = out_dir / "exp2_scatter.png"
    fig.savefig(p2, dpi=160); plt.close(fig)
    print(f"[exp2] wrote {p2}")

    summary = {m: {"exact_n":  ns[i],
                   "exact_mean_std": means[i],
                   "exact_sem_std":  sems[i]}
               for i, m in enumerate(methods)}
    return summary


# --------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHOD_RUN_DIRS.keys()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-dirs", nargs="*", default=None)
    ap.add_argument("--n-graphs", type=int, default=N_TARGET_GRAPHS)
    ap.add_argument("--plot-only", action="store_true",
                    help="Only aggregate JSONs under OUT_DIR and regenerate plots")
    ap.add_argument("--no-plot", action="store_true",
                    help="Run experiment(s) only; skip aggregation/plot (use --plot-only after)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not (args.method or args.all or args.plot_only):
        ap.error("specify --method, --all, or --plot-only")
    if args.plot_only and args.no_plot:
        ap.error("cannot combine --plot-only and --no-plot")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if not args.plot_only:
        test_graphs = C.PCQMGraphs.load()
        cands_cache = OUT_DIR / "candidates.npz"
        if cands_cache.exists():
            print(f"[exp2] reusing candidate list from {cands_cache}")
            z = np.load(cands_cache, allow_pickle=True)
            cands = list(z["cands"])
        else:
            cands = select_candidates(test_graphs, n_target=args.n_graphs, seed=0)
            np.savez(cands_cache, cands=np.array(cands, dtype=object))

        methods = list(METHOD_RUN_DIRS) if args.all else [args.method]
        for m in methods:
            dirs = args.run_dirs if (args.method == m and args.run_dirs) \
                   else METHOD_RUN_DIRS[m]
            for rd in dirs:
                run_one_checkpoint(m, rd, test_graphs, cands, device,
                                   rng_seed=hash((m, rd)) & 0xFFFF)

    results: Dict[str, List[dict]] = {}
    if args.no_plot:
        print("[exp2] --no-plot: skipping aggregation and figures.")
        return
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name in ("exp2_summary.json",): continue
        with open(path) as f:
            payload = json.load(f)
        results.setdefault(payload["method"], []).append(payload)
    if not results:
        print("[exp2] no result json files yet."); return
    summary = plot_results(results, OUT_DIR)
    with open(OUT_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp2] wrote {OUT_DIR / 'exp2_summary.json'}")


if __name__ == "__main__":
    main()
