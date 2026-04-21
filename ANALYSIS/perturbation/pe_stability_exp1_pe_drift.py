"""Experiment 1 (PE-level) - PE drift under non-bridge edge removal.

Same perturbation protocol as ``pe_stability_exp1.py``, but the observable is
the positional encoding tensor itself, not a scalar model prediction:

    δ_PE(G, G') = || PE(G) - PE(G') ||_F / || PE(G) ||_F
                 (relative Frobenius drift; NaN if ||PE(G)||_F ≈ 0)

No trained weights and no diffusion-time parameters ever enter the
comparison. Both PEs are descriptor-intrinsic at the truncations the
trained models use. We do not even touch checkpoints: the only thing this
script needs from the codebase-side config is ``max_freqs`` / ``num_eigvec``
so that ``compute_posenc_stats`` produces the correct tensor shape -- and
those live in the YAML configs under ``configs/GPS/``.

Methods
-------
  * LapPE-aligned   -- truncated Laplacian eigenvectors EigVecs ∈ R^{n × 8},
                      sign-aligned column-wise to the base before differencing.
                      Quotients out the trivial sign gauge; residual subspace-
                      rotation ambiguity in genuinely degenerate subspaces
                      (bin A) is left in the drift on purpose.

  * fix-L-HKS-K8    -- full heat-kernel descriptor
                         HKS[v, j] = Σ_{i=1..8} exp(-λ_i · t_j) · φ_i(v)^2
                      at K = 8 analytic fixed diffusion times
                         t_j = exp(linspace(log 0.01, log 100, 8)).
                      This is the same init grid the LHKS encoder uses when
                      ``freeze_times = True`` (see
                      graphgps/encoder/lhks_encoder.py:51), so the descriptor
                      is strictly intrinsic.

Positioning
-----------
This is a PE-level robustness experiment under practical truncation. It is
not a numerical verification of the full-HKS stability theorem (which is
analytic over the full spectrum and all times); the theorem is already
known analytically. What this script tests is whether that protection
survives the two practical compromises the models rely on: a small
eigenbasis (k = 8) and a finite bank of diffusion times (K = 8).

Usage
-----
    python ANALYSIS/perturbation/pe_stability_exp1_pe_drift.py --all
    python ANALYSIS/perturbation/pe_stability_exp1_pe_drift.py --method LapPE-aligned
    python ANALYSIS/perturbation/pe_stability_exp1_pe_drift.py --plot-only

Outputs (ANALYSIS/perturbation/exp1_pe_drift_results/)
    <method>.json      per-graph PE drift per k_remove
    main.png           LapPE-aligned vs. fix-L-HKS-K8
    summary.json       aggregate table + bin_stats + meta (including the
                       exact diffusion-time grid used)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.random import default_rng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


# =============================== methods ==================================
# ``pe_kind``:   "lappe" -> EigVecs drift (with sign_align below)
#                "hks"   -> HKS descriptor drift reconstructed from
#                           (pestat_LHKS_eigvals, pestat_LHKS_eigvecs_sq, times)
#
# ``cfg_path``:  YAML in configs/GPS/. We only need its ``posenc_*`` block so
#                ``compute_posenc_stats`` truncates to the right rank. No
#                checkpoint, no model forward pass, no trained parameter.
METHODS: Dict[str, dict] = {
    "LapPE-aligned": {
        "cfg_path":   "configs/GPS/pcqm4m-subset-GPS+LapPE.yaml",
        "pe_kind":    "lappe",
        "sign_align": True,
    },
    "fix-L-HKS-K8": {
        "cfg_path":   "configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3-fixed.yaml",
        "pe_kind":    "hks",
        "hks_K":      8,
    },
}

K_REMOVE_LEVELS = (1, 2, 3)
N_PERTURBATIONS = 20
N_TARGET_GRAPHS = 5000
OUT_DIR         = Path("ANALYSIS/perturbation/exp1_pe_drift_results")


# ============================ cfg-only loader ============================

def _load_cfg_only(cfg_path: str) -> Tuple[object, str]:
    """Merge a GraphGym YAML into the global ``cfg`` without building any model.

    This is the only thing ``compute_posenc_stats`` (called from
    ``C.recompute_pe``) needs to truncate the PE correctly. Returns the live
    global ``cfg`` node and the enabled PE type (e.g. ``"LapPE"`` / ``"LHKS"``).
    """
    import graphgps  # noqa: F401  (registers custom posenc_* config nodes)
    from torch_geometric.graphgym.config import cfg, set_cfg
    from graphgps.finetuning import set_new_cfg_allowed

    set_cfg(cfg)
    set_new_cfg_allowed(cfg, True)
    cfg.merge_from_file(cfg_path)
    # Fields some dataset encoders assume at import time.
    cfg.share.dim_in  = 9
    cfg.share.dim_out = 1
    cfg.wandb.use = False

    pe_type: Optional[str] = None
    for key in cfg.keys():
        if key.startswith("posenc_") and getattr(cfg, key).enable:
            pe_type = key.split("_", 1)[1]
            break
    if pe_type is None:
        raise RuntimeError(
            f"No posenc_*.enable=True in {cfg_path}; this script needs a PE.")
    return cfg, pe_type


# ============================ PE construction =============================

def _analytic_times(K: int) -> torch.Tensor:
    """t_j = exp(linspace(log 0.01, log 100, K)) -- same grid the LHKS encoder
    uses when ``freeze_times=True`` (graphgps/encoder/lhks_encoder.py:51)."""
    return torch.exp(torch.linspace(math.log(0.01), math.log(100.0), K))


def _hks_from_ingredients(eigvals: torch.Tensor, eigvecs_sq: torch.Tensor,
                           times: torch.Tensor) -> torch.Tensor:
    """Vectorised HKS reconstruction. Mirrors lhks_encoder.py:101-105 exactly.

    Shapes: eigvals, eigvecs_sq: [N, k]  ; times: [K]  ; return: [N, K].
    """
    ev  = eigvals.to(torch.float64)
    es  = eigvecs_sq.to(torch.float64)
    t   = times.to(torch.float64)
    exp_terms = torch.exp(-ev.unsqueeze(-1) * t)       # [N, k, K]
    hks = (exp_terms * es.unsqueeze(-1)).sum(dim=1)    # [N, K]
    return hks


def _sign_align(base: torch.Tensor, pert: torch.Tensor) -> torch.Tensor:
    """Column-wise sign flip of ``pert`` to maximise <base[:, j], pert[:, j]>.
    No-op for nonnegative tensors (e.g. HKS) because all inner products >= 0.
    """
    if base.shape != pert.shape or base.ndim != 2:
        return pert
    dots = (base * pert).sum(dim=0)
    flip = torch.where(dots < 0, -1.0, 1.0).to(pert.dtype)
    return pert * flip


def _lappe_tensor(data) -> torch.Tensor:
    t = getattr(data, "EigVecs", None)
    if t is None:
        raise RuntimeError("recompute_pe did not set data.EigVecs for LapPE.")
    return t.detach().to(torch.float64).cpu()


def _hks_tensor(data, times: torch.Tensor) -> torch.Tensor:
    ev = getattr(data, "pestat_LHKS_eigvals", None)
    es = getattr(data, "pestat_LHKS_eigvecs_sq", None)
    if ev is None or es is None:
        raise RuntimeError(
            "recompute_pe did not set data.pestat_LHKS_eigvals / "
            "data.pestat_LHKS_eigvecs_sq for LHKS.")
    return _hks_from_ingredients(ev.detach().cpu(), es.detach().cpu(), times)


def _pe_tensor(data, method_cfg: dict,
               times: Optional[torch.Tensor]) -> torch.Tensor:
    if method_cfg["pe_kind"] == "lappe":
        return _lappe_tensor(data)
    if method_cfg["pe_kind"] == "hks":
        assert times is not None
        return _hks_tensor(data, times)
    raise ValueError(f"Unknown pe_kind {method_cfg['pe_kind']!r}")


def pe_drift(base_pe: torch.Tensor, pert_pe: torch.Tensor,
             *, sign_align: bool) -> float:
    """δ_PE = ||base - pert||_F / ||base||_F after optional sign alignment."""
    if base_pe.shape != pert_pe.shape:
        return float("nan")
    if sign_align:
        pert_pe = _sign_align(base_pe, pert_pe)
    base_norm = torch.linalg.norm(base_pe.reshape(-1)).item()
    if not math.isfinite(base_norm) or base_norm < 1e-30:
        return float("nan")
    diff = (base_pe - pert_pe).reshape(-1)
    return float(torch.linalg.norm(diff).item() / base_norm)


# ========================== bin validation ================================

_BIN_LABEL = {"A": "<1e-10", "B": "[1e-10,0.05)",
              "C": "[0.05,0.15)", "D": ">=0.15"}


def _bin_stats_from_sample(test_graphs: C.PCQMGraphs,
                           sample_ids: np.ndarray,
                           sample_dmin: np.ndarray) -> Dict[str, dict]:
    """Validate the stratification: print and return per-bin statistics,
    including mean / median #non-bridges so we can rule out the concern that
    some bins are simply easier to perturb.
    """
    bins = np.array([C.deltamin_bin(float(d)) for d in sample_dmin])
    n_nb: List[int] = []
    for gi in sample_ids.tolist():
        d = test_graphs.get(gi)
        n_nb.append(len(C.find_non_bridge_edges(d.edge_index, d.num_nodes)))
    n_nb = np.asarray(n_nb, dtype=np.int64)

    stats: Dict[str, dict] = {}
    header = (f"  {'bin':>3}  {'n':>6}  {'mean δ_min':>12}  "
              f"{'median δ_min':>13}  {'min δ_min':>12}  {'max δ_min':>12}  "
              f"{'mean #nb':>9}  {'median #nb':>10}  label")
    print("\n[exp1-pe] Bin validation (stratified sample):")
    print(header); print("  " + "-" * (len(header) - 2))

    prev_max = -np.inf
    monotone_ok = True
    for b in ["A", "B", "C", "D"]:
        mask = bins == b
        vals = sample_dmin[mask]
        nbs  = n_nb[mask]
        label = _BIN_LABEL[b]
        if vals.size == 0:
            stats[b] = {"n": 0, "label": label}
            print(f"  {b:>3}  {0:>6}  {'-':>12}  {'-':>13}  {'-':>12}  "
                  f"{'-':>12}  {'-':>9}  {'-':>10}  {label}  [empty]")
            continue
        me, md = float(vals.mean()), float(np.median(vals))
        mn, mx = float(vals.min()),  float(vals.max())
        nm, nmd = float(nbs.mean()),  float(np.median(nbs))
        stats[b] = {
            "n": int(vals.size), "label": label,
            "mean_dmin": me, "median_dmin": md,
            "min_dmin":  mn, "max_dmin":    mx,
            "mean_nonbridges": nm, "median_nonbridges": nmd,
        }
        print(f"  {b:>3}  {vals.size:>6}  {me:>12.3e}  {md:>13.3e}  "
              f"{mn:>12.3e}  {mx:>12.3e}  {nm:>9.2f}  {nmd:>10.1f}  {label}")
        if mn + 1e-15 < prev_max:
            print(f"  [warn] bin {b} min ({mn:.3e}) < previous bin max "
                  f"({prev_max:.3e}); monotone δ_min ordering violated!")
            monotone_ok = False
        prev_max = max(prev_max, mx)

    if not any(stats[b].get("n", 0) > 0 for b in "ABCD"):
        raise RuntimeError("[exp1-pe] all bins are empty; refusing to run.")
    if not monotone_ok:
        print("[exp1-pe] WARNING: bin ordering is not monotone; interpret "
              "'A = smallest gap, D = largest gap' with caution.")
    return stats


# ========================= per-method experiment ==========================

def run_one_method(method: str, test_graphs: C.PCQMGraphs,
                   sample_ids: np.ndarray, sample_dmin: np.ndarray) -> dict:
    method_cfg = METHODS[method]
    cfg_path   = method_cfg["cfg_path"]
    print(f"\n[exp1-pe] === {method} :: cfg={cfg_path} ===")

    cfg, pe_type = _load_cfg_only(cfg_path)
    if method_cfg["pe_kind"] == "lappe" and pe_type != "LapPE":
        raise RuntimeError(f"[{method}] expected LapPE cfg, got pe_type="
                           f"{pe_type!r} in {cfg_path}.")
    if method_cfg["pe_kind"] == "hks" and pe_type != "LHKS":
        raise RuntimeError(f"[{method}] expected LHKS cfg, got pe_type="
                           f"{pe_type!r} in {cfg_path}.")

    times: Optional[torch.Tensor] = None
    times_meta: dict = {}
    if method_cfg["pe_kind"] == "hks":
        K = int(method_cfg["hks_K"])
        times = _analytic_times(K)
        times_meta = {
            "K": K, "hks_times_source": "analytic",
            "times": [float(x) for x in times.tolist()],
        }
        # Sanity: if the YAML sets freeze_times at the same K, the encoder's
        # frozen init grid IS our analytic grid. Assert that to document the
        # descriptor-intrinsic claim.
        ckpt_K = int(cfg.posenc_LHKS.kernel_times)
        if bool(cfg.posenc_LHKS.freeze_times) and ckpt_K == K:
            times_meta["matches_encoder_freeze_times_init"] = True
        print(f"[exp1-pe] HKS times: K={K} "
              f"times[0,1,-2,-1]={times.tolist()[:2]}...{times.tolist()[-2:]}")

    sign_align = bool(method_cfg.get("sign_align", False))
    rows: List[dict] = []
    t0 = time.time()

    for gi_pos, gi in enumerate(sample_ids.tolist()):
        base = test_graphs.get(gi)
        n    = int(base.num_nodes)
        nb   = C.find_non_bridge_edges(base.edge_index, n)
        if not nb:
            continue

        # Base PE, computed once per graph.
        base_data = C.recompute_pe(base.clone(), cfg, pe_type)
        base_pe   = _pe_tensor(base_data, method_cfg, times)

        # Graph-keyed RNG -> every method sees identical perturbation picks.
        rng = default_rng(int(gi))

        per_k: Dict[int, dict] = {}
        for k_rm in K_REMOVE_LEVELS:
            if len(nb) < k_rm:
                per_k[k_rm] = {"n": 0, "mean_drift": float("nan"),
                               "median_drift": float("nan"),
                               "std_drift": float("nan"), "drifts": []}
                continue
            drifts: List[float] = []
            for _ in range(N_PERTURBATIONS):
                picks_idx = rng.choice(len(nb), size=k_rm, replace=False)
                picks     = [nb[i] for i in picks_idx]
                d = C.remove_edges(base, picks)
                d = C.recompute_pe(d, cfg, pe_type)
                pp = _pe_tensor(d, method_cfg, times)
                drifts.append(pe_drift(base_pe, pp, sign_align=sign_align))
            darr = np.asarray(drifts, dtype=np.float64)
            per_k[k_rm] = {
                "n":            int(darr.size),
                "mean_drift":   float(np.nanmean(darr)),
                "median_drift": float(np.nanmedian(darr)),
                "std_drift":    float(np.nanstd(darr)),
                "drifts":       [float(x) for x in darr.tolist()],
            }

        rows.append({
            "gi":             int(gi),
            "num_nodes":      n,
            "num_nonbridges": int(len(nb)),
            "delta_min_k8":   float(sample_dmin[gi_pos]),
            "bin":            C.deltamin_bin(float(sample_dmin[gi_pos])),
            "per_k":          per_k,
        })
        if (gi_pos + 1) % 100 == 0 or gi_pos + 1 == len(sample_ids):
            dt = time.time() - t0
            print(f"[exp1-pe] {gi_pos + 1:5d}/{len(sample_ids)} "
                  f"({dt:6.1f}s, {dt / (gi_pos + 1):5.2f}s/graph)")

    out_path = OUT_DIR / f"{method}.json"
    C.ensure_output_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump({
            "method":     method,
            "cfg_path":   cfg_path,
            "pe_type":    pe_type,
            "pe_kind":    method_cfg["pe_kind"],
            "sign_align": sign_align,
            "times_meta": times_meta,
            "rows":       rows,
        }, f)
    print(f"[exp1-pe] wrote {out_path} ({len(rows)} graphs)")
    return {"method": method, "cfg_path": cfg_path, "pe_type": pe_type,
            "rows": rows}


# =============================== aggregation ==============================

def _aggregate(results_per_method: Dict[str, List[dict]]) -> dict:
    agg: Dict[str, dict] = {}
    for method, seeds in results_per_method.items():
        per_bin_per_k: Dict[str, Dict[int, list]] = {
            b: {k: [] for k in K_REMOVE_LEVELS} for b in "ABCD"
        }
        for seed in seeds:
            for r in seed["rows"]:
                b = r["bin"]
                if b not in per_bin_per_k:
                    continue
                for k_str, v in r["per_k"].items():
                    k = int(k_str)
                    s = v.get("mean_drift", float("nan"))
                    if np.isfinite(s):
                        per_bin_per_k[b][k].append(float(s))
        agg[method] = {
            b: {k: (float(np.mean(v)) if v else float("nan"), int(len(v)))
                for k, v in kr.items()}
            for b, kr in per_bin_per_k.items()
        }
    return agg


# ================================ plotting =================================

_COLORS  = {"LapPE-aligned": "#d62728", "fix-L-HKS-K8": "#17becf"}
_MARKERS = {"LapPE-aligned": "o",       "fix-L-HKS-K8": "v"}


def plot_main(agg: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    have = [m for m in METHODS if m in agg]
    if not have:
        print(f"[exp1-pe] skip plot {out_path.name}: no data for any method")
        return
    bins = ["A", "B", "C", "D"]
    bin_labels = [r"$<10^{-10}$", r"$<0.05$", r"$<0.15$", r"$\geq 0.15$"]
    fig, axes = plt.subplots(1, len(K_REMOVE_LEVELS),
                             figsize=(4.2 * len(K_REMOVE_LEVELS), 3.8),
                             sharey=True)
    for ax, k in zip(axes, K_REMOVE_LEVELS):
        for method in have:
            per_bin = agg[method]
            ys = [per_bin[b].get(k, (float("nan"), 0))[0] for b in bins]
            ns = [per_bin[b].get(k, (float("nan"), 0))[1] for b in bins]
            ys_plot = [y if np.isfinite(y) and y > 0 else np.nan for y in ys]
            ax.plot(bins, ys_plot, marker=_MARKERS.get(method, "o"),
                    color=_COLORS.get(method), label=method)
            for x, y, n in zip(bins, ys, ns):
                if not np.isfinite(y) or y <= 0:
                    continue
                ax.annotate(f"n={n}", (x, y), fontsize=6, alpha=0.6,
                            xytext=(0, 4), textcoords="offset points",
                            ha="center")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\delta_{\min}$ bin "
                      r"(A=smallest gap $\rightarrow$ D=largest)")
        ax.set_xticks(range(4)); ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_title(f"remove {k} non-bridge edge(s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(
        r"$\delta_{\mathrm{PE}} = \|PE - PE'\|_F / \|PE\|_F$")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("LapPE (sign-aligned) vs. fix-L-HKS (K=8) "
                 "- PE drift under non-bridge edge removal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[exp1-pe] wrote {out_path}")


# =================================== main =================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS.keys()))
    ap.add_argument("--all", action="store_true",
                    help="Run both methods in METHODS.")
    ap.add_argument("--n-graphs", type=int, default=N_TARGET_GRAPHS)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if not (args.method or args.all or args.plot_only):
        ap.error("specify --method, --all, or --plot-only")
    if args.plot_only and args.no_plot:
        ap.error("cannot combine --plot-only and --no-plot")

    C.ensure_output_dir(OUT_DIR)

    bin_stats: Optional[Dict[str, dict]] = None
    if not args.plot_only:
        test_graphs = C.PCQMGraphs.load()
        sample_ids, sample_dmin, counts = C.stratified_subsample(
            test_graphs, n_target=args.n_graphs, bin_k=8,
            seed=0, skip_trees=True, skip_tiny=True)
        bin_stats = _bin_stats_from_sample(test_graphs, sample_ids, sample_dmin)

        to_run = [args.method] if args.method else list(METHODS.keys())
        for m in to_run:
            out_path = OUT_DIR / f"{m}.json"
            if out_path.exists():
                print(f"[skip] {m} :: result already exists at {out_path}")
                continue
            run_one_method(m, test_graphs, sample_ids, sample_dmin)

    if args.no_plot:
        print("[exp1-pe] --no-plot: skipping aggregation and figure.")
        return

    # Aggregate every method JSON in the output dir.
    results_per_method: Dict[str, List[dict]] = {}
    meta_runs: Dict[str, dict] = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        with open(path) as f:
            payload = json.load(f)
        results_per_method.setdefault(payload["method"], []).append(payload)
        meta_runs[payload["method"]] = {
            "cfg_path":   payload.get("cfg_path"),
            "pe_type":    payload.get("pe_type"),
            "pe_kind":    payload.get("pe_kind"),
            "sign_align": payload.get("sign_align"),
            "times_meta": payload.get("times_meta", {}),
        }
    if not results_per_method:
        print("[exp1-pe] no result json files yet.")
        return
    agg = _aggregate(results_per_method)

    summary = {
        "agg":       agg,
        "bin_stats": bin_stats,
        "meta": {
            "drift_metric":    "||PE(G)-PE(G')||_F / ||PE(G)||_F",
            "bin_k":           8,
            "n_target":        args.n_graphs if not args.plot_only else None,
            "N_PERTURBATIONS": N_PERTURBATIONS,
            "K_REMOVE_LEVELS": list(K_REMOVE_LEVELS),
            "methods":         list(METHODS.keys()),
            "runs":            meta_runs,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp1-pe] wrote {OUT_DIR / 'summary.json'}")

    plot_main(agg, OUT_DIR / "main.png")


if __name__ == "__main__":
    main()
