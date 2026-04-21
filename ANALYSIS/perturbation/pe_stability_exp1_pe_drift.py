import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import default_rng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

import graphgps  # noqa: F401
from torch_geometric.graphgym.config import cfg as GLOBAL_CFG, set_cfg
from graphgps.finetuning import set_new_cfg_allowed

METHODS = {
    "LapPE-aligned": dict(
        cfg="configs/GPS/pcqm4m-subset-GPS+LapPE.yaml",
        kind="lappe", align=True),
    "fix-L-HKS-K8": dict(
        cfg="configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3-fixed.yaml",
        kind="hks", align=False),
}
K_REMOVE        = (1, 2, 3)
N_PERTURBATIONS = 20
OUT_DIR         = Path("ANALYSIS/perturbation/exp1_pe_drift_results")
HKS_TIMES       = torch.exp(torch.linspace(math.log(0.01), math.log(100.0), 8))


def load_cfg(cfg_path: str):
    set_cfg(GLOBAL_CFG)
    set_new_cfg_allowed(GLOBAL_CFG, True)
    GLOBAL_CFG.merge_from_file(cfg_path)
    GLOBAL_CFG.share.dim_in, GLOBAL_CFG.share.dim_out, GLOBAL_CFG.wandb.use = 9, 1, False
    pe_type = next(
        k.split("_", 1)[1] for k in GLOBAL_CFG.keys()
        if k.startswith("posenc_") and getattr(GLOBAL_CFG, k).enable)
    return GLOBAL_CFG, pe_type


def pe_tensor(data, kind: str) -> torch.Tensor:
    if kind == "lappe":
        return data.EigVecs.detach().double().cpu()
    ev = data.pestat_LHKS_eigvals.detach().double().cpu()
    es = data.pestat_LHKS_eigvecs_sq.detach().double().cpu()
    t  = HKS_TIMES.double()
    return (torch.exp(-ev.unsqueeze(-1) * t) * es.unsqueeze(-1)).sum(1)


def drift(base: torch.Tensor, pert: torch.Tensor, align: bool) -> float:
    if align:
        sign = torch.where((base * pert).sum(0) < 0, -1.0, 1.0).to(pert.dtype)
        pert = pert * sign
    num = torch.linalg.norm(base - pert, dim=1)
    den = torch.linalg.norm(base,        dim=1) + 1e-10
    return float((num / den).mean().item())


def run_method(method, spec, test_graphs, sample_ids, sample_dmin):
    print(f"\n=== {method} ===", flush=True)
    cfg, pe_type = load_cfg(spec["cfg"])
    rows, t0 = [], time.time()
    for i, gi in enumerate(sample_ids.tolist()):
        g  = test_graphs.get(gi)
        nb = C.find_non_bridge_edges(g.edge_index, int(g.num_nodes))
        if not nb:
            continue
        base = pe_tensor(C.recompute_pe(g.clone(), cfg, pe_type), spec["kind"])
        rng  = default_rng(int(gi))

        per_k = {}
        for k in K_REMOVE:
            if len(nb) < k:
                per_k[k] = []
                continue
            drifts = []
            for _ in range(N_PERTURBATIONS):
                picks = [nb[j] for j in rng.choice(len(nb), size=k, replace=False)]
                gp    = C.recompute_pe(C.remove_edges(g, picks), cfg, pe_type)
                drifts.append(drift(base, pe_tensor(gp, spec["kind"]),
                                    spec["align"]))
            per_k[k] = drifts

        rows.append(dict(gi=int(gi), n=int(g.num_nodes),
                         dmin=float(sample_dmin[i]),
                         bin=C.deltamin_bin(float(sample_dmin[i])),
                         per_k=per_k))
        if (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            dt  = time.time() - t0
            eta = dt / (i + 1) * (len(sample_ids) - i - 1)
            print(f"  {i + 1:5d}/{len(sample_ids)}  "
                  f"({dt:5.1f}s, ETA {eta / 60:4.1f}min)", flush=True)

    out = OUT_DIR / f"{method}.json"
    out.write_text(json.dumps(dict(method=method, pe_type=pe_type,
                                   align=spec["align"], rows=rows)))
    print(f"  wrote {out} ({len(rows)} graphs)")
    return rows


def aggregate(rows_per_method):
    out = {}
    for method, rows in rows_per_method.items():
        buckets = {b: {k: [] for k in K_REMOVE} for b in "ABCD"}
        for r in rows:
            for k, ds in r["per_k"].items():
                if ds:
                    buckets[r["bin"]][int(k)].append(float(np.nanmean(ds)))
        per_method = {}
        for b, kr in buckets.items():
            per_method[b] = {}
            for k, v in kr.items():
                if not v:
                    per_method[b][k] = (float("nan"), 0)
                    continue
                arr = np.asarray(v, dtype=np.float64)
                per_method[b][k] = (float(np.nanmean(arr)),
                                    int(np.isfinite(arr).sum()))
        out[method] = per_method
    return out


def plot(agg, out_path):
    from matplotlib.ticker import MaxNLocator
    colors  = {"LapPE-aligned": "#d62728", "fix-L-HKS-K8": "#17becf"}
    markers = {"LapPE-aligned": "o",       "fix-L-HKS-K8": "v"}
    bins       = ["A", "B", "C", "D"]
    ticklabels = [r"$<10^{-10}$", r"$<0.05$", r"$<0.15$", r"$\geq 0.15$"]

    all_y = [agg[m][b][k][0] for m in agg for b in bins for k in K_REMOVE]
    finite = [y for y in all_y if np.isfinite(y)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad    = 0.05 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
        ylim   = (max(0.0, lo - pad), hi + pad)
    else:
        ylim = (0.0, 1.0)

    fig, axes = plt.subplots(1, len(K_REMOVE), figsize=(12, 3.8), sharey=True)
    for ax, k in zip(axes, K_REMOVE):
        for m, per_bin in agg.items():
            ys = [per_bin[b][k][0] for b in bins]
            ax.plot(bins, [y if np.isfinite(y) else np.nan for y in ys],
                    marker=markers[m], color=colors[m], label=m)
        ax.set_title(f"remove {k} edge(s)")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlabel(r"$\delta_{\min}$")
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(
        r"$\frac{1}{n}\sum_v \|PE(v)-PE'(v)\|_2 / \|PE(v)\|_2$")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-graphs", type=int, default=5000)
    args = ap.parse_args()

    C.ensure_output_dir(OUT_DIR)
    print("loading PCQM4Mv2 ...", flush=True)
    test_graphs = C.PCQMGraphs.load()
    sample_ids, sample_dmin, counts = C.stratified_subsample(
        test_graphs, n_target=args.n_graphs, bin_k=8, seed=0,
        skip_trees=True, skip_tiny=True)
    print(f"bin counts: {counts}", flush=True)

    rows_per_method = {}
    for method, spec in METHODS.items():
        out = OUT_DIR / f"{method}.json"
        if out.exists():
            print(f"[skip] {method} :: {out} exists")
            rows = json.loads(out.read_text())["rows"]
            for r in rows:
                r["per_k"] = {
                    int(k): (v["drifts"] if isinstance(v, dict) else v)
                    for k, v in r["per_k"].items()}
            rows_per_method[method] = rows
        else:
            rows_per_method[method] = run_method(
                method, spec, test_graphs, sample_ids, sample_dmin)

    agg = aggregate(rows_per_method)
    (OUT_DIR / "summary.json").write_text(json.dumps(dict(
        agg=agg, bin_counts=counts,
        meta=dict(drift_metric="mean_v ||PE(v) - PE'(v)||_2 / (||PE(v)||_2 + 1e-10)",
                  n_graphs=args.n_graphs,
                  N_PERTURBATIONS=N_PERTURBATIONS,
                  K_REMOVE=list(K_REMOVE),
                  hks_times=HKS_TIMES.tolist()),
    ), indent=2))
    plot(agg, OUT_DIR / "main.png")


if __name__ == "__main__":
    main()
