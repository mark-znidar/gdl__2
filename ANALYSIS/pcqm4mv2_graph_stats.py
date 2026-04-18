"""PCQM4Mv2 graph statistics: size, bridges, Laplacian eigenvalue gaps.

Reproduces the same split as graphgps/loader/master_loader.py (seed=42):
 - 10% train subset       = first 10% of train after carving out 150k for val
 - test set               = original OGB 'valid' split
Loads the processed PyG file directly (equivalent to PygPCQM4Mv2Dataset).

Writes histograms under ANALYSIS/pcqm4mv2_graph_stats/.
"""
from __future__ import annotations
import argparse
import os
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import default_rng


DATA_PT  = "datasets/pcqm4m-v2/processed/geometric_data_processed.pt"
SPLIT_PT = "datasets/pcqm4m-v2/split_dict.pt"
OUT_DIR  = "ANALYSIS/pcqm4mv2_graph_stats"


# ---------- data loading & splits ----------------------------------------

def load_dataset():
    data, slices = torch.load(DATA_PT, map_location="cpu", weights_only=False)
    split = torch.load(SPLIT_PT, weights_only=False)
    x_slices  = slices["x"].numpy().astype(np.int64)
    ei_slices = slices["edge_index"].numpy().astype(np.int64)
    edge_index = data.edge_index.numpy().astype(np.int64)   # local 0..n-1 ids
    return x_slices, ei_slices, edge_index, split


def splits(split):
    """Return (subtrain_idx, test_idx) matching master_loader.py (seed=42)."""
    rng = default_rng(seed=42)
    train_idx = rng.permutation(split["train"])[150000:]
    subtrain_idx = train_idx[: int(0.1 * len(train_idx))]
    test_idx = np.asarray(split["valid"])
    return np.asarray(subtrain_idx, dtype=np.int64), test_idx.astype(np.int64)


def edges_of(gi, x_s, ei_s, ei):
    """Return (n, u, v) for graph gi, with u<v for each undirected edge."""
    n  = int(x_s[gi + 1] - x_s[gi])
    lo = int(ei_s[gi]); hi = int(ei_s[gi + 1])
    row, col = ei[0, lo:hi], ei[1, lo:hi]
    mask = row < col
    return n, row[mask].astype(np.int64), col[mask].astype(np.int64)


# ---------- Part 1 — basic size stats ------------------------------------

def size_stats(name, idx, x_s, ei_s):
    n_nodes = (x_s[idx + 1] - x_s[idx]).astype(np.float64)
    n_edges = ((ei_s[idx + 1] - ei_s[idx]) / 2.0)  # undirected
    deg = np.where(n_nodes > 0, 2.0 * n_edges / np.clip(n_nodes, 1, None), 0.0)
    print(f"\n=== [sizes] {name} — {len(idx):,} graphs ===")
    print(f"  nodes/graph     : mean {n_nodes.mean():.2f} +- {n_nodes.std():.2f}"
          f"   (min {int(n_nodes.min())}, median {int(np.median(n_nodes))}, max {int(n_nodes.max())})")
    print(f"  edges/graph     : mean {n_edges.mean():.2f} +- {n_edges.std():.2f}"
          f"   (min {int(n_edges.min())}, median {int(np.median(n_edges))}, max {int(n_edges.max())})  [undirected]")
    print(f"  avg node degree : {deg.mean():.3f} +- {deg.std():.3f}")
    print(f"  totals          : {int(n_nodes.sum()):,} nodes / {int(n_edges.sum()):,} undirected edges")


# ---------- Part 2 — bridges (iterative Tarjan) --------------------------

def _count_bridges(n, adj):
    disc = [-1] * n; low = [0] * n; t = 0; bridges = 0
    for root in range(n):
        if disc[root] != -1: continue
        disc[root] = low[root] = t; t += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, peid, it = stack[-1]; advanced = False
            for v, eid in it:
                if eid == peid: continue
                if disc[v] == -1:
                    disc[v] = low[v] = t; t += 1
                    stack.append((v, eid, iter(adj[v])))
                    advanced = True; break
                else:
                    if disc[v] < low[u]:
                        low[u] = disc[v]; stack[-1] = (u, peid, it)
            if not advanced:
                stack.pop()
                if stack:
                    pu, _, _ = stack[-1]
                    if low[u] < low[pu]: low[pu] = low[u]
                    if low[u] > disc[pu]: bridges += 1
    return bridges


def _bridge_chunk(args):
    idx, x_s, ei_s, ei = args
    frac = np.empty(len(idx), dtype=np.float32)
    m_arr = np.empty(len(idx), dtype=np.int32)
    nb_arr = np.empty(len(idx), dtype=np.int32)
    for k, gi in enumerate(idx):
        n, u, v = edges_of(gi, x_s, ei_s, ei)
        m = u.size; m_arr[k] = m
        if m == 0 or n == 0:
            frac[k] = np.nan; nb_arr[k] = 0; continue
        adj = [[] for _ in range(n)]
        for eid in range(m):
            adj[int(u[eid])].append((int(v[eid]), eid))
            adj[int(v[eid])].append((int(u[eid]), eid))
        br = _count_bridges(n, adj)
        nb_arr[k] = m - br
        frac[k] = (m - br) / m
    return frac, m_arr, nb_arr


def bridge_stats(name, idx, x_s, ei_s, ei, workers):
    print(f"\n=== [bridges] {name} — {len(idx):,} graphs (workers={workers}) ===")
    t0 = time.time()
    chunks = np.array_split(idx, max(workers * 4, 1))
    args = [(c, x_s, ei_s, ei) for c in chunks]
    with Pool(workers) as p:
        results = p.map(_bridge_chunk, args)
    frac = np.concatenate([r[0] for r in results])
    m    = np.concatenate([r[1] for r in results])
    nb   = np.concatenate([r[2] for r in results])
    print(f"  ({time.time()-t0:.1f}s)")

    valid = ~np.isnan(frac); f = frac[valid]
    tot_m = int(m[valid].sum()); tot_nb = int(nb[valid].sum())
    print(f"  non-bridge fraction / graph : mean {f.mean()*100:.2f}% +- {f.std()*100:.2f}%  "
          f"(median {np.median(f)*100:.2f}%)")
    print(f"  pure trees (0% non-bridge)  : {int((f == 0).sum()):,}  ({(f == 0).mean()*100:.2f}%)")
    print(f"  graphs with >=1 cycle       : {int((f > 0).sum()):,}  ({(f > 0).mean()*100:.2f}%)")
    print(f"  pooled over all edges       : {tot_nb:,}/{tot_m:,} = {tot_nb/tot_m*100:.2f}% non-bridge")


# ---------- Part 3 — Laplacian eigenvalue gaps ---------------------------

K_MAX      = 21
K_VALUES   = (8, 21)
GAP_THRESH = (1e-10, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1)


def _gap_chunk(args):
    """Return a structured result per graph: all eigenvalues up to K_MAX (padded)."""
    idx, x_s, ei_s, ei = args
    N = len(idx)
    eigs = np.full((N, K_MAX), np.nan, dtype=np.float64)
    keep = np.zeros(N, dtype=bool)
    for k, gi in enumerate(idx):
        n, u, v = edges_of(gi, x_s, ei_s, ei)
        if n < 3: continue
        A = np.zeros((n, n), dtype=np.float64)
        A[u, v] = 1.0; A[v, u] = 1.0
        L = np.diag(A.sum(axis=1)) - A
        w = np.linalg.eigvalsh(L)
        w = np.clip(w, 0.0, None)
        w.sort()
        take = min(K_MAX, n)
        eigs[k, :take] = w[:take]
        keep[k] = True
    return keep, eigs


def _summarise_deltamin(label, dmin):
    dmin = dmin[~np.isnan(dmin)]
    q = np.percentile(dmin, [5, 10, 25, 50, 75, 90, 95])
    print(f"  [{label}] N={len(dmin):,}  "
          f"mean {dmin.mean():.4g}  median {np.median(dmin):.4g}  std {dmin.std():.4g}  "
          f"min {dmin.min():.4g}  max {dmin.max():.4g}")
    print(f"         percentiles  5/10/25/50/75/90/95 : "
          + "  ".join(f"{x:.4g}" for x in q))
    for th in (1e-3, 1e-2, 5e-2, 1e-1, 2e-1):
        print(f"         P(delta_min < {th:<6g}) = {(dmin < th).mean()*100:6.2f}%")


def laplacian_gap_stats(name, idx, x_s, ei_s, ei, workers, out_dir):
    print(f"\n=== [Laplacian gaps] {name} — {len(idx):,} graphs (workers={workers}, k={K_MAX}) ===")
    t0 = time.time()
    chunks = np.array_split(idx, max(workers * 4, 1))
    args = [(c, x_s, ei_s, ei) for c in chunks]
    with Pool(workers) as p:
        results = p.map(_gap_chunk, args)
    keep = np.concatenate([r[0] for r in results])
    eigs = np.concatenate([r[1] for r in results], axis=0)
    print(f"  eigendecompositions done in {time.time()-t0:.1f}s  "
          f"(kept {keep.sum():,} / {len(keep):,} graphs with n>=3)")

    # Skip lambda_0; compute consecutive gaps on lambda_1..lambda_{k-1}.
    dmin_per_k = {}
    for k in K_VALUES:
        g = eigs[keep, 1:k]                            # (N, k-1) possibly NaN-padded
        diffs = np.diff(g, axis=1)                     # (N, k-2)
        dmin_per_k[k] = np.nanmin(diffs, axis=1)

    # Extra counts (done at K_MAX, the full set).
    g_full   = eigs[keep, 1:K_MAX]
    diffs_f  = np.diff(g_full, axis=1)
    # per-graph counts of small gaps (ignore NaNs from padding)
    counts = {}
    for th in GAP_THRESH:
        counts[th] = np.nansum(diffs_f < th, axis=1).astype(np.int64)
    total_eig = np.isfinite(eigs[keep]).sum(axis=1)

    print("\n  delta_min summaries:")
    for k in K_VALUES:
        _summarise_deltamin(f"k={k}", dmin_per_k[k])

    print(f"\n  fraction with >=1 exact multiplicity (gap < 1e-10) : "
          f"{(counts[1e-10] > 0).mean()*100:.2f}%")
    near = counts[1e-2] > 0
    mean_near = counts[1e-2][near].mean() if near.any() else 0.0
    print(f"  fraction with >=1 near-degenerate (gap < 0.01)   : "
          f"{near.mean()*100:.2f}%  "
          f"(mean #near-degenerate pairs among those = {mean_near:.2f})")
    print(f"  average #eigenvalues computed per graph          : {total_eig.mean():.2f}")

    os.makedirs(out_dir, exist_ok=True)

    # (a) delta_min histograms (log-x), k=8 and k=21 overlaid
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.logspace(-8, 1, 60)
    for k, color in zip(K_VALUES, ("#377eb8", "#e41a1c")):
        d = dmin_per_k[k]; d = d[np.isfinite(d)]
        d = np.clip(d, bins[0], None)
        ax.hist(d, bins=bins, histtype="step", linewidth=1.6,
                color=color, label=f"k={k}  (median={np.median(d):.3g})")
    ax.set_xscale("log"); ax.set_xlabel(r"$\delta_{\min}$  (smallest consecutive gap in $\lambda_1\ldots\lambda_{k-1}$)")
    ax.set_ylabel("# graphs"); ax.set_title(f"{name}: Laplacian $\\delta_{{\\min}}$")
    ax.grid(alpha=0.3); ax.legend()
    p1 = os.path.join(out_dir, "delta_min_hist.png")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"\n  saved {p1}")

    # (b) histogram of ALL consecutive gaps (across all graphs) at k=K_MAX
    all_gaps = diffs_f[np.isfinite(diffs_f)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.logspace(-8, 1, 60)
    ax.hist(np.clip(all_gaps, bins[0], None), bins=bins,
            color="#4daf4a", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log"); ax.set_xlabel(r"consecutive gap $\lambda_{i+1}-\lambda_i$  ($i\geq 1$)")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: all consecutive Laplacian gaps (k up to {K_MAX})")
    ax.grid(alpha=0.3)
    p2 = os.path.join(out_dir, "all_gaps_hist.png")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"  saved {p2}")


# ---------- main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sizes",   action="store_true")
    ap.add_argument("--skip-bridges", action="store_true")
    ap.add_argument("--skip-laplacian", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(cpu_count(), 8)))
    args = ap.parse_args()

    print("Loading processed PCQM4Mv2 ...")
    x_s, ei_s, ei, split = load_dataset()
    subtrain_idx, test_idx = splits(split)
    print(f"  subtrain (10%): {len(subtrain_idx):,} graphs  |  "
          f"test (OGB valid): {len(test_idx):,} graphs")

    if not args.skip_sizes:
        size_stats("10% train subset", subtrain_idx, x_s, ei_s)
        size_stats("test (OGB valid)",  test_idx,     x_s, ei_s)

    if not args.skip_bridges:
        bridge_stats("10% train subset", subtrain_idx, x_s, ei_s, ei, args.workers)
        bridge_stats("test (OGB valid)",  test_idx,     x_s, ei_s, ei, args.workers)

    if not args.skip_laplacian:
        laplacian_gap_stats("test (OGB valid)", test_idx, x_s, ei_s, ei,
                            args.workers, OUT_DIR)


if __name__ == "__main__":
    main()
