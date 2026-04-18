"""Eigenvalue-gap histogram for PCQM4Mv2.

For every (or a random subset of) graph in PCQM4Mv2 we compute

  Delta_lambda_min = min gap between consecutive non-zero eigenvalues
                     of the graph Laplacian,

using the exact same convention as `experiments/run_stability.py`
(`compute_eigval_gap`). We do this for BOTH:

  * the unnormalized Laplacian  L = D - A          (matches LapPE cfg
                                                   `laplacian_norm: none`)
  * the symmetric normalized Laplacian
    L_sym = I - D^{-1/2} A D^{-1/2}                (eigvals bounded in [0, 2])

The normalized version makes "< 0.1" a scale-invariant threshold, so it
is the right number to quote when motivating LapPE instability.  Both
are saved and both are plotted.

Usage:
    # Same 10% training slice as GraphGPS ``PCQM4Mv2-subset`` (``master_loader``):
    python experiments/run_gap_histogram.py --split graphgps-subset-train \\
        --num-graphs -1 --workers 0

    # Random global sample (not the paper subset):
    python experiments/run_gap_histogram.py --split random --num-graphs 20000

    # Full OGB train+valid+test (very large):
    python experiments/run_gap_histogram.py --split full --num-graphs -1 --workers 0

Outputs:
    results/pcqm4mv2_gaps.npz                 # raw arrays + sizes
    figures/figure0_pcqm4mv2_gap_hist.pdf     # figure for the paper
    figures/figure0_pcqm4mv2_gap_hist.png     # quick-view version
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

# Only import torch/PyG lazily inside main so `-h` is snappy.


THRESHOLDS = (0.5, 0.1, 0.05, 0.01, 1e-3, 1e-4)
EPS = 1e-8

# Set in main() before fork-pool workers read graphs by index (ordered imap).
_MP_DATASET = None
_MP_INDICES = None


def graphgps_pcqm4mv2_subset_train_indices(dataset) -> np.ndarray:
    """Indices into full PygPCQM4Mv2 for the 10% training subset.

    Mirrors ``graphgps.loader.master_loader.preformat_OGB_PCQM4Mv2`` with
    ``name == 'subset'``: permute original train with ``numpy.random.default_rng(42)``,
    hold out 150k as (internal) valid pool, then take the first 10% of the
    remaining train indices.
    """
    import torch

    split_idx = dataset.get_idx_split()
    rng = np.random.default_rng(42)
    train_perm = rng.permutation(split_idx["train"].numpy())
    train_idx = torch.from_numpy(train_perm.astype(np.int64, copy=False))
    _valid_pool, train_idx = train_idx[:150000], train_idx[150000:]
    subset_ratio = 0.1
    n_sub = int(subset_ratio * len(train_idx))
    subtrain = train_idx[:n_sub].numpy().astype(np.int64, copy=False)
    return subtrain


def _mp_compute_at_i(i: int):
    """Fork-pool worker: row i of the job table ``_MP_INDICES``."""
    idx = int(_MP_INDICES[i])
    data = _MP_DATASET[idx]
    ei = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    N = int(data.num_nodes)
    return _compute_one((ei, N))


def _laplacians_from_edges(
    edge_index: np.ndarray, num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build dense unnormalized + symmetric-normalized Laplacians (float64).

    Matches the conventions of torch_geometric.utils.get_laplacian for
    `normalization in {None, 'sym'}` on an undirected graph.
    """
    N = int(num_nodes)
    A = np.zeros((N, N), dtype=np.float64)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        # Deduplicate (u, v) pairs and drop self-loops before symmetrizing
        mask = src != dst
        src, dst = src[mask], dst[mask]
        A[src, dst] = 1.0
        A[dst, src] = 1.0

    deg = A.sum(axis=1)
    L_un = np.diag(deg) - A

    # D^{-1/2} with safe handling of isolated nodes (deg == 0).
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    # Force symmetry to kill rounding noise before eigh.
    L_sym = 0.5 * (L_sym + L_sym.T)
    return L_un, L_sym


def _min_consecutive_nonzero_gap(evals: np.ndarray) -> float:
    """Delta_lambda_min := min_i (lambda_{i+1} - lambda_i) over non-zero eigs.

    Matches `experiments/run_stability.py::compute_eigval_gap`.
    Returns NaN if fewer than two non-zero eigenvalues exist.
    """
    evals = np.sort(evals)
    nonzero = evals[evals > EPS]
    if nonzero.size < 2:
        return float("nan")
    return float(np.diff(nonzero).min())


def _compute_one(args):
    """Worker: (edge_index_np, num_nodes) -> (gap_un, gap_sym, N)."""
    edge_index, num_nodes = args
    if num_nodes < 2:
        return float("nan"), float("nan"), int(num_nodes)
    L_un, L_sym = _laplacians_from_edges(edge_index, num_nodes)
    evals_un = np.linalg.eigvalsh(L_un)
    evals_sym = np.linalg.eigvalsh(L_sym)
    return (
        _min_consecutive_nonzero_gap(evals_un),
        _min_consecutive_nonzero_gap(evals_sym),
        int(num_nodes),
    )


def _iter_tasks(dataset, indices):
    """Yield (edge_index_np, num_nodes) without pickling whole Data objects."""
    for idx in indices:
        data = dataset[int(idx)]
        ei = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        N = int(data.num_nodes)
        yield ei, N


def _report_hist(label: str, gaps: np.ndarray) -> None:
    finite = gaps[np.isfinite(gaps)]
    print(f"\n  [{label}]  n_graphs = {finite.size}")
    if finite.size == 0:
        return
    qs = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    print(
        "    quantiles  1%={:.3e}  5%={:.3e}  50%={:.3e}  95%={:.3e}  99%={:.3e}".format(
            *qs
        )
    )
    for t in THRESHOLDS:
        frac = float((finite < t).mean())
        print(f"    P(Delta_lambda_min < {t:<7g}) = {frac * 100:6.2f}%")


def _plot(
    gaps_un: np.ndarray,
    gaps_sym: np.ndarray,
    sizes: np.ndarray,
    out_pdf: str,
    out_png: str,
    dataset_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_un = gaps_un[np.isfinite(gaps_un)]
    finite_sym = gaps_sym[np.isfinite(gaps_sym)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)

    panels = [
        ("Unnormalized  L = D - A", finite_un, axes[0]),
        (r"Normalized  $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$",
         finite_sym, axes[1]),
    ]

    for title, arr, ax in panels:
        # Log-spaced bins from the 0.5th percentile to the max.
        if arr.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue
        lo = max(np.quantile(arr, 0.005), 1e-8)
        hi = max(arr.max(), lo * 10)
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.hist(arr, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Delta\lambda_{\min}$  (min gap of non-zero eigvals)")
        ax.set_ylabel("number of graphs")

        med = float(np.median(arr))
        frac01 = float((arr < 0.1).mean()) * 100
        frac001 = float((arr < 0.01).mean()) * 100

        ax.axvline(med, color="black", linestyle="--", linewidth=1.0,
                   label=f"median = {med:.3g}")
        ax.axvline(0.1, color="#d94b3b", linestyle=":", linewidth=1.2,
                   label=f"0.1  ({frac01:.1f}% below)")
        ax.axvline(0.01, color="#d94b3b", linestyle=":", linewidth=1.0, alpha=0.6,
                   label=f"0.01 ({frac001:.1f}% below)")

        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"{dataset_label}: distribution of minimum eigenvalue gaps  "
        f"(N = {gaps_un.size} graphs, median graph size = {int(np.median(sizes))} nodes)",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n  wrote {out_pdf}")
    print(f"  wrote {out_png}")


def load_dataset(dataset_dir: str):
    """Return a PyG dataset object for PCQM4Mv2 (raw OGB-LSC, no split)."""
    try:
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        print("ERROR: could not import PygPCQM4Mv2Dataset from ogb.lsc. "
              "Make sure `ogb` (and rdkit) is installed.", file=sys.stderr)
        raise
    t0 = time.time()
    print(f"Loading PCQM4Mv2 from {dataset_dir} ...")
    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    print(f"  dataset size = {len(dataset):,} graphs  "
          f"(loaded in {time.time() - t0:.1f}s)")
    return dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="datasets",
                   help="root passed to PygPCQM4Mv2Dataset (OGB will look "
                        "for `<dir>/pcqm4m-v2/`; default: datasets)")
    p.add_argument(
        "--split",
        choices=("graphgps-subset-train", "random", "full"),
        default="graphgps-subset-train",
        help="which graphs: GraphGPS PCQM4Mv2-subset 10% train slice, "
             "random global sample, or all graphs in OGB order",
    )
    p.add_argument("--num-graphs", type=int, default=-1,
                   help="cap / sample size: with --split random, draw this many "
                        "uniformly at random; with graphgps-subset-train, if "
                        "positive randomly subsample that many from the 10% "
                        "train slice; -1 means use every graph in the chosen "
                        "split (default -1)",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the random subset")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel processes (0 = all CPUs, 1 = no multiproc)")
    p.add_argument("--out-data", default="results/pcqm4mv2_gaps.npz")
    p.add_argument("--out-pdf", default="figures/figure0_pcqm4mv2_gap_hist.pdf")
    p.add_argument("--out-png", default="figures/figure0_pcqm4mv2_gap_hist.png")
    p.add_argument("--dataset-label", default=None,
                   help="figure title (default: derived from --split)")
    args = p.parse_args()

    sys.path.insert(0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset = load_dataset(args.dataset_dir)
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)

    if args.split == "full":
        pool_idx = np.arange(n_total, dtype=np.int64)
    elif args.split == "graphgps-subset-train":
        pool_idx = graphgps_pcqm4mv2_subset_train_indices(dataset)
        print(
            f"GraphGPS PCQM4Mv2-subset 10% train pool: {len(pool_idx):,} graphs "
            f"(of {n_total:,} in full OGB release)"
        )
    else:
        pool_idx = np.arange(n_total, dtype=np.int64)

    if args.split == "random":
        if args.num_graphs < 0 or args.num_graphs > len(pool_idx):
            indices = pool_idx
        else:
            indices = rng.choice(pool_idx, size=args.num_graphs, replace=False)
            indices.sort()
    else:
        if args.num_graphs > 0 and args.num_graphs < len(pool_idx):
            indices = rng.choice(pool_idx, size=args.num_graphs, replace=False)
            indices.sort()
        else:
            indices = pool_idx

    if args.dataset_label is None:
        if args.split == "graphgps-subset-train":
            args.dataset_label = "PCQM4Mv2 (GraphGPS 10% train subset)"
        elif args.split == "random":
            args.dataset_label = "PCQM4Mv2 (random sample)"
        else:
            args.dataset_label = "PCQM4Mv2 (full OGB)"

    print(f"Processing {len(indices):,} graphs "
          f"(split={args.split}, of {n_total:,} full release) ...")

    gaps_un = np.empty(indices.size, dtype=np.float64)
    gaps_sym = np.empty(indices.size, dtype=np.float64)
    sizes = np.empty(indices.size, dtype=np.int32)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=indices.size, smoothing=0.1, unit="graph")
    except Exception:
        pbar = None

    t0 = time.time()
    n_workers = args.workers if args.workers != 0 else (os.cpu_count() or 1)

    if n_workers == 1:
        for i, task in enumerate(_iter_tasks(dataset, indices)):
            g_un, g_sym, N = _compute_one(task)
            gaps_un[i] = g_un
            gaps_sym[i] = g_sym
            sizes[i] = N
            if pbar is not None:
                pbar.update(1)
    else:
        import multiprocessing as mp

        global _MP_DATASET, _MP_INDICES
        _MP_DATASET = dataset
        _MP_INDICES = indices
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            chunksize = max(1, len(indices) // (n_workers * 32))
            for i, (g_un, g_sym, N) in enumerate(
                pool.imap(_mp_compute_at_i, range(len(indices)), chunksize=chunksize)
            ):
                gaps_un[i] = g_un
                gaps_sym[i] = g_sym
                sizes[i] = N
                if pbar is not None:
                    pbar.update(1)
        _MP_DATASET = None
        _MP_INDICES = None

    if pbar is not None:
        pbar.close()
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s "
          f"({indices.size / max(dt, 1e-6):.0f} graphs/s, "
          f"workers={n_workers})")

    _report_hist("unnormalized L", gaps_un)
    _report_hist("normalized L_sym", gaps_sym)

    os.makedirs(os.path.dirname(args.out_data) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_data,
        gaps_unnormalized=gaps_un,
        gaps_normalized=gaps_sym,
        num_nodes=sizes,
        indices=indices,
        split=np.array([args.split]),
    )
    print(f"\n  wrote {args.out_data}")

    _plot(
        gaps_un=gaps_un,
        gaps_sym=gaps_sym,
        sizes=sizes,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
        dataset_label=args.dataset_label,
    )


if __name__ == "__main__":
    main()
