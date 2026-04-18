"""
PE-Level Stability Experiment.

Measures how much LapPE and HKS positional encodings change when a graph is
slightly perturbed, and relates the change to the graph's minimum non-zero
eigenvalue gap.

Expectation:
    * LapPE change correlates strongly (negatively) with small spectral gaps.
    * HKS change is largely insensitive to the spectral gap.

Outputs (written next to this script):
    - pe_stability_scatter.png   (Figure A - main figure)
    - pe_stability_lines.png     (Figure B - secondary)
    - pe_stability_results.json  (raw per-graph/per-epsilon numbers)
    - pe_stability_summary.json  (aggregate key numbers)
"""

from __future__ import annotations

import json
import os
import time
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
K_PE = 8
TIMES = np.exp(np.linspace(np.log(0.01), np.log(100), 16))  # matches L-HKS init
EPSILONS = [0.01, 0.05, 0.10]
M_PERTURBATIONS = 30


# --------------------------------------------------------------------------- #
# Step 1: Graph generation                                                    #
# --------------------------------------------------------------------------- #

def generate_graphs() -> List[nx.Graph]:
    graphs: List[nx.Graph] = []

    # Cycles (highly degenerate spectra, very small gaps)
    for n in range(10, 51):
        graphs.append(nx.cycle_graph(n))

    # Grids (moderate degeneracy)
    for r in range(3, 11):
        for c in range(r, 11):
            G = nx.grid_2d_graph(r, c)
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)

    # Erdos-Renyi (variable gaps)
    for seed in range(300):
        rs = np.random.RandomState(seed)
        n = rs.randint(15, 41)
        p = rs.uniform(0.1, 0.5)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        s = seed
        while not nx.is_connected(G):
            s += 1000
            G = nx.erdos_renyi_graph(n, p, seed=s)
        graphs.append(G)

    # Named regular graphs (extremely small gaps)
    graphs.append(nx.petersen_graph())
    graphs.append(nx.dodecahedral_graph())
    graphs.append(nx.icosahedral_graph())
    # Cuboctahedral graph was added in newer networkx; fall back gracefully.
    if hasattr(nx, "cuboctahedral_graph"):
        graphs.append(nx.cuboctahedral_graph())

    # Make sure every graph has consecutive integer node labels.
    graphs = [nx.convert_node_labels_to_integers(G) for G in graphs]
    return graphs


# --------------------------------------------------------------------------- #
# Step 2: Eigendecomposition and gap                                          #
# --------------------------------------------------------------------------- #

def compute_eigen(G: nx.Graph):
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals, eigvecs = eigh(L)
    return eigvals, eigvecs, L


def min_gap(eigvals: np.ndarray) -> float:
    nonzero = eigvals[eigvals > 1e-10]
    if len(nonzero) < 2:
        return float("inf")
    sorted_eigs = np.sort(nonzero)
    gaps = np.diff(sorted_eigs)
    return float(gaps.min())


# --------------------------------------------------------------------------- #
# Step 3: PE computations                                                     #
# --------------------------------------------------------------------------- #

def compute_lappe(eigvals: np.ndarray, eigvecs: np.ndarray, k: int = K_PE) -> np.ndarray:
    """Returns [n, k'] matrix of the first k non-trivial eigenvectors."""
    idx = np.argsort(eigvals)
    start = 1  # skip the zero eigenvalue
    end = min(start + k, len(eigvals))
    return eigvecs[:, idx[start:end]]


def compute_hks(eigvals: np.ndarray, eigvecs: np.ndarray, times: np.ndarray,
                k: int = K_PE) -> np.ndarray:
    """Returns [n, len(times)] matrix of HKS values using the lowest k eigenpairs."""
    idx = np.argsort(eigvals)
    k_eff = min(k, len(eigvals))
    evals = eigvals[idx[:k_eff]]
    evecs = eigvecs[:, idx[:k_eff]]

    n = evecs.shape[0]
    K = len(times)
    hks = np.zeros((n, K))
    for j, t in enumerate(times):
        weights = np.exp(-evals * t)
        hks[:, j] = (evecs ** 2) @ weights
    return hks


# --------------------------------------------------------------------------- #
# Step 4: Perturbation                                                        #
# --------------------------------------------------------------------------- #

def perturb_graph(G: nx.Graph, epsilon: float, rng: np.random.RandomState) -> nx.Graph:
    """Remove each existing edge w.p. epsilon; add non-edges to keep expected |E|."""
    n = G.number_of_nodes()
    edges = set(tuple(sorted(e)) for e in G.edges())
    total_possible = n * (n - 1) // 2
    non_edges_count = total_possible - len(edges)
    add_prob = epsilon * len(edges) / max(non_edges_count, 1)

    new_edges = set()
    for e in edges:
        if rng.random() > epsilon:
            new_edges.add(e)

    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edges:
                continue
            if rng.random() < add_prob:
                new_edges.add((u, v))

    G_new = nx.Graph()
    G_new.add_nodes_from(range(n))
    G_new.add_edges_from(new_edges)
    return G_new


# --------------------------------------------------------------------------- #
# Step 5: Change metrics                                                      #
# --------------------------------------------------------------------------- #

def lappe_distance(U_orig: np.ndarray, U_pert: np.ndarray) -> float:
    """Sign-aligned, column-wise, size-normalised Frobenius distance."""
    k = min(U_orig.shape[1], U_pert.shape[1])
    total = 0.0
    for i in range(k):
        d_plus = np.linalg.norm(U_orig[:, i] - U_pert[:, i])
        d_minus = np.linalg.norm(U_orig[:, i] + U_pert[:, i])
        total += min(d_plus, d_minus) ** 2
    return float(np.sqrt(total) / np.sqrt(U_orig.shape[0]))


def hks_distance(hks_orig: np.ndarray, hks_pert: np.ndarray) -> float:
    return float(
        np.linalg.norm(hks_orig - hks_pert) / np.sqrt(hks_orig.shape[0])
    )


# --------------------------------------------------------------------------- #
# Step 6: Main loop                                                           #
# --------------------------------------------------------------------------- #

def run_experiment() -> list:
    graphs = generate_graphs()
    print(f"[pe-stability] Generated {len(graphs)} graphs")

    results = []
    t0 = time.time()

    for graph_idx, G in enumerate(graphs):
        eigvals, eigvecs, L = compute_eigen(G)
        gap = min_gap(eigvals)
        n = G.number_of_nodes()

        lappe_orig = compute_lappe(eigvals, eigvecs, k=K_PE)
        hks_orig = compute_hks(eigvals, eigvecs, TIMES, k=K_PE)

        for eps in EPSILONS:
            lappe_changes = []
            hks_changes = []

            for m in range(M_PERTURBATIONS):
                rng = np.random.RandomState(42 + graph_idx * 1000 + m)
                G_pert = perturb_graph(G, eps, rng)

                if G_pert.number_of_edges() == 0:
                    continue
                if not nx.is_connected(G_pert):
                    continue

                eigvals_p, eigvecs_p, L_p = compute_eigen(G_pert)
                lappe_pert = compute_lappe(eigvals_p, eigvecs_p, k=K_PE)
                hks_pert = compute_hks(eigvals_p, eigvecs_p, TIMES, k=K_PE)

                k_common = min(lappe_orig.shape[1], lappe_pert.shape[1])
                if k_common == 0:
                    continue

                # Normalise by spectral norm of the perturbation E = L - L'
                # so that we measure "PE change per unit perturbation",
                # isolating the sensitivity predicted by the theorems.
                norm_E = float(np.linalg.norm(L - L_p, ord=2))
                if norm_E < 1e-12:
                    continue

                lappe_changes.append(
                    lappe_distance(lappe_orig[:, :k_common], lappe_pert[:, :k_common])
                    / norm_E
                )
                hks_changes.append(hks_distance(hks_orig, hks_pert) / norm_E)

            if len(lappe_changes) == 0:
                continue

            results.append({
                "graph_idx": graph_idx,
                "n_nodes": n,
                "eigval_gap": gap,
                "epsilon": eps,
                "n_valid_perturbations": len(lappe_changes),
                "lappe_mean": float(np.mean(lappe_changes)),
                "lappe_std": float(np.std(lappe_changes)),
                "hks_mean": float(np.mean(hks_changes)),
                "hks_std": float(np.std(hks_changes)),
            })

        if (graph_idx + 1) % 25 == 0 or graph_idx == len(graphs) - 1:
            elapsed = time.time() - t0
            print(
                f"[pe-stability] {graph_idx + 1}/{len(graphs)} graphs "
                f"({elapsed:.1f}s, {elapsed / (graph_idx + 1):.2f}s/graph)"
            )

    return results


# --------------------------------------------------------------------------- #
# Step 7: Plots and summary                                                   #
# --------------------------------------------------------------------------- #

def plot_scatter(results: list, eps_target: float = 0.05) -> None:
    df = [r for r in results if abs(r["epsilon"] - eps_target) < 1e-9]
    gaps = np.array([r["eigval_gap"] for r in df], dtype=float)
    lappe = np.array([r["lappe_mean"] for r in df], dtype=float)
    hks = np.array([r["hks_mean"] for r in df], dtype=float)

    finite = np.isfinite(gaps) & (gaps > 0)
    gaps_f = gaps[finite]
    lappe_f = lappe[finite]
    hks_f = hks[finite]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(gaps_f, lappe_f, alpha=0.4, s=12,
               label="LapPE (sign-aligned)", c="red")
    ax.scatter(gaps_f, hks_f, alpha=0.4, s=12, label="HKS", c="blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Min eigenvalue gap $\Delta\lambda_{\min}$")
    ax.set_ylabel(r"Mean PE change per unit perturbation ($\|\Delta\mathrm{PE}\|/\|E\|_2$)")
    ax.set_title(f"PE stability at $\\varepsilon = {eps_target}$")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pe_stability_scatter.png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[pe-stability] Wrote {out}")


def plot_lines(results: list) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for method, label, color in [
        ("lappe_mean", "LapPE", "red"),
        ("hks_mean", "HKS", "blue"),
    ]:
        means = []
        for eps in EPSILONS:
            vals = [r[method] for r in results
                    if abs(r["epsilon"] - eps) < 1e-9]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        ax.plot(EPSILONS, means, "o-", label=label, color=color)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel(r"Mean PE change per unit perturbation ($\|\Delta\mathrm{PE}\|/\|E\|_2$)")
    ax.set_title("PE sensitivity vs perturbation level")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pe_stability_lines.png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[pe-stability] Wrote {out}")


def compute_summary(results: list, eps_target: float = 0.05) -> dict:
    df = [r for r in results if abs(r["epsilon"] - eps_target) < 1e-9]
    gaps = np.array([r["eigval_gap"] for r in df], dtype=float)
    lappe = np.array([r["lappe_mean"] for r in df], dtype=float)
    hks = np.array([r["hks_mean"] for r in df], dtype=float)

    pos = np.isfinite(gaps) & (gaps > 0) & (lappe > 0) & (hks > 0)
    log_gap = np.log(gaps[pos])
    log_lappe = np.log(lappe[pos])
    log_hks = np.log(hks[pos])

    r_lappe, p_lappe = pearsonr(log_gap, log_lappe) if log_gap.size > 2 else (float("nan"), float("nan"))
    r_hks, p_hks = pearsonr(log_gap, log_hks) if log_gap.size > 2 else (float("nan"), float("nan"))

    small_gap_mask = gaps < 0.01
    ratio_small_gap = (
        float(np.mean(lappe[small_gap_mask]) / np.mean(hks[small_gap_mask]))
        if small_gap_mask.sum() > 0 and np.mean(hks[small_gap_mask]) > 0
        else float("nan")
    )

    summary = {
        "epsilon": eps_target,
        "n_points_total": int(len(df)),
        "n_points_used_for_correlation": int(pos.sum()),
        "pearson_log_gap_vs_log_lappe": {"r": float(r_lappe), "p": float(p_lappe)},
        "pearson_log_gap_vs_log_hks": {"r": float(r_hks), "p": float(p_hks)},
        "n_graphs_with_small_gap_lt_0.01": int(small_gap_mask.sum()),
        "mean_lappe_small_gap": float(np.mean(lappe[small_gap_mask])) if small_gap_mask.any() else float("nan"),
        "mean_hks_small_gap": float(np.mean(hks[small_gap_mask])) if small_gap_mask.any() else float("nan"),
        "ratio_lappe_over_hks_small_gap": ratio_small_gap,
    }
    return summary


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    results = run_experiment()

    raw_path = os.path.join(OUT_DIR, "pe_stability_results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[pe-stability] Wrote {raw_path} ({len(results)} rows)")

    plot_scatter(results, eps_target=0.05)
    plot_lines(results)

    summary = compute_summary(results, eps_target=0.05)
    summary_path = os.path.join(OUT_DIR, "pe_stability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pe-stability] Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
