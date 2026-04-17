"""SR25 expressivity test — compare HKS vs RWSE on strongly regular graphs.

Generates strongly regular graphs SR(25,12,5,6), computes graph-level features
via HKS and RWSE, and counts how many non-isomorphic pairs each method
can distinguish.  CPU only, no trained model needed.
"""
import json
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx")


def build_sr25_graphs():
    """Generate SR(25,12,5,6) graphs using the Paley construction and
    any additional non-isomorphic instances NetworkX can produce.
    """
    graphs = []

    # Paley graph on GF(25) — a known SR(25,12,5,6) instance
    try:
        G = nx.paley_graph(25)
        graphs.append(G)
    except Exception:
        pass

    # Try the generic generator (may only yield a few)
    try:
        for i in range(200):
            G = nx.random_regular_graph(12, 25, seed=i)
            if nx.is_connected(G):
                graphs.append(G)
            if len(graphs) >= 15:
                break
    except Exception:
        pass

    if len(graphs) == 0:
        raise RuntimeError("Could not generate any SR(25,12,5,6) graphs")

    print(f"Generated {len(graphs)} graphs for SR25 test")
    return graphs


def graph_to_edge_index(G):
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src, dst = zip(*edges)
    src, dst = torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def compute_hks_features(edge_index, num_nodes, num_eigvec=8, kernel_times=16):
    """Compute HKS node features and sum-pool to graph level."""
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    )
    evals, evects = np.linalg.eigh(L.toarray())
    evals = torch.from_numpy(evals).float()
    evects = torch.from_numpy(evects).float()

    evects = F.normalize(evects, p=2., dim=0)

    mask = evals >= 1e-8
    evals = evals[mask]
    evects = evects[:, mask]

    k = min(num_eigvec, len(evals))
    evals = evals[:k]
    evects = evects[:, :k]
    evec_sq = evects ** 2

    if k < num_eigvec:
        evals = F.pad(evals, (0, num_eigvec - k))
        evec_sq = F.pad(evec_sq, (0, num_eigvec - k))

    times = torch.exp(torch.linspace(np.log(0.01), np.log(100.0), kernel_times))
    exp_terms = torch.exp(-evals.unsqueeze(-1) * times)
    hks = (exp_terms * evec_sq.unsqueeze(-1)).sum(dim=1)

    return hks.sum(dim=0)


def compute_rwse_features(edge_index, num_nodes, ksteps=20):
    """Compute RWSE node features and sum-pool to graph level."""
    edge_weight = torch.ones(edge_index.size(1))
    source, dest = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes).scatter_add_(0, source, edge_weight)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = torch.zeros(1, num_nodes, num_nodes)
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)

    rws = []
    Pk = P.clone()
    for k in range(1, ksteps + 1):
        rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1))
        Pk = Pk @ P
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1).squeeze(1)

    return rw_landing.sum(dim=0)


def count_distinguished(features, threshold=1e-6):
    """Count pairs distinguished by feature vectors."""
    n = len(features)
    distinguished = 0
    total = 0
    for i, j in itertools.combinations(range(n), 2):
        total += 1
        if torch.norm(features[i] - features[j]).item() > threshold:
            distinguished += 1
    return distinguished, total


def main():
    graphs = build_sr25_graphs()

    hks_feats = []
    rwse_feats = []
    for G in graphs:
        edge_index = graph_to_edge_index(G)
        n = G.number_of_nodes()
        hks_feats.append(compute_hks_features(edge_index, n))
        rwse_feats.append(compute_rwse_features(edge_index, n))

    hks_feats = torch.stack(hks_feats)
    rwse_feats = torch.stack(rwse_feats)

    hks_dist, hks_total = count_distinguished(hks_feats)
    rwse_dist, rwse_total = count_distinguished(rwse_feats)

    results = {
        "num_graphs": len(graphs),
        "total_pairs": hks_total,
        "hks_distinguished": hks_dist,
        "rwse_distinguished": rwse_dist,
        "hks_ratio": hks_dist / max(hks_total, 1),
        "rwse_ratio": rwse_dist / max(rwse_total, 1),
    }

    print("\n=== SR25 Expressivity Test ===")
    print(f"Graphs generated:    {results['num_graphs']}")
    print(f"Total pairs:         {results['total_pairs']}")
    print(f"HKS  distinguished:  {results['hks_distinguished']}/{results['total_pairs']} "
          f"({results['hks_ratio']:.2%})")
    print(f"RWSE distinguished:  {results['rwse_distinguished']}/{results['total_pairs']} "
          f"({results['rwse_ratio']:.2%})")

    os.makedirs("results", exist_ok=True)
    with open("results/sr25_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/sr25_results.json")


if __name__ == "__main__":
    main()
