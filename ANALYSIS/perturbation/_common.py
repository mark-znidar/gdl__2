"""Shared utilities for pe_stability_exp1.py and pe_stability_exp2.py.

Keeps the two experiment scripts short and avoids touching anything under
``graphgps/``.  We reuse the codebase's PE computation code where possible:
``graphgps.transform.posenc_stats.compute_posenc_stats`` is called on
cloned Data objects so PE attributes are derived *exactly* as they were at
training time.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from numpy.random import default_rng
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# GraphGym / codebase hooks -- imports happen lazily to keep top-level import
# cheap (tests / smoke checks don't pay the `import graphgps` cost).
# ---------------------------------------------------------------------------

DATA_PT  = "datasets/pcqm4m-v2/processed/geometric_data_processed.pt"
SPLIT_PT = "datasets/pcqm4m-v2/split_dict.pt"

METHOD_PE_TYPE = {             # codebase PE-type key per method
    "LapPE":            "LapPE",
    "SignNet-MLP":      "SignNet",
    "SignNet-DeepSets": "SignNet",
    "L-HKS":            "LHKS",
    "fix-L-HKS":        "LHKS",
}
METHOD_IS_EIGENVECTOR = {      # True => eigenvector-based PE (not HKS)
    "LapPE": True, "SignNet-MLP": True, "SignNet-DeepSets": True,
    "L-HKS": False, "fix-L-HKS": False,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find(run_dir: str, pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No {pattern} found under {run_dir}")
    return hits[-1]


def _torch_load_checkpoint(path: str, *, map_location="cpu"):
    """Load a Lightning / GraphGym checkpoint dict.

    PyTorch >= 2.6 defaults ``torch.load(..., weights_only=True)``, which
    rejects these files; ``main.py`` patches ``torch.load`` when training,
    but analysis scripts are often run directly, so we set ``weights_only``
    explicitly here.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_ckpt(run_dir: str) -> str:
    hits = glob.glob(os.path.join(run_dir, "**", "*.ckpt"), recursive=True)
    hits += glob.glob(os.path.join(run_dir, "**", "*.pt"),   recursive=True)
    hits = [p for p in hits if "config" not in os.path.basename(p).lower()]
    if not hits:
        raise FileNotFoundError(f"No checkpoint (.ckpt/.pt) under {run_dir}")
    # Prefer the largest integer filename (final epoch) if the names parse.
    def key(p):
        try:    return int(Path(p).stem)
        except: return -1
    hits.sort(key=lambda p: (key(p), os.path.getmtime(p)))
    return hits[-1]


def load_model_and_cfg(run_dir: str, device: torch.device):
    """Build the GPSModel for a given run_dir and load its checkpoint weights.

    ``run_dir`` may point to the seed directory itself (containing
    ``config.yaml`` + ``ckpt/<epoch>.ckpt``) or anywhere higher up -- we
    recurse.  Returns ``(model, cfg, pe_type)``; ``cfg`` is the live global
    ``cfg`` node patched for inference.
    """
    import graphgps  # noqa: F401  (registers custom modules)
    from torch_geometric.graphgym.config import cfg, set_cfg
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.checkpoint import MODEL_STATE
    from graphgps.finetuning import set_new_cfg_allowed

    set_cfg(cfg)
    set_new_cfg_allowed(cfg, True)
    cfg.merge_from_file(_find(run_dir, "config.yaml"))
    cfg.share.dim_in  = 9                # raw OGB atom features
    cfg.share.dim_out = 1                # regression target
    cfg.wandb.use = False                # no wandb side effects
    cfg.train.auto_resume = False
    cfg.accelerator = "cuda" if device.type == "cuda" else "cpu"

    # Which PE type should be (re)computed for this method
    pe_type = None
    for key in cfg.keys():
        if key.startswith("posenc_"):
            if getattr(cfg, key).enable:
                pe_type = key.split("_", 1)[1]
                break
    if pe_type is None:
        raise RuntimeError(f"No posenc_*.enable=True in config for {run_dir}")

    model = create_model()
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    ckpt = _torch_load_checkpoint(_find_ckpt(run_dir), map_location="cpu")
    state = ckpt.get(MODEL_STATE, ckpt)
    first = next(iter(state.keys()))
    if first.startswith("model."):       # stored with graphgym's wrapper prefix
        try:   model.load_state_dict(state)
        except RuntimeError:
            state = {k[len("model."):]: v for k, v in state.items()}
            model.load_state_dict(state)
    else:
        try:   model.load_state_dict(state)
        except RuntimeError:
            state = {f"model.{k}": v for k, v in state.items()}
            model.load_state_dict(state)

    return model, cfg, pe_type


# ---------------------------------------------------------------------------
# PCQM4Mv2 test-set molecule access (single-graph Data objects)
# ---------------------------------------------------------------------------

@dataclass
class PCQMGraphs:
    data: "Data"                  # the collated dataset
    slices: Dict[str, np.ndarray] # per-key prefix sums
    test_idx: np.ndarray          # indices into ``data`` for OGB-valid split

    @classmethod
    def load(cls) -> "PCQMGraphs":
        data, slices = torch.load(DATA_PT, map_location="cpu", weights_only=False)
        split        = torch.load(SPLIT_PT, weights_only=False)
        slices_np = {k: v.numpy().astype(np.int64) for k, v in slices.items()}
        return cls(data=data, slices=slices_np,
                   test_idx=np.asarray(split["valid"]).astype(np.int64))

    def get(self, gi: int) -> Data:
        """Return an independent single-graph Data object (local 0..n-1 ids)."""
        s  = self.slices
        d  = self.data
        lo_x, hi_x  = int(s["x"][gi]),  int(s["x"][gi + 1])
        lo_e, hi_e  = int(s["edge_index"][gi]), int(s["edge_index"][gi + 1])
        lo_y, hi_y  = int(s["y"][gi]),  int(s["y"][gi + 1])
        data = Data(
            x          = d.x[lo_x:hi_x].clone(),
            edge_index = d.edge_index[:, lo_e:hi_e].clone(),
            edge_attr  = d.edge_attr[lo_e:hi_e].clone() if d.edge_attr is not None else None,
            y          = d.y[lo_y:hi_y].clone(),
            num_nodes  = hi_x - lo_x,
        )
        return data


# ---------------------------------------------------------------------------
# Graph utilities: Tarjan's bridges, dense Laplacian, gap stats
# ---------------------------------------------------------------------------

def undirected_edges(edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (u, v) with u < v for each undirected edge represented once."""
    r = edge_index[0].numpy(); c = edge_index[1].numpy()
    m = r < c
    return r[m], c[m]


def find_non_bridge_edges(edge_index: torch.Tensor, num_nodes: int) \
        -> List[Tuple[int, int]]:
    """Iterative Tarjan's algorithm. Returns undirected non-bridge edges (u<v)."""
    u, v = undirected_edges(edge_index)
    m = u.shape[0]
    if m == 0: return []
    adj = [[] for _ in range(num_nodes)]
    for eid in range(m):
        adj[int(u[eid])].append((int(v[eid]), eid))
        adj[int(v[eid])].append((int(u[eid]), eid))
    disc = [-1] * num_nodes; low = [0] * num_nodes; t = 0
    is_bridge = [False] * m
    for root in range(num_nodes):
        if disc[root] != -1: continue
        disc[root] = low[root] = t; t += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            ux, peid, it = stack[-1]; advanced = False
            for vx, eid in it:
                if eid == peid: continue
                if disc[vx] == -1:
                    disc[vx] = low[vx] = t; t += 1
                    stack.append((vx, eid, iter(adj[vx])))
                    advanced = True; break
                else:
                    if disc[vx] < low[ux]:
                        low[ux] = disc[vx]; stack[-1] = (ux, peid, it)
            if not advanced:
                stack.pop()
                if stack:
                    pu, _, _ = stack[-1]
                    if low[ux] < low[pu]: low[pu] = low[ux]
                    # The edge we took down into ux (id=peid) is a bridge iff
                    # low[ux] > disc[pu].
                    if low[ux] > disc[pu] and peid >= 0:
                        is_bridge[peid] = True
    return [(int(u[i]), int(v[i])) for i in range(m) if not is_bridge[i]]


def dense_laplacian(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    r = edge_index[0].numpy(); c = edge_index[1].numpy()
    A[r, c] = 1.0; A[c, r] = 1.0
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def eig_and_gap(edge_index: torch.Tensor, num_nodes: int, k: int = 8) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (sorted evals, evecs, δ_min over λ_1..λ_{min(k,n)-1})."""
    L = dense_laplacian(edge_index, num_nodes)
    w, V = np.linalg.eigh(L)
    # numpy.eigh already sorts ascending for symmetric; enforce anyway.
    order = np.argsort(w); w = w[order]; V = V[:, order]
    w = np.clip(w, 0.0, None)
    take = min(k, num_nodes)
    if take < 3:
        dmin = np.nan
    else:
        dmin = float(np.min(np.diff(w[1:take])))
    return w, V, dmin


def deltamin_bin(dmin: float) -> str:
    if not np.isfinite(dmin):         return "N"
    if dmin < 1e-10:                  return "A"
    if dmin < 0.05:                   return "B"
    if dmin < 0.15:                   return "C"
    return "D"


# ---------------------------------------------------------------------------
# PE (re)computation
# ---------------------------------------------------------------------------

def recompute_pe(data: Data, cfg, pe_type: str) -> Data:
    """Run the codebase's PE preprocessing on ``data`` in place (cloned ok)."""
    from graphgps.transform.posenc_stats import compute_posenc_stats
    # Drop stale PE tensors so compute_posenc_stats overwrites cleanly.
    for attr in ("EigVals", "EigVecs", "eigvals_sn", "eigvecs_sn",
                 "pestat_LHKS_eigvals", "pestat_LHKS_eigvecs_sq"):
        if hasattr(data, attr):
            delattr(data, attr)
    return compute_posenc_stats(data, pe_types=[pe_type],
                                is_undirected=True, cfg=cfg)


def _get_lap_decomp_stats(evals_np, evects_np, max_freqs, eigvec_norm):
    """Thin wrapper around codebase helper, accepting precomputed (evals, evecs)."""
    from graphgps.transform.posenc_stats import get_lap_decomp_stats
    return get_lap_decomp_stats(evals=evals_np, evects=evects_np,
                                max_freqs=max_freqs, eigvec_norm=eigvec_norm)


def apply_pe_from_decomp(data: Data, evals: np.ndarray, evecs: np.ndarray,
                         cfg, pe_type: str) -> Data:
    """Fill PE attributes on ``data`` using the supplied (evals, evecs) in place
    of a fresh eigendecomposition.  Used by Exp 2 (rotation injection)."""
    import torch.nn.functional as F
    N = data.num_nodes
    if pe_type == "LapPE":
        EigVals, EigVecs = _get_lap_decomp_stats(
            evals, evecs,
            cfg.posenc_LapPE.eigen.max_freqs,
            cfg.posenc_LapPE.eigen.eigvec_norm)
        data.EigVals, data.EigVecs = EigVals, EigVecs
    elif pe_type == "SignNet":
        EigVals, EigVecs = _get_lap_decomp_stats(
            evals, evecs,
            cfg.posenc_SignNet.eigen.max_freqs,
            cfg.posenc_SignNet.eigen.eigvec_norm)
        data.eigvals_sn, data.eigvecs_sn = EigVals, EigVecs
    elif pe_type == "LHKS":
        ev   = torch.from_numpy(evals).double()
        evec = torch.from_numpy(evecs).double()
        evec = F.normalize(evec.float(), p=2., dim=0)
        ev   = ev.float()
        mask = ev >= 1e-8
        ev   = ev[mask]; evec = evec[:, mask]
        num_eigvec = cfg.posenc_LHKS.num_eigvec
        k = min(num_eigvec, ev.numel())
        ev   = ev[:k]; evec = evec[:, :k]
        evec_sq = evec ** 2
        if k < num_eigvec:
            ev      = F.pad(ev,      (0, num_eigvec - k), value=0.0)
            evec_sq = F.pad(evec_sq, (0, num_eigvec - k), value=0.0)
        data.pestat_LHKS_eigvals    = ev.unsqueeze(0).expand(N, -1).clone().float()
        data.pestat_LHKS_eigvecs_sq = evec_sq.float()
    else:
        raise ValueError(f"Unsupported pe_type {pe_type}")
    return data


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def batched_forward(model, data_list: Sequence[Data], device: torch.device,
                    batch_size: int = 64) -> np.ndarray:
    """Run ``model`` on ``data_list`` and return a 1-D array of scalar preds."""
    preds: List[np.ndarray] = []
    for i in range(0, len(data_list), batch_size):
        batch = Batch.from_data_list(list(data_list[i:i + batch_size]))
        batch.split = "test"
        batch = batch.to(device)
        out = model(batch)
        pred = out[0] if isinstance(out, tuple) else out
        preds.append(pred.detach().float().reshape(-1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Perturbation: remove edges
# ---------------------------------------------------------------------------

def remove_edges(data: Data, to_remove: Sequence[Tuple[int, int]]) -> Data:
    """Return a new Data with the given undirected edges removed (both
    directions + corresponding edge_attr rows)."""
    if not to_remove:
        return data.clone()
    rm = {(min(u, v), max(u, v)) for u, v in to_remove}
    ei = data.edge_index
    r = ei[0].tolist(); c = ei[1].tolist()
    keep = [(min(a, b), max(a, b)) not in rm for a, b in zip(r, c)]
    keep_t = torch.tensor(keep, dtype=torch.bool)
    new = data.clone()
    new.edge_index = ei[:, keep_t]
    if data.edge_attr is not None:
        new.edge_attr = data.edge_attr[keep_t]
    return new


# ---------------------------------------------------------------------------
# Stratified subsample across δ_min bins
# ---------------------------------------------------------------------------

def stratified_subsample(test_graphs: PCQMGraphs, n_target: int, bin_k: int = 8,
                         seed: int = 0, skip_trees: bool = False,
                         skip_tiny: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                          Dict[str, int]]:
    """Pass once over the test set, compute δ_min @ k=bin_k and stratify across
    four bins (A/B/C/D).  Returns (sampled_graph_ids, δ_min per sample, counts).

    If ``skip_trees`` we additionally require >=1 non-bridge edge.
    """
    rng = default_rng(seed)
    dmins: List[float] = []
    kept: List[int] = []
    print(f"[common] Computing δ_min (k={bin_k}) for {len(test_graphs.test_idx):,} "
          f"test graphs (CPU eigh) ...")
    for gi in test_graphs.test_idx.tolist():
        d = test_graphs.get(gi)
        n = d.num_nodes
        if skip_tiny and n < 3:
            continue
        if skip_trees:
            if not find_non_bridge_edges(d.edge_index, n):
                continue
        _, _, dmin = eig_and_gap(d.edge_index, n, k=bin_k)
        dmins.append(dmin); kept.append(gi)
    dmins = np.asarray(dmins, dtype=np.float64)
    kept  = np.asarray(kept,  dtype=np.int64)
    bins = np.array([deltamin_bin(d) for d in dmins])
    per_bin = max(1, n_target // 4)
    sampled_ids, sampled_dmin = [], []
    counts = {}
    for b in ["A", "B", "C", "D"]:
        mask = bins == b
        idxs = np.where(mask)[0]
        take = min(per_bin, len(idxs))
        if take == 0:
            counts[b] = 0
            continue
        pick = rng.choice(idxs, size=take, replace=False)
        sampled_ids.append(kept[pick]); sampled_dmin.append(dmins[pick])
        counts[b] = int(take)
    sampled_ids  = np.concatenate(sampled_ids)
    sampled_dmin = np.concatenate(sampled_dmin)
    print(f"[common] Stratified sample per bin: {counts}  "
          f"(total {len(sampled_ids)})")
    return sampled_ids, sampled_dmin, counts
