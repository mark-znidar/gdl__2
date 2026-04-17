"""Stability perturbation experiment.

For trained L-HKS and RWSE models, measures prediction variance under random
edge perturbations at various epsilon levels.  Also records the minimum
eigenvalue gap of each original test graph.

Usage:
    python experiments/run_stability.py --gpu 0
"""
import argparse
import os
import sys
from collections import deque
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, makedirs_rm_exist
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.utils import (to_undirected, get_laplacian,
                                    to_scipy_sparse_matrix, to_dense_adj)

# Make project modules importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphgps  # noqa — registers custom modules

from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig


EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.20]
N_PERTURBATIONS = 50
N_TEST_GRAPHS = 1000


def new_optimizer_config(cfg):
    from torch_geometric.graphgym.optim import OptimizerConfig
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def load_trained_model(cfg_path, run_dir, device):
    """Load a trained model from a checkpoint directory."""
    from torch_geometric.graphgym.cmd_args import parse_args

    set_cfg(cfg)

    # Minimal args to load config
    class FakeArgs:
        cfg_file = cfg_path
        opts = []

    load_cfg(cfg, FakeArgs())
    cfg.run_dir = run_dir
    cfg.accelerator = device
    cfg.wandb.use = False

    model = create_model()
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    ckpt_dir = os.path.join(run_dir, 'ckpt')
    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        load_ckpt(model, optimizer, scheduler, epoch=-1)
    else:
        print(f"  WARNING: no checkpoint in {ckpt_dir}, using random weights")

    model.to(device)
    model.eval()
    return model


def compute_eigval_gap(data):
    """Compute minimum gap among non-zero eigenvalues of the graph Laplacian."""
    N = data.num_nodes
    edge_index = to_undirected(data.edge_index) if not data.is_undirected() else data.edge_index
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=N)
    )
    evals = np.linalg.eigvalsh(L.toarray())
    evals = np.sort(evals)
    nonzero = evals[evals > 1e-8]
    if len(nonzero) < 2:
        return 0.0
    gaps = np.diff(nonzero)
    return float(gaps.min())


def perturb_graph(data, epsilon, rng):
    """Create a perturbed copy by randomly adding/removing edges."""
    data = deepcopy(data)
    N = data.num_nodes
    edge_index = data.edge_index

    # Build undirected edge set
    edges = set()
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u != v:
            edges.add((min(u, v), max(u, v)))

    n_orig = len(edges)
    new_edges = set()

    # Remove edges with probability epsilon
    for e in edges:
        if rng.random() >= epsilon:
            new_edges.add(e)

    # Add non-edges with probability to keep expected edge count constant
    n_possible_new = N * (N - 1) // 2 - n_orig
    if n_possible_new > 0:
        p_add = epsilon * n_orig / n_possible_new
        n_to_sample = min(int(p_add * n_possible_new * 2) + 10, n_possible_new)
        sampled = 0
        attempts = 0
        while sampled < n_to_sample and attempts < n_to_sample * 3:
            u = rng.randint(0, N - 1)
            v = rng.randint(0, N - 1)
            if u != v:
                e = (min(u, v), max(u, v))
                if e not in edges and e not in new_edges:
                    if rng.random() < p_add / max(p_add, 0.5):
                        new_edges.add(e)
                    sampled += 1
            attempts += 1

    if len(new_edges) == 0:
        new_edges = edges

    src, dst = [], []
    for u, v in new_edges:
        src.extend([u, v])
        dst.extend([v, u])

    data.edge_index = torch.tensor([src, dst], dtype=torch.long)

    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        n_new_edges = data.edge_index.size(1)
        data.edge_attr = torch.zeros(n_new_edges, dtype=data.edge_attr.dtype)

    return data


def get_pe_types_from_cfg():
    """Determine which PE types are enabled in the current config."""
    pe_types = []
    # cfg is a YACS CfgNode (dict subclass); keys are not listed by dir(cfg).
    for key in list(cfg.keys()):
        if not key.startswith('posenc_'):
            continue
        pecfg = getattr(cfg, key, None)
        if pecfg is None or not hasattr(pecfg, 'enable'):
            continue
        if pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_types.append(pe_name)
    return pe_types


@torch.no_grad()
def predict_single(model, data, device):
    """Run model on a single graph and return scalar prediction."""
    data = data.clone()
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.to(device)
    pred, _ = model(data)
    return pred.cpu().item()


def run_experiment(method_name, cfg_path, run_dir, test_dataset, device):
    """Run stability experiment for one method."""
    print(f"\n=== {method_name} ===")
    print(f"  Config: {cfg_path}")
    print(f"  Run dir: {run_dir}")

    model = load_trained_model(cfg_path, run_dir, device)
    pe_types = get_pe_types_from_cfg()
    is_undirected = True

    results = {eps: [] for eps in EPSILONS}
    n_graphs = min(N_TEST_GRAPHS, len(test_dataset))

    for g_idx in range(n_graphs):
        if (g_idx + 1) % 100 == 0:
            print(f"  Graph {g_idx + 1}/{n_graphs}")

        data = test_dataset[g_idx]
        eigval_gap = compute_eigval_gap(data)

        for eps in EPSILONS:
            preds = []
            for p in range(N_PERTURBATIONS):
                rng = np.random.RandomState(seed=g_idx * 1000 + p)
                pert_data = perturb_graph(data, eps, rng)
                if pe_types:
                    pert_data = compute_posenc_stats(
                        pert_data, pe_types, is_undirected, cfg)
                pred = predict_single(model, pert_data, device)
                preds.append(pred)

            variance = float(np.var(preds))
            results[eps].append({
                'graph_idx': g_idx,
                'variance': variance,
                'eigval_gap': eigval_gap,
            })

    return results


def find_run_dir_with_ckpt(seed_root, max_depth=4):
    """Resolve e.g. .../exp1_lhks/seed0 to the leaf dir that contains ckpt/."""
    if not os.path.isdir(seed_root):
        return None

    dq = deque([(seed_root, 0)])
    while dq:
        path, depth = dq.popleft()
        ckpt = os.path.join(path, "ckpt")
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            return path
        if depth >= max_depth:
            continue
        try:
            subs = sorted(os.listdir(path))
        except OSError:
            continue
        for name in subs:
            sub = os.path.join(path, name)
            if os.path.isdir(sub):
                dq.append((sub, depth + 1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load test dataset
    from torch_geometric.datasets import ZINC
    test_dataset = ZINC(root='datasets/ZINC', subset=True, split='test')
    print(f"Test dataset: {len(test_dataset)} graphs")

    outbase = os.environ.get("EXPERIMENT_OUTBASE", "").strip()
    if not outbase:
        outbase = (
            "results/slurm1_2seed"
            if os.path.isdir("results/slurm1_2seed")
            else "results"
        )
    print(f"EXPERIMENT_OUTBASE (resolved): {outbase}")

    specs = [
        ("LHKS", "configs/GPS/zinc-GPS+LHKS.yaml",
         os.path.join(outbase, "exp1_lhks", "seed0")),
        ("RWSE", "configs/GPS/zinc-GPS+RWSE.yaml",
         os.path.join(outbase, "exp0_rwse", "seed0")),
    ]

    all_results = {}
    for method_name, cfg_path, seed_root in specs:
        run_dir = find_run_dir_with_ckpt(seed_root)
        if run_dir is None:
            print(f"  SKIP {method_name}: no checkpoint under {seed_root}")
            continue
        print(f"  Using run_dir: {run_dir}")
        all_results[method_name] = run_experiment(
            method_name, cfg_path, run_dir, test_dataset, device)

    os.makedirs("results", exist_ok=True)
    save_path = "results/stability_results.pt"
    torch.save(all_results, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
