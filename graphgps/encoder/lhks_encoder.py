import math

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('LHKS')
class LHKSNodeEncoder(torch.nn.Module):
    """Learnable Heat Kernel Signature positional encoding encoder.

    Unlike HKdiagSE which uses fixed diffusion times, this encoder has
    LEARNABLE diffusion times that are optimized during training.

    The HKS at node v and time t is:
        HKS_t(v) = sum_i exp(-lambda_i * t) * phi_i(v)^2

    We compute this at K learnable time scales and map through an MLP.

    Preprocessing stores eigenpairs (eigenvalues repeated per node, squared
    eigenvectors) on the Data object.  The actual HKS computation happens
    here so that gradients flow back to the learned times.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features ``x`` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in

        pecfg = cfg.posenc_LHKS
        dim_pe = pecfg.dim_pe
        K = pecfg.kernel_times
        model_type = pecfg.model.lower()
        n_layers = pecfg.layers
        norm_type = pecfg.raw_norm_type.lower()
        self.pass_as_var = pecfg.pass_as_var

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Learnable diffusion times parameterised in log-space so t_j = exp(r_j) > 0.
        init_log_times = torch.linspace(math.log(0.01), math.log(100.0), K)
        if pecfg.freeze_times or not pecfg.learn_times:
            self.register_buffer('log_times', init_log_times)
        else:
            self.log_times = nn.Parameter(init_log_times)

        # Optional normalisation of the raw K-dim HKS vector.
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(K)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(K, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(K, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(K, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        if not (hasattr(batch, 'pestat_LHKS_eigvals')
                and hasattr(batch, 'pestat_LHKS_eigvecs_sq')):
            raise ValueError(
                "Precomputed LHKS eigenpairs are required for "
                f"{self.__class__.__name__}; set config "
                "'posenc_LHKS.enable' to True")

        eigvals = batch.pestat_LHKS_eigvals       # [total_nodes, k]
        eigvecs_sq = batch.pestat_LHKS_eigvecs_sq  # [total_nodes, k]

        times = torch.exp(self.log_times)  # [K], always positive

        # HKS_{t_j}(v) = sum_i exp(-lambda_i * t_j) * phi_i(v)^2
        # eigvals:   [total_nodes, k]    -> [total_nodes, k, 1]
        # times:     [K]                 -> [1, 1, K]
        # exp_terms: [total_nodes, k, K]
        exp_terms = torch.exp(-eigvals.unsqueeze(-1) * times)

        # eigvecs_sq: [total_nodes, k] -> [total_nodes, k, 1]
        # hks:        [total_nodes, K]
        hks = (exp_terms * eigvecs_sq.unsqueeze(-1)).sum(dim=1)

        if self.raw_norm:
            hks = self.raw_norm(hks)
        pos_enc = self.pe_encoder(hks)  # [total_nodes, dim_pe]

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        batch.x = torch.cat((h, pos_enc), 1)

        if self.pass_as_var:
            batch.pe_LHKS = pos_enc
        return batch
