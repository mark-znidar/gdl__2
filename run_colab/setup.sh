#!/usr/bin/env bash
# One-time dependency setup for running gdl__2 on Google Colab.
# Detects Colab's PyTorch version and installs matching PyG wheels + project deps.
set -euo pipefail

echo "=== Detecting torch / CUDA ==="
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+v.replace('.','')) if v else print('cpu')")
echo "torch=${TORCH_VERSION}  cuda_tag=${CUDA_TAG}"

echo "=== Installing PyG + extensions ==="
pip install -q torch-geometric
pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html" || {
        echo "WARN: extension wheels not found for torch-${TORCH_VERSION}+${CUDA_TAG};"
        echo "      falling back to core torch-geometric only (some ops slower but still correct)."
    }

echo "=== Installing project deps (from pixi.toml [pypi-dependencies]) ==="
pip install -q yacs tensorboardX ogb performer-pytorch pytorch-lightning \
    torchmetrics rdkit networkx wandb fsspec

echo "=== Installing gdl__2 as editable package ==="
pip install -q -e .

echo "=== Sanity check ==="
python - <<'PY'
import torch, torch_geometric
print(f"torch:  {torch.__version__}")
print(f"pyg:    {torch_geometric.__version__}")
print(f"cuda:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu:    {torch.cuda.get_device_name(0)}")
try:
    import torch_scatter, torch_sparse
    print("ext:    torch-scatter + torch-sparse OK")
except ImportError as e:
    print(f"ext:    missing ({e}); continuing with core PyG only")
PY

echo "=== Setup complete ==="
