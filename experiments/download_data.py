"""Pre-download ZINC dataset so training scripts don't block on I/O."""
from torch_geometric.datasets import ZINC

for split in ['train', 'val', 'test']:
    ZINC(root='datasets/ZINC', subset=True, split=split)
    print(f"ZINC {split}: OK")

print("All datasets downloaded.")
