# PCQM4Mv2-subset — best/test_mae by group (mean ± std)

**Entity:** `znidar-mark-stanford-university`  
**Projects:** `pcqm4m-subset-lhks-mlp-ablation`, `pcqm4m-subset-stability-baselines`  
**Metric:** `best/test_mae` (lower is better). Std is sample std (ddof=1).

| Group | n | Per-seed MAE | **Mean ± Std** |
|---|---:|---|---:|
| SignNetMLP | 2 | 0.12295, 0.12454 | **0.12374 ± 0.00112** |
| LHKS-fixed | 3 | 0.12683, 0.12604, 0.12513 | **0.12600 ± 0.00085** |
| LHKS (MLP3) | 3 | 0.12628, 0.12595, 0.12633 | **0.12619 ± 0.00021** |
| LapPE | 2 | 0.12673, 0.12683 | **0.12678 ± 0.00007** |

## Run names per group

**SignNetMLP** (n=2):
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer.r0+SignNetMLP` → 0.12295
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer.r1+SignNetMLP` → 0.12454

**LHKS-fixed** (n=3):
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS-fixed.r0` → 0.12683
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS-fixed.r1` → 0.12604
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS-fixed.r2` → 0.12513

**LHKS (MLP3)** (n=3):
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS.r0MLP3-s0` → 0.12628
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS.r1MLP3-s1` → 0.12595
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LHKS.r2MLP3-s2` → 0.12633

**LapPE** (n=2):
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LapPE.r0` → 0.12673
- `PCQM4Mv2-subset.GPS.CustomGatedGCN+Transformer+LapPE.r1` → 0.12683
