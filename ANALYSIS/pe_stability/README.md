# PE-Level Stability Experiment

Measures how LapPE and HKS positional encodings change under small random
edge perturbations, and relates that change to each graph's minimum non-zero
eigenvalue gap.

**Expected finding:** LapPE change correlates (negatively) with small
eigenvalue gaps; HKS change is largely insensitive to the gap.

## Run

From the repo root:

```bash
pixi run python ANALYSIS/pe_stability/run_pe_stability.py
```

or from this folder:

```bash
pixi run python run_pe_stability.py
```

On a single CPU this takes a few minutes to ~15 minutes depending on the
machine (~450 graphs × 3 epsilons × 30 perturbations, each requiring a dense
eigendecomposition of a small Laplacian).

## Outputs (all written next to the script)

| File | Description |
| --- | --- |
| `pe_stability_scatter.png` | Figure A — PE change vs min eigenvalue gap at ε = 0.05 (paper Figure 2 candidate). |
| `pe_stability_lines.png`   | Figure B — Mean PE change vs ε (appendix candidate). |
| `pe_stability_results.json` | Per-graph / per-ε raw means and stds. |
| `pe_stability_summary.json` | Pearson correlations (log-log) of gap vs LapPE/HKS change at ε = 0.05, plus the small-gap ratio. |

## Method in one paragraph

For each of ~450 connected graphs (cycles, grids, Erdős–Rényi, a few named
regular graphs) we compute the full Laplacian spectrum, the lowest-8 LapPE
eigenvectors, and a 16-time HKS built from the same 8 eigenpairs. We perturb
each graph 30 times at ε ∈ {0.01, 0.05, 0.10} — each existing edge is
removed with probability ε and non-edges are added with a matching
probability so the expected edge count is preserved. For each connected
perturbation we compare LapPE with a per-column sign-aligned (best-case)
Frobenius distance and HKS with a plain normalized Frobenius distance. All
distances are normalized by √n so graphs of different sizes are comparable.
