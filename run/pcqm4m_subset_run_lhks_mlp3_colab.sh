#!/usr/bin/env bash
# PCQM4Mv2-Subset L-HKS K=16 MLP layers=3 (learned diffusion times).
# Colab-friendly runner: no sbatch, no cluster paths, seeds run sequentially
# on whatever single GPU is attached to the current runtime.
#
# Usage (from repo root, inside a Colab cell):
#   !bash run/pcqm4m_subset_run_lhks_mlp3_colab.sh

# ============================================================
# ENTER ANY SEEDS YOU WANT TO RUN HERE
# ============================================================
SEEDS=(3 4 5)
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CFG="configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3.yaml"
OUTBASE="${REPO_DIR}/results_pcqm4m_subset/mlp_ablation/mlp3"

echo "=== PCQM4Mv2-Subset L-HKS K=16 MLP3 — Colab (seeds: ${SEEDS[*]}) ==="
echo "Repo:    ${REPO_DIR}"
echo "Config:  ${CFG}"
echo "Outbase: ${OUTBASE}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(no nvidia-smi found)"
echo ""

SKIPPED=""
RAN=""

for S in "${SEEDS[@]}"; do
    OUTDIR="${OUTBASE}/seed${S}"
    if compgen -G "${OUTDIR}/*/ckpt/*.ckpt" >/dev/null; then
        echo "[skip] seed${S}: existing checkpoint found at ${OUTDIR}"
        SKIPPED="${SKIPPED} ${S}"
        continue
    fi
    mkdir -p "${OUTDIR}"
    echo "============================================================"
    echo ">>> mlp3 seed=${S}  start: $(date)"
    echo ">>> outdir: ${OUTDIR}"
    echo "============================================================"
    python main.py --cfg "${CFG}" --repeat 1 seed "${S}" out_dir "${OUTDIR}"
    echo ">>> mlp3 seed=${S}  done:  $(date)"
    RAN="${RAN} ${S}"
done

echo ""
echo "=== Summary ==="
[ -n "$RAN" ]     && echo "Completed seeds:${RAN}"
[ -n "$SKIPPED" ] && echo "Skipped seeds:  ${SKIPPED}"
echo "Results: ${OUTBASE}"
echo "WandB:   project 'pcqm4m-subset-lhks-mlp-ablation'"
