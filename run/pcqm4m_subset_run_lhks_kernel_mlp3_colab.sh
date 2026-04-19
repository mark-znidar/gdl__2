#!/usr/bin/env bash
# PCQM4Mv2-Subset L-HKS MLP3 kernel-times ablation: K ∈ {2,4,8,32}.
# Identical stack to pcqm4m-subset-GPS+LHKS-K16-MLP3.yaml except kernel_times.
# Colab-friendly: sequential jobs on one GPU (same pattern as
# run/pcqm4m_subset_run_lhks_mlp3_colab.sh).
#
# Usage (from repo root, e.g. inside Colab after %cd gdl__2):
#   !bash run/pcqm4m_subset_run_lhks_kernel_mlp3_colab.sh
#
# Prefer the notebook run_colab/06_run_lhks_mlp3_kernel_times_ablation.ipynb
# for Drive symlinks + wandb login.

# ============================================================
# Three seeds per K (edit to any list); four K values → 12 runs by default.
SEEDS=(3 4 5)
KERNEL_TIMES=(2 4 8 32)
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

OUTROOT="${REPO_DIR}/results_pcqm4m_subset/mlp_ablation/kernel_times_mlp3"

echo "=== L-HKS MLP3 kernel-times ablation (K: ${KERNEL_TIMES[*]}, seeds: ${SEEDS[*]}) ==="
echo "Repo:    ${REPO_DIR}"
echo "Outroot: ${OUTROOT}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(no nvidia-smi found)"
echo ""

for K in "${KERNEL_TIMES[@]}"; do
    CFG="configs/GPS/pcqm4m-subset-GPS+LHKS-K${K}-MLP3.yaml"
    if [[ ! -f "${CFG}" ]]; then
        echo "ERROR: missing ${CFG}" >&2
        exit 1
    fi
    for S in "${SEEDS[@]}"; do
        OUTDIR="${OUTROOT}/K${K}/seed${S}"
        if compgen -G "${OUTDIR}/*/ckpt/*.ckpt" >/dev/null; then
            echo "[skip] K=${K} seed=${S}: ckpt at ${OUTDIR}"
            continue
        fi
        mkdir -p "${OUTDIR}"
        echo ""
        echo ">>> K=${K} seed=${S}"
        python main.py --cfg "${CFG}" --repeat 1 seed "${S}" out_dir "${OUTDIR}"
    done
done

echo ""
echo "=== Done. Checkpoints under: ${OUTROOT}/K<K>/seed<seed>/.../ckpt/ ==="
