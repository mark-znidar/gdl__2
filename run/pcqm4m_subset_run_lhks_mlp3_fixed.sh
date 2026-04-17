#!/bin/bash
# PCQM4Mv2-Subset L-HKS K=16 MLP layers=3, FIXED diffusion times (not learned).
#
# Companion to the MLP-depth ablation; identical architecture to
# configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3.yaml but with
# learn_times: False / freeze_times: True. Isolates the contribution of
# learnable times on top of the canonical 3-layer MLP encoder.
#
# 1 config x 3 seeds = 3 training jobs.
# WandB project: pcqm4m-subset-lhks-mlp-ablation (shared with MLP-depth ablation)
# Output base:   results_pcqm4m_subset/mlp_ablation/mlp3_fixed

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
MEM="48G"
TIME_TRAIN="08:00:00"
# ============================================================
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# ============================================================

OUTBASE="${REPO_DIR}/results_pcqm4m_subset/mlp_ablation"
LOGDIR="${REPO_DIR}/slurm_logs_pcqm4m_subset_mlp_ablation"
mkdir -p "$LOGDIR"

ACCT_FLAG=""
[ -n "$ACCOUNT" ] && ACCT_FLAG="--account=${ACCOUNT}"

submit() {
    local NAME=$1 CFG=$2 SEED=$3 OUTDIR=$4
    sbatch --parsable \
        --job-name="${NAME}" \
        --partition="${PARTITION}" ${ACCT_FLAG} \
        --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
        --constraint=ampere \
        --time="${TIME_TRAIN}" \
        --output="${LOGDIR}/${NAME}_%j.out" \
        --error="${LOGDIR}/${NAME}_%j.err" \
        --wrap="
cd ${REPO_DIR}
export XDG_CACHE_HOME=/dfs/user/mznidar/.cache
export TORCH_HOME=\${XDG_CACHE_HOME}/torch
export HF_HOME=\${XDG_CACHE_HOME}/huggingface
export PIP_CACHE_DIR=\${XDG_CACHE_HOME}/pip
export PIXI_CACHE_DIR=/dfs/scratch0/mznidar/pixi-cache
export TMPDIR=/dfs/scratch0/mznidar/tmp
mkdir -p \$TMPDIR \$XDG_CACHE_HOME
${ACTIVATE_CMD}
echo \"Job: ${NAME} | Node: \$(hostname) | Start: \$(date)\"
python main.py --cfg ${CFG} --repeat 1 seed ${SEED} out_dir ${OUTDIR}
echo \"Done: \$(date)\"
"
}

echo "=== Submitting PCQM4Mv2-Subset L-HKS K=16 MLP3 FIXED-times (3 seeds: 0,1,2) ==="
ALL=""

for S in 0 1 2; do
    J=$(submit "pcqm_lhks_K16_mlp3_fixed_s${S}" \
        "configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3-fixed.yaml" \
        $S "${OUTBASE}/mlp3_fixed/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K16_mlp3_fixed_s${S} -> ${J}"
done

DEPS="${ALL#:}"
echo ""
echo "Training jobs (3): $(echo $DEPS | tr ':' ' ')"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/pcqm_lhks_K16_mlp3_fixed_s0_*.out"
echo "Results: ${OUTBASE}/mlp3_fixed"
echo "WandB:   project 'pcqm4m-subset-lhks-mlp-ablation'"
