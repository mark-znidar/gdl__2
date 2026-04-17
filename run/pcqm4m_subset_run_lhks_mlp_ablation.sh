#!/bin/bash
# PCQM4Mv2-Subset L-HKS K=16 MLP-depth ablation.
#
# Keeps K=16, num_eigvec=8, dim_pe=20, raw_norm=BatchNorm identical to the
# current Linear baseline; only varies the PE encoder depth:
#   MLP layers=2 (~1.5k PE params)
#   MLP layers=3 (~3.1k PE params)  <-- canonical universal-approximator MLP
#   MLP layers=4 (~4.8k PE params)
#
# Stability theorem is preserved: HKS is stable -> Lipschitz ReLU-MLP composition
# is still stable (only the Lipschitz constant changes).
#
# 3 configs x 3 seeds = 9 training jobs.
# WandB project: pcqm4m-subset-lhks-mlp-ablation
# Output base:   results_pcqm4m_subset/mlp_ablation

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
# PCQM4Mv2 (even subset) needs up to 48GB system RAM per paper Appendix A.4.
MEM="48G"
# ~2-3 h per run; budget 8 h with headroom.
TIME_TRAIN="08:00:00"
TIME_POST="02:00:00"
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

echo "=== Submitting PCQM4Mv2-Subset L-HKS K=16 MLP-depth ablation (3 seeds: 0,1,2) ==="
ALL=""

# Exp A: L-HKS K=16 MLP layers=2
for S in 0 1 2; do
    J=$(submit "pcqm_lhks_K16_mlp2_s${S}" \
        "configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP2.yaml" \
        $S "${OUTBASE}/mlp2/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K16_mlp2_s${S} -> ${J}"
done

# Exp B: L-HKS K=16 MLP layers=3 (canonical)
for S in 0 1 2; do
    J=$(submit "pcqm_lhks_K16_mlp3_s${S}" \
        "configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP3.yaml" \
        $S "${OUTBASE}/mlp3/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K16_mlp3_s${S} -> ${J}"
done

# Exp C: L-HKS K=16 MLP layers=4
for S in 0 1 2; do
    J=$(submit "pcqm_lhks_K16_mlp4_s${S}" \
        "configs/GPS/pcqm4m-subset-GPS+LHKS-K16-MLP4.yaml" \
        $S "${OUTBASE}/mlp4/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K16_mlp4_s${S} -> ${J}"
done

DEPS="${ALL#:}"
echo ""
echo "Training jobs (9): $(echo $DEPS | tr ':' ' ')"

# Post-processing (optional; runs after all training)
POST=$(sbatch --parsable \
    --job-name="pcqm_mlp_ablation_post" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --constraint=ampere \
    --time="${TIME_POST}" \
    --dependency="afterok:${DEPS}" \
    --output="${LOGDIR}/post_%j.out" \
    --error="${LOGDIR}/post_%j.err" \
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
export EXPERIMENT_OUTBASE=\"${OUTBASE}\"
echo \"=== PCQM4Mv2-Subset MLP-ablation post-processing: \$(date) ===\"
python experiments/collect_results.py || true
echo \"=== Done: \$(date) ===\"
")

echo "Post-processing: ${POST} (afterok:${DEPS})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/pcqm_lhks_K16_mlp3_s0_*.out"
echo "Results: ${OUTBASE}"
echo "WandB:   project 'pcqm4m-subset-lhks-mlp-ablation'"
