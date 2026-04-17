#!/bin/bash
# PCQM4Mv2-Subset (10%) L-HKS / HKdiagSE benchmark submission script.
#
# Submits one Slurm job per (config, seed). 4 seeds (0,1,2,3) per config.
# 6 configs × 4 seeds = 24 training jobs. On 10 A100s in parallel, wall-clock ~6-7 h.
#
# Uses GPS-small backbone (5 layers, dim_hidden=304, GatedGCN+Transformer)
# matching the paper's Appendix A protocol for PCQM4Mv2-Subset ablation (Table B.2).
#
# Configs expected at configs/GPS/slurm2seed/pcqm4m-subset-GPS+*.yaml

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

OUTBASE="${REPO_DIR}/results_pcqm4m_subset/slurm1_4seed"
LOGDIR="${REPO_DIR}/slurm_logs_pcqm4m_subset"
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

echo "=== Submitting PCQM4Mv2-Subset L-HKS experiments (4 seeds: 0,1,2,3) ==="
ALL=""

# Exp 1: L-HKS K=32 (main method, learnable)
for S in 0 1 2 3; do
    J=$(submit "pcqm_lhks_K32_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+LHKS-K32.yaml" \
        $S "${OUTBASE}/exp1_lhks_K32/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K32_s${S} -> ${J}"
done

# Exp 2: L-HKS K=16 (learnable, K ablation)
for S in 0 1 2 3; do
    J=$(submit "pcqm_lhks_K16_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+LHKS-K16.yaml" \
        $S "${OUTBASE}/exp2_lhks_K16/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K16_s${S} -> ${J}"
done

# Exp 3: L-HKS K=4 (learnable, K ablation)
for S in 0 1 2 3; do
    J=$(submit "pcqm_lhks_K4_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+LHKS-K4.yaml" \
        $S "${OUTBASE}/exp3_lhks_K4/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K4_s${S} -> ${J}"
done

# Exp 3b: L-HKS K=5 (learnable, K ablation)
for S in 0 1 2 3; do
    J=$(submit "pcqm_lhks_K5_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+LHKS-K5.yaml" \
        $S "${OUTBASE}/exp3b_lhks_K5/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_lhks_K5_s${S} -> ${J}"
done

# Exp 4: L-HKS K=32 FIXED (not learnable, learnable-vs-fixed ablation)
for S in 0 1 2 3; do
    J=$(submit "pcqm_fixed_K32_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+LHKS-K32-fixed.yaml" \
        $S "${OUTBASE}/exp4_lhks_K32_fixed/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_fixed_K32_s${S} -> ${J}"
done

# Exp 5: HKdiagSE (fixed integer times 1..20) — spectral baseline
for S in 0 1 2 3; do
    J=$(submit "pcqm_hkdiag_s${S}" \
        "configs/GPS/slurm2seed/pcqm4m-subset-GPS+HKdiagSE.yaml" \
        $S "${OUTBASE}/exp5_hkdiag/seed${S}")
    ALL="${ALL}:${J}"; echo "  pcqm_hkdiag_s${S} -> ${J}"
done

DEPS="${ALL#:}"
echo ""
echo "Training jobs (24): $(echo $DEPS | tr ':' ' ')"

# Post-processing (optional; runs after all training)
POST=$(sbatch --parsable \
    --job-name="pcqm_post_4seed" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --constraint=ampere \
    --time="${TIME_POST}" \
    --dependency="afterok:${DEPS}" \
    --output="${LOGDIR}/post_4seed_%j.out" \
    --error="${LOGDIR}/post_4seed_%j.err" \
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
echo \"=== PCQM4Mv2-Subset post-processing: \$(date) ===\"
python experiments/collect_results.py || true
echo \"=== Done: \$(date) ===\"
")

echo "Post-processing: ${POST} (afterok:${DEPS})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/pcqm_lhks_K32_s0_*.out"
echo "Results: ${OUTBASE}"
