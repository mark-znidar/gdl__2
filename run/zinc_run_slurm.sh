#!/bin/bash
# Submit all experiments to Slurm. Edit config block below first.

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
MEM="42G"
TIME_TRAIN="14:00:00"
TIME_POST="02:00:00"
# ============================================================
# For pixi (default for this repo):
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# For conda instead, uncomment:
# ACTIVATE_CMD="source activate graphgps 2>/dev/null || conda activate graphgps"
# ============================================================

OUTBASE="${REPO_DIR}/results/slurm1"
LOGDIR="${REPO_DIR}/slurm_logs"
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

echo "=== Submitting L-HKS experiments ==="
ALL=""

# Exp 0: RWSE baseline
for S in 0 1 2 3; do
    J=$(submit "rwse_s${S}" "configs/GPS/zinc-GPS+RWSE.yaml" $S "${OUTBASE}/exp0_rwse/seed${S}")
    ALL="${ALL}:${J}"; echo "  rwse_s${S} -> ${J}"
done

# Exp 1: L-HKS main
for S in 0 1 2 3; do
    J=$(submit "lhks_s${S}" "configs/GPS/zinc-GPS+LHKS.yaml" $S "${OUTBASE}/exp1_lhks/seed${S}")
    ALL="${ALL}:${J}"; echo "  lhks_s${S} -> ${J}"
done

# Exp 4a: Fixed times
for S in 0 1 2 3; do
    J=$(submit "fixed_s${S}" "configs/GPS/zinc-GPS+LHKS-fixed.yaml" $S "${OUTBASE}/exp4a_fixed/seed${S}")
    ALL="${ALL}:${J}"; echo "  fixed_s${S} -> ${J}"
done

# Exp 4b: K ablations
for K in 4 8 32; do
    J=$(submit "K${K}" "configs/GPS/zinc-GPS+LHKS-K${K}.yaml" 0 "${OUTBASE}/exp4b_K${K}/seed0")
    ALL="${ALL}:${J}"; echo "  K${K} -> ${J}"
done

# Exp 0b: No PE
J=$(submit "noPE" "configs/GPS/zinc-GPS-noPE.yaml" 0 "${OUTBASE}/exp0b_noPE/seed0")
ALL="${ALL}:${J}"; echo "  noPE -> ${J}"

DEPS="${ALL#:}"
echo ""
echo "Training jobs: $(echo $DEPS | tr ':' ' ')"

# Post-processing (runs after all training)
POST=$(sbatch --parsable \
    --job-name="lhks_post" \
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
echo \"=== Post-processing: \$(date) ===\"
python experiments/run_stability.py --gpu 0
python experiments/run_sr25.py
python experiments/collect_results.py
python experiments/plot_results.py
echo \"=== Done: \$(date) ===\"
")

echo "Post-processing: ${POST} (afterok:${DEPS})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/lhks_s0_*.out"
