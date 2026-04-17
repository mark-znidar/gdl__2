#!/bin/bash
# Same experiment suite as run_slurm.sh, but submits pairs of training runs as a
# single Slurm job so that two processes share one GPU (2 per GPU).
# 16 training runs -> 8 paired Slurm jobs -> needs 8 GPUs to run concurrently.
#
# Pairings (chosen to match configs of similar duration):
#   rwse_s{0,1,2,3} + lhks_s{0,1,2,3}  -> 4 jobs  (same seed, different PE)
#   fixed_s0+fixed_s1, fixed_s2+fixed_s3 -> 2 jobs
#   K4+K8                                -> 1 job
#   K32+noPE                             -> 1 job
#
# Usage:  bash run/run_slurm_2per.sh

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
MEM="64G"
TIME_TRAIN="14:00:00"
TIME_POST="02:00:00"
# ============================================================
# For pixi (default for this repo):
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# For conda instead, uncomment:
# ACTIVATE_CMD="source activate graphgps 2>/dev/null || conda activate graphgps"
# ============================================================

OUTBASE="${REPO_DIR}/results/slurm_2per"
LOGDIR="${REPO_DIR}/slurm_logs"
mkdir -p "$LOGDIR"

ACCT_FLAG=""
[ -n "$ACCOUNT" ] && ACCT_FLAG="--account=${ACCOUNT}"

# submit_pair NAME CFG1 SEED1 OUT1 CFG2 SEED2 OUT2
# Launches two python main.py processes concurrently on the same GPU, then waits.
submit_pair() {
    local NAME=$1 CFG1=$2 SEED1=$3 OUT1=$4 CFG2=$5 SEED2=$6 OUT2=$7
    sbatch --parsable \
        --job-name="${NAME}" \
        --partition="${PARTITION}" ${ACCT_FLAG} \
        --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
        --constraint=ampere \
        --time="${TIME_TRAIN}" \
        --output="${LOGDIR}/${NAME}_%j.out" \
        --error="${LOGDIR}/${NAME}_%j.err" \
        --wrap="
set -e
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
python main.py --cfg ${CFG1} --repeat 1 seed ${SEED1} out_dir ${OUT1} &
python main.py --cfg ${CFG2} --repeat 1 seed ${SEED2} out_dir ${OUT2} &
wait
echo \"Done: \$(date)\"
"
}

echo "=== Submitting L-HKS experiments (2 jobs / GPU) ==="
ALL=""

# Exp 0 + Exp 1: RWSE and L-HKS share the same seed — natural pairing
for S in 0 1 2 3; do
    J=$(submit_pair "rwse_lhks_s${S}" \
        "configs/GPS/zinc-GPS+RWSE.yaml" $S "${OUTBASE}/exp0_rwse/seed${S}" \
        "configs/GPS/zinc-GPS+LHKS.yaml" $S "${OUTBASE}/exp1_lhks/seed${S}")
    ALL="${ALL}:${J}"; echo "  rwse_lhks_s${S} -> ${J}"
done

# Exp 4a: Fixed times — pair consecutive seeds
J=$(submit_pair "fixed_s01" \
    "configs/GPS/zinc-GPS+LHKS-fixed.yaml" 0 "${OUTBASE}/exp4a_fixed/seed0" \
    "configs/GPS/zinc-GPS+LHKS-fixed.yaml" 1 "${OUTBASE}/exp4a_fixed/seed1")
ALL="${ALL}:${J}"; echo "  fixed_s01 -> ${J}"

J=$(submit_pair "fixed_s23" \
    "configs/GPS/zinc-GPS+LHKS-fixed.yaml" 2 "${OUTBASE}/exp4a_fixed/seed2" \
    "configs/GPS/zinc-GPS+LHKS-fixed.yaml" 3 "${OUTBASE}/exp4a_fixed/seed3")
ALL="${ALL}:${J}"; echo "  fixed_s23 -> ${J}"

# Exp 4b: K ablations — K4+K8 together, K32+noPE together
J=$(submit_pair "K4_K8" \
    "configs/GPS/zinc-GPS+LHKS-K4.yaml"  0 "${OUTBASE}/exp4b_K4/seed0" \
    "configs/GPS/zinc-GPS+LHKS-K8.yaml"  0 "${OUTBASE}/exp4b_K8/seed0")
ALL="${ALL}:${J}"; echo "  K4_K8 -> ${J}"

J=$(submit_pair "K32_noPE" \
    "configs/GPS/zinc-GPS+LHKS-K32.yaml" 0 "${OUTBASE}/exp4b_K32/seed0" \
    "configs/GPS/zinc-GPS-noPE.yaml"      0 "${OUTBASE}/exp0b_noPE/seed0")
ALL="${ALL}:${J}"; echo "  K32_noPE -> ${J}"

DEPS="${ALL#:}"
echo ""
echo "Training jobs (8): $(echo $DEPS | tr ':' ' ')"

# Post-processing (runs after all training jobs finish)
POST=$(sbatch --parsable \
    --job-name="lhks_post_2per" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    --constraint=ampere \
    --time="${TIME_POST}" \
    --dependency="afterok:${DEPS}" \
    --output="${LOGDIR}/post_2per_%j.out" \
    --error="${LOGDIR}/post_2per_%j.err" \
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
echo "Logs:    tail -f ${LOGDIR}/rwse_lhks_s0_*.out"
echo "Results: ${OUTBASE}"
