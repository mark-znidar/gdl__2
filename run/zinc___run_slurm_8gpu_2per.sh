#!/bin/bash
# Submit a single Slurm job that runs the full L-HKS suite with 2 training
# processes per GPU (4 GPUs). Same result layout as run_8gpu.sh.
#
# Usage:
#   ./run/run_slurm_8gpu_2per.sh
# Or from repo root:
#   sbatch run/run_slurm_8gpu_2per.sh   # if you convert to sbatch-only (see below)
#
# Edit the CONFIG block before submitting.

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="gpu"
ACCOUNT=""
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# One node, 4 GPUs — enough for 8 concurrent training jobs (2 per GPU)
GRES="gpu:4"
CPUS=32
MEM="128G"
TIME_TRAIN="14:00:00"
TIME_POST="02:00:00"
# ============================================================
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# ============================================================

LOGDIR="${REPO_DIR}/slurm_logs"
mkdir -p "$LOGDIR"

ACCT_FLAG=""
[ -n "$ACCOUNT" ] && ACCT_FLAG="--account=${ACCOUNT}"

JOB=$(sbatch --parsable \
    --job-name="lhks_2per" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --time="${TIME_TRAIN}" \
    --output="${LOGDIR}/lhks_2per_%j.out" \
    --error="${LOGDIR}/lhks_2per_%j.err" \
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
echo \"=== Train (2/GPU): \$(hostname) | \$(date) ===\"
bash run/run_8gpu_2per.sh
echo \"=== Train finished: \$(date) ===\"
")

echo "Submitted training job: ${JOB}"
echo "Logs: ${LOGDIR}/lhks_2per_${JOB}.out"

POST=$(sbatch --parsable \
    --job-name="lhks_post" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    --time="${TIME_POST}" \
    --dependency="afterok:${JOB}" \
    --output="${LOGDIR}/lhks_post_2per_%j.out" \
    --error="${LOGDIR}/lhks_post_2per_%j.err" \
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
echo \"=== Post-processing: \$(date) ===\"
python experiments/run_stability.py --gpu 0
python experiments/run_sr25.py
python experiments/collect_results.py
python experiments/plot_results.py
echo \"=== Done: \$(date) ===\"
")

echo "Post-processing: ${POST} (afterok:${JOB})"
echo "Monitor: squeue -u \$USER"
