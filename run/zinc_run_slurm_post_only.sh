#!/bin/bash
# Post-processing only: stability, SR25, results table, figures.
# Use after training finished (e.g. results under OUTBASE from run_slurm_2seed.sh).
# Does not submit training jobs or Slurm dependencies.

# ============================================================
# CLUSTER CONFIG — EDIT THESE (keep in sync with run_slurm_2seed.sh)
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
MEM="42G"
TIME_POST="02:00:00"
# Where checkpoints live (override: OUTBASE=/path bash run/run_slurm_post_only.sh)
OUTBASE="${OUTBASE:-${REPO_DIR}/results/slurm1_2seed}"
# ============================================================
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# ============================================================

LOGDIR="${REPO_DIR}/slurm_logs"
mkdir -p "$LOGDIR"

ACCT_FLAG=""
[ -n "$ACCOUNT" ] && ACCT_FLAG="--account=${ACCOUNT}"

JOB=$(sbatch --parsable \
    --job-name="lhks_post_only" \
    --partition="${PARTITION}" ${ACCT_FLAG} \
    --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --constraint=ampere \
    --time="${TIME_POST}" \
    --output="${LOGDIR}/post_only_%j.out" \
    --error="${LOGDIR}/post_only_%j.err" \
    --wrap="
cd ${REPO_DIR}
export EXPERIMENT_OUTBASE=${OUTBASE}
export XDG_CACHE_HOME=/dfs/user/mznidar/.cache
export TORCH_HOME=\${XDG_CACHE_HOME}/torch
export HF_HOME=\${XDG_CACHE_HOME}/huggingface
export PIP_CACHE_DIR=\${XDG_CACHE_HOME}/pip
export PIXI_CACHE_DIR=/dfs/scratch0/mznidar/pixi-cache
export TMPDIR=/dfs/scratch0/mznidar/tmp
mkdir -p \$TMPDIR \$XDG_CACHE_HOME
${ACTIVATE_CMD}
echo \"=== Post-only | Node: \$(hostname) | EXPERIMENT_OUTBASE=\${EXPERIMENT_OUTBASE} | \$(date) ===\"
python experiments/run_stability.py --gpu 0
python experiments/run_sr25.py
python experiments/collect_results.py
python experiments/plot_results.py
echo \"=== Done: \$(date) ===\"
")

echo "Submitted post-only job: ${JOB}"
echo "Logs: ${LOGDIR}/post_only_${JOB}.out | ${LOGDIR}/post_only_${JOB}.err"
echo "Monitor: squeue -j ${JOB} -u \$USER"
