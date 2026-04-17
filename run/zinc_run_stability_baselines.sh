#!/bin/bash
# Train 3 PE-baseline checkpoints on ZINC for the perturbation/stability study:
#   LapPE, SignNet-MLP, SignNet-DeepSet
# One seed each (matches how LHKS/RWSE stability is run in run_stability.py).
#
# After all 3 finish, remember to:
#   1) add entries to experiments/run_stability.py -> `specs` list for these
#      method names and seed paths (see SEED0 below), or reuse
#      find_run_dir_with_ckpt on the printed paths.
#   2) run `python experiments/run_stability.py --gpu 0`

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
# ============================================================
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# ============================================================

OUTBASE="${REPO_DIR}/results/zinc_pe_baselines"
LOGDIR="${REPO_DIR}/slurm_logs_zinc_baselines"
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

SEED0=0
echo "=== Submitting ZINC PE-baseline jobs for stability study ==="
ALL=""

J=$(submit "zinc_lappe_s${SEED0}" \
    "configs/GPS/zinc-GPS+LapPE.yaml" $SEED0 \
    "${OUTBASE}/lappe/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  zinc_lappe_s${SEED0} -> ${J}  (out: ${OUTBASE}/lappe/seed${SEED0})"

J=$(submit "zinc_snmlp_s${SEED0}" \
    "configs/GPS/zinc-GPS+SNMLP.yaml" $SEED0 \
    "${OUTBASE}/snmlp/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  zinc_snmlp_s${SEED0} -> ${J}  (out: ${OUTBASE}/snmlp/seed${SEED0})"

J=$(submit "zinc_snds_s${SEED0}" \
    "configs/GPS/zinc-GPS+SNDS.yaml" $SEED0 \
    "${OUTBASE}/snds/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  zinc_snds_s${SEED0} -> ${J}  (out: ${OUTBASE}/snds/seed${SEED0})"

DEPS="${ALL#:}"
echo ""
echo "Training jobs (3): $(echo $DEPS | tr ':' ' ')"
echo ""
echo "Checkpoints (after training):"
echo "  LapPE           -> ${OUTBASE}/lappe/seed${SEED0}/zinc-GPS+LapPE/${SEED0}/ckpt/"
echo "  SignNet-MLP     -> ${OUTBASE}/snmlp/seed${SEED0}/zinc-GPS+SNMLP/${SEED0}/ckpt/"
echo "  SignNet-DeepSet -> ${OUTBASE}/snds/seed${SEED0}/zinc-GPS+SNDS/${SEED0}/ckpt/"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/zinc_lappe_s${SEED0}_*.out"
