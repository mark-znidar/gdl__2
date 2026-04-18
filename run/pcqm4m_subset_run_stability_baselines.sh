#!/bin/bash
# Train 4 PE-baseline checkpoints on PCQM4Mv2-Subset for the perturbation/stability study:
#   LapPE, SignNet-MLP, SignNet-DeepSets, noPE
# One seed each (matches how LHKS/RWSE stability is run in experiments/run_stability.py).
# noPE is the "no positional encoding" reference; Exp 1 (edge removal) still gives
# a signal (pure MPNN sensitivity to topology edits), Exp 2 (eigenvector rotation)
# is trivially zero for it since no PE is computed -- we exclude it from Exp 2 plots.
#
# After all 3 finish, remember to:
#   1) add entries to experiments/run_stability.py -> `specs` list for these
#      method names and seed paths (see SEED0 below), or reuse
#      find_run_dir_with_ckpt on the printed paths.
#   2) run the stability script on PCQM4Mv2-Subset test graphs.

# ============================================================
# CLUSTER CONFIG — EDIT THESE
# ============================================================
PARTITION="il"
ACCOUNT="infolab"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRES="gpu:1"
CPUS=12
# PCQM4Mv2 needs up to 48GB system RAM per paper Appendix A.4.
MEM="48G"
TIME_TRAIN="12:00:00"
# ============================================================
ACTIVATE_CMD="export PATH=${REPO_DIR}/.pixi/envs/default/bin:\$PATH"
# ============================================================

OUTBASE="${REPO_DIR}/results_pcqm4m_subset/stability_baselines"
LOGDIR="${REPO_DIR}/slurm_logs_pcqm4m_subset_baselines"
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

SEED0=1
echo "=== Submitting PCQM4Mv2-Subset PE-baseline jobs for stability study ==="
ALL=""

J=$(submit "pcqm_lappe_s${SEED0}" \
    "configs/GPS/pcqm4m-subset-GPS+LapPE.yaml" $SEED0 \
    "${OUTBASE}/lappe/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  pcqm_lappe_s${SEED0} -> ${J}  (out: ${OUTBASE}/lappe/seed${SEED0})"

J=$(submit "pcqm_snmlp_s${SEED0}" \
    "configs/GPS/pcqm4m-subset-GPS+SNMLP.yaml" $SEED0 \
    "${OUTBASE}/snmlp/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  pcqm_snmlp_s${SEED0} -> ${J}  (out: ${OUTBASE}/snmlp/seed${SEED0})"

J=$(submit "pcqm_snds_s${SEED0}" \
    "configs/GPS/pcqm4m-subset-GPS+SNDS.yaml" $SEED0 \
    "${OUTBASE}/snds/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  pcqm_snds_s${SEED0} -> ${J}  (out: ${OUTBASE}/snds/seed${SEED0})"

J=$(submit "pcqm_nope_s${SEED0}" \
    "configs/GPS/pcqm4m-subset-GPS-noPE.yaml" $SEED0 \
    "${OUTBASE}/nope/seed${SEED0}")
ALL="${ALL}:${J}"; echo "  pcqm_nope_s${SEED0} -> ${J}  (out: ${OUTBASE}/nope/seed${SEED0})"

DEPS="${ALL#:}"
echo ""
echo "Training jobs (4): $(echo $DEPS | tr ':' ' ')"
echo ""
echo "Checkpoints (after training):"
echo "  LapPE              -> ${OUTBASE}/lappe/seed${SEED0}/pcqm4m-subset-GPS+LapPE/${SEED0}/ckpt/"
echo "  SignNet-MLP        -> ${OUTBASE}/snmlp/seed${SEED0}/pcqm4m-subset-GPS+SNMLP/${SEED0}/ckpt/"
echo "  SignNet-DeepSets   -> ${OUTBASE}/snds/seed${SEED0}/pcqm4m-subset-GPS+SNDS/${SEED0}/ckpt/"
echo "  noPE               -> ${OUTBASE}/nope/seed${SEED0}/pcqm4m-subset-GPS-noPE/${SEED0}/ckpt/"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOGDIR}/pcqm_lappe_s${SEED0}_*.out"
