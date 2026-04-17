#!/bin/bash
# Same experiment suite as run_8gpu.sh, but runs TWO training processes per GPU
# to better fill A100s when each job uses little VRAM (~0.7 GiB).
#
# Phase 1: 8 jobs on GPUs 0–3 (RWSE seed S + LHKS seed S on each GPU).
# Phase 2: 8 jobs on GPUs 0–3 (paired fixed seeds; K4+K8; K32+noPE).
#
# Requires 4 physical GPUs visible as 0–3 (set CUDA_VISIBLE_DEVICES if needed).
set -e
OUTBASE="${OUTBASE:-results}"
echo "=== L-HKS experiments (2 jobs / GPU, 4 GPUs) | Start: $(date) ==="

# Phase 1: RWSE + LHKS same seed share one GPU
for S in 0 1 2 3; do
    ( CUDA_VISIBLE_DEVICES=$S python main.py \
        --cfg configs/GPS/zinc-GPS+RWSE.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp0_rwse/seed${S} ) &
    ( CUDA_VISIBLE_DEVICES=$S python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp1_lhks/seed${S} ) &
done
wait
echo "=== Phase 1 done: $(date) ==="

# Phase 2: four fixed seeds on 0–1 (two per GPU); K4+K8 on 2; K32+noPE on 3
( CUDA_VISIBLE_DEVICES=0 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4a_fixed/seed0 ) &
( CUDA_VISIBLE_DEVICES=0 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
    --repeat 1 seed 1 out_dir ${OUTBASE}/exp4a_fixed/seed1 ) &
( CUDA_VISIBLE_DEVICES=1 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
    --repeat 1 seed 2 out_dir ${OUTBASE}/exp4a_fixed/seed2 ) &
( CUDA_VISIBLE_DEVICES=1 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
    --repeat 1 seed 3 out_dir ${OUTBASE}/exp4a_fixed/seed3 ) &
( CUDA_VISIBLE_DEVICES=2 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K4.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K4/seed0 ) &
( CUDA_VISIBLE_DEVICES=2 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K8.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K8/seed0 ) &
( CUDA_VISIBLE_DEVICES=3 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K32.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K32/seed0 ) &
( CUDA_VISIBLE_DEVICES=3 python main.py \
    --cfg configs/GPS/zinc-GPS-noPE.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp0b_noPE/seed0 ) &
wait
echo "=== Phase 2 done: $(date) ==="

echo "Running post-processing..."
python experiments/run_stability.py --gpu 0
python experiments/run_sr25.py
python experiments/collect_results.py
python experiments/plot_results.py
echo "=== All done: $(date) ==="
