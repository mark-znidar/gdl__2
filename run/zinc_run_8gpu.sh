#!/bin/bash
set -e
OUTBASE="results"
echo "=== L-HKS experiments (8 GPU) | Start: $(date) ==="

# Phase 1: baselines on GPUs 0-7
for S in 0 1 2 3; do
    ( CUDA_VISIBLE_DEVICES=$S python main.py \
        --cfg configs/GPS/zinc-GPS+RWSE.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp0_rwse/seed${S} ) &
done
for S in 0 1 2 3; do
    GPU=$((S + 4))
    ( CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp1_lhks/seed${S} ) &
done
wait
echo "=== Phase 1 done: $(date) ==="

# Phase 2: ablations on GPUs 0-7
for S in 0 1 2 3; do
    ( CUDA_VISIBLE_DEVICES=$S python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp4a_fixed/seed${S} ) &
done
( CUDA_VISIBLE_DEVICES=4 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K4.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K4/seed0 ) &
( CUDA_VISIBLE_DEVICES=5 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K8.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K8/seed0 ) &
( CUDA_VISIBLE_DEVICES=6 python main.py \
    --cfg configs/GPS/zinc-GPS+LHKS-K32.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K32/seed0 ) &
( CUDA_VISIBLE_DEVICES=7 python main.py \
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
