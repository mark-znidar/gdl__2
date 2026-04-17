#!/bin/bash
set -e
OUTBASE="results"
echo "=== L-HKS experiments (4 GPU) | Start: $(date) ==="

# GPU 0: RWSE baseline (4 seeds)
(
for S in 0 1 2 3; do
    echo "[GPU 0] RWSE seed $S"
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --cfg configs/GPS/zinc-GPS+RWSE.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp0_rwse/seed${S}
done
) &

# GPU 1: L-HKS learned (4 seeds)
(
for S in 0 1 2 3; do
    echo "[GPU 1] L-HKS seed $S"
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp1_lhks/seed${S}
done
) &

# GPU 2: L-HKS fixed (4 seeds)
(
for S in 0 1 2 3; do
    echo "[GPU 2] Fixed seed $S"
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS-fixed.yaml \
        --repeat 1 seed $S out_dir ${OUTBASE}/exp4a_fixed/seed${S}
done
) &

# GPU 3: K ablations + no-PE
(
for K in 4 8 32; do
    echo "[GPU 3] K=$K"
    CUDA_VISIBLE_DEVICES=3 python main.py \
        --cfg configs/GPS/zinc-GPS+LHKS-K${K}.yaml \
        --repeat 1 seed 0 out_dir ${OUTBASE}/exp4b_K${K}/seed0
done
echo "[GPU 3] No-PE"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --cfg configs/GPS/zinc-GPS-noPE.yaml \
    --repeat 1 seed 0 out_dir ${OUTBASE}/exp0b_noPE/seed0
) &

wait
echo "=== Training done: $(date) ==="
echo "Running post-processing..."
python experiments/run_stability.py --gpu 0
python experiments/run_sr25.py
python experiments/collect_results.py
python experiments/plot_results.py
echo "=== All done: $(date) ==="
