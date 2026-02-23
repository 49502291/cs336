#!/bin/bash
set -e

TRAIN=/data/encoded/tinystories_train.npy
VAL=/data/encoded/tinystories_valid.npy

mkdir -p logs

for LR in 3e-4 3e-3 1e-2; do
    RUN="lr_${LR}"
    LR_MIN=$(python3 -c "print($LR / 10)")
    echo "========================================"
    echo "Starting run: $RUN  (lr_max=$LR  lr_min=$LR_MIN)"
    echo "========================================"
    uv run python3 -m cs336_basics.training \
        --train_path $TRAIN \
        --val_path $VAL \
        --device cuda \
        --lr_max $LR \
        --lr_min $LR_MIN \
        --run_name $RUN \
        --log_dir logs \
        2>&1 | tee logs/${RUN}.txt
    echo "Done: $RUN"
done

echo "All runs complete. Logs in logs/"
