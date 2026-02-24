#!/bin/bash
set -e
TRAIN=./data/encoded/tinystories_train.npy
VAL=./data/encoded/tinystories_valid.npy
mkdir -p logs

for BS in 64 128 256; do
    RUN="bs_${BS}"
    uv run python3 -m cs336_basics.training \
        --train_path $TRAIN --val_path $VAL \
        --lr_max 1e-2 --lr_min 1e-3 \
        --batch_size $BS \
        --run_name $RUN --log_dir logs
done
