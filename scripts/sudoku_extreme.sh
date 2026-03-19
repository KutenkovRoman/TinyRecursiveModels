#!/bin/bash

run_name="galore_norms_monitoring"
CUDA_VISIBLE_DEVICES=2 python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 \
eval_interval=5000 \
checkpoint_every_eval=False \
optim="galore" \
lr=1e-4 \
lr_warmup_steps=2000 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True \
arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=6 \
+run_name=${run_name} \
ema=True

#lr_min_ratio=0.1 \
