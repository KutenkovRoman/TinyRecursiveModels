#!/bin/bash

run_name="whatever"
CUDA_VISIBLE_DEVICES=1 python pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
global_batch_size=512 \
epochs=50000 \
eval_interval=5000 \
checkpoint_every_eval=False \
optim="adamw" \
lr=1e-4 \
lr_warmup_steps=2200 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=4 \
+run_name=${run_name} \
ema=True \
use_wandb=False

#global_batch_size=384 \
