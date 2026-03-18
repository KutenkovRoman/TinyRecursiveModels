#!/bin/bash

run_name="prodigy_no_warmup"
CUDA_VISIBLE_DEVICES=4 python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 \
eval_interval=5000 \
checkpoint_every_eval=False \
optim="prodigy" \
lr=1e-3 \
lr_warmup_steps=0 \
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

#arch.halt_max_steps=8 \
#arch.halt_exploration_prob=0.0 \
#+load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/muon_retry_with_checkpointing/step_32550" \
