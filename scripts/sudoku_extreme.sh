#!/bin/bash

run_name="lookahead_adaptive_thresh"
CUDA_VISIBLE_DEVICES=3 python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 \
eval_interval=5000 \
checkpoint_every_eval=False \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True \
arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=6 \
+run_name=${run_name} \
lookahead=True \
lookahead_eta=0.6 \
lookahead_mu=0.6
