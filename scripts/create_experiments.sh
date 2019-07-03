#!/bin/bash

# USAGE: bash scripts/create_experiments.sh datasets/eye_v5

cp -r "$1/fold_1" "data/eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_2-bn"
cp -r "$1/fold_2" "data/eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_2-bn"
cp -r "$1/fold_3" "data/eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_2-bn"
cp -r "$1/fold_4" "data/eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_2-bn"
