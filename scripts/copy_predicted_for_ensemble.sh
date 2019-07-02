#!/bin/bash

s1="eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_{fold}-lr_1e_2-bn"
s2="eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_{fold}-lr_1e_2-bn"
s3="eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_{fold}-lr_1e_2-bn"
s4="eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_{fold}-lr_1e_2-bn"
s5="eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_{fold}-lr_1e_2-bn"

function copy() {
    data_dir="data"
    ensemble_dir="$data_dir/ensemble"
    for i in {1..4}; do
        fold_dir="$ensemble_dir/fold_$i"
        for s in {1..5}; do
            si="s$s"

            model_fold_name="${!si}"
            model_fold_name="${model_fold_name/\{fold\}/$i}"
            predicted_dir="$data_dir/$model_fold_name"

            si_dir="$fold_dir/$si"
            mkdir -p "$si_dir"

            # copy all predicted segment images to s?
            cp $predicted_dir/test-predicted/*.png "$si_dir/"
        done

        labels_src_dir="datasets/eye_v5/fold_$i/test/labels/"
        labels_dest_dir="$fold_dir/labels"
        mkdir -p "$labels_dest_dir"
        cp $labels_src_dir/*.jpg $labels_dest_dir
    done

}

$1
