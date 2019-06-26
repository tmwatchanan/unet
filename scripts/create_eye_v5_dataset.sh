#!/bin/bash

set_1_filenames=(E-1-1.jpg E-2-2.jpg E-7-1.jpg E-9-2.jpg E-16-4.jpg E-21-2.jpg E-22-2.jpg E-23-2.jpg 0956134_20170331__R_a1.jpg 2976988_20161216_a1.jpg 1387253_20170303_b2-1.jpg 1426104_20161223__Image_R_c1-1.jpg 1790390_20170303_after_c1.jpg 1790390_20170303_before_b1-1.jpg 1934357_20170106__R_a3.jpg 2011248_20161215__L_b2.jpg)
set_2_filenames=(E-1-2.jpg E-2-1.jpg E-3-1.jpg E-13-4.jpg E-18-3.jpg E-21-4.jpg E-24-3.jpg E-25-1.jpg 0956134_20170331__R_c1.jpg 2011248_20161215__R_b1.jpg 3792354_20170428__L_a3.jpg 01102560_3.jpg 1426104_20161223__Image_R_a2.jpg 2438008_20161223__R_b3.jpg 2446177_20170428__L_b1-3.jpg 2892188_20161125_145119_b3.jpg)
set_3_filenames=(E-1-3.jpg E-4-4.jpg E-8-4.jpg E-10-3.jpg E-12-4.jpg E-14-2.jpg E-18-2.jpg E-30-1.jpg 0956134_20170331__R_b1.jpg 3792354_20170428_b1_L_01.jpg 0827849_20161216__L_c1.jpg 1790390_20170303_a1.jpg 2203721_20170113__L_b1-1.jpg 2446177_20170428__Lc1-1.jpg 2725606_20170310_a3_R_01.jpg 2976988_20161215_a3.jpg)
set_4_filenames=(E-1-4.jpg E-5-2.jpg E-6-3.jpg E-11-2.jpg E-15-3.jpg E-20-2.jpg E-20-3.jpg E-21-1.jpg R3.jpg 0328944_a1.jpg 0956134_20170331__R_b2.jpg 1387253_20170303_after_b2.jpg 2011248_20161215__R_b3.jpg 2238864_20170428_c3_L_01.jpg 2323956_20161223__R_a1.jpg 2930625_20170106__R_b3.jpg)

function create_images_set() {
    for i in {1..4}; do
        set_dir="set_$i"
        set_images_dir="$set_dir/images"
        set_labels_dir="$set_dir/labels"
        mkdir -p $set_images_dir
        mkdir -p $set_labels_dir
        set_filenames="${set_dir}_filenames"
        echo $set_filenames "-------------"
        files=$set_filenames[@]
        for image_filename in ${!files}; do
            echo $image_filename
            cp "images/$image_filename" $set_images_dir
            cp "labels/$image_filename" $set_labels_dir
        done
    done
}

function create_fold() {
    epoch=1
    echo $epoch
    fold_name="fold_$epoch"
    training_dir="$fold_name/train"
    validation_dir="$fold_name/validation"
    test_dir="$fold_name/test"
    mkdir -p $training_dir
    mkdir -p $validation_dir
    mkdir -p $test_dir
    cp -r set_1/* "$training_dir"
    cp -r set_2/* "$training_dir"
    cp -r set_3/* "$validation_dir"
    cp -r set_4/* "$test_dir"

    epoch=2
    echo $epoch
    fold_name="fold_$epoch"
    training_dir="$fold_name/train"
    validation_dir="$fold_name/validation"
    test_dir="$fold_name/test"
    mkdir -p $training_dir
    mkdir -p $validation_dir
    mkdir -p $test_dir
    cp -r set_2/* "$training_dir"
    cp -r set_3/* "$training_dir"
    cp -r set_4/* "$validation_dir"
    cp -r set_1/* "$test_dir"

    epoch=3
    echo $epoch
    fold_name="fold_$epoch"
    training_dir="$fold_name/train"
    validation_dir="$fold_name/validation"
    test_dir="$fold_name/test"
    mkdir -p $training_dir
    mkdir -p $validation_dir
    mkdir -p $test_dir
    cp -r set_3/* "$training_dir"
    cp -r set_4/* "$training_dir"
    cp -r set_1/* "$validation_dir"
    cp -r set_2/* "$test_dir"

    epoch=4
    echo $epoch
    fold_name="fold_$epoch"
    training_dir="$fold_name/train"
    validation_dir="$fold_name/validation"
    test_dir="$fold_name/test"
    mkdir -p $training_dir
    mkdir -p $validation_dir
    mkdir -p $test_dir
    cp -r set_4/* "$training_dir"
    cp -r set_1/* "$training_dir"
    cp -r set_2/* "$validation_dir"
    cp -r set_3/* "$test_dir"
}

cd $1
$2
# create_fold
