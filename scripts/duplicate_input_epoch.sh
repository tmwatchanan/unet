#!/bin/bash

cd $1
image_filenames=($(ls 2900))
for image_filename in "${image_filenames[@]}"; do
    echo $image_filename
    for epoch in $(seq -f "%08g" 500 100 5000); do
        cp 2900/$image_filename ${image_filename/00002900/$epoch}
    done
done
