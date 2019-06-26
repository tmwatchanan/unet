import numpy as np
import os
import skimage
import glob

segment_path = os.path.join("datasets", "eye_v5-s1", "fold_1", "test", "segments", "2")
segment_filenames = sorted(glob.glob((segment_path + "/*")))
groundtruth_path = os.path.join("datasets", "eye_v5-s1", "fold_1", "test", "labels")
groundtruth_filenames = sorted(glob.glob((groundtruth_path + "/*")))

total_true = 0
total_pixel = 0

accuracies = []

for groundtruth_filename, segment_filename in zip(
    groundtruth_filenames, segment_filenames
):
    segment = skimage.io.imread(segment_filename)
    groundtruth = skimage.io.imread(groundtruth_filename)
    target_size = segment.shape
    output_shape = (target_size[0], target_size[1], 3)
    groundtruth = skimage.transform.resize(groundtruth, output_shape)
    groundtruth[groundtruth > 0.5] = 255
    groundtruth = groundtruth.astype("uint8")

    equal_count = (groundtruth == segment).all(axis=2)
    true_count = (equal_count == True).sum()

    pixel_count = target_size[0] * target_size[1]

    accuracy = true_count / pixel_count
    accuracies.append(accuracy)
    print(f"{segment_filename} = {accuracy}")

print(f"accuracy = {np.mean(accuracies)}")
