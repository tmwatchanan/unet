import numpy as np
import os
import skimage
import glob


def read_groundtruth(groundtruth_filename, target_size):
    groundtruth = skimage.io.imread(groundtruth_filename)
    groundtruth = preprocess_image(groundtruth, target_size)
    return groundtruth


def preprocess_image(groundtruth, target_size):
    output_shape = (target_size[0], target_size[1], 3)
    groundtruth = skimage.transform.resize(groundtruth, output_shape, order=0)
    # groundtruth = groundtruth.astype("uint8")
    return groundtruth


def main():
    segment_path = os.path.join(
        "datasets", "eye_v5-s1", "fold_1", "test", "segments", "s3"
    )
    segment_filenames = get_filenames_from_dir(segment_path)
    groundtruth_path = os.path.join("datasets", "eye_v5-s1", "fold_1", "test", "labels")
    groundtruth_filenames = get_filenames_from_dir(groundtruth_path)

    accuracies = []
    for groundtruth_filename, segment_filename in zip(
        groundtruth_filenames, segment_filenames
    ):
        segment = skimage.io.imread(segment_filename)
        target_size = segment.shape
        groundtruth = read_groundtruth(groundtruth_filename, target_size)

        accuracy = calculate_accuracy(groundtruth, segment, target_size)
        accuracies.append(accuracy)
        print(f"{segment_filename} = {format_accuracy(accuracy)}")

    print(f"accuracy = {format_accuracy(np.mean(accuracies))}")


def get_filenames_from_dir(path):
    filenames = sorted(glob.glob((path + "/*")))
    return filenames


def calculate_accuracy(groundtruth, segment, target_size):
    argmax_groundtruth = np.argmax(groundtruth, axis=-1)
    argmax_segment = np.argmax(segment, axis=-1)
    true_count = (argmax_groundtruth == argmax_segment).sum()
    pixel_count = target_size[0] * target_size[1]
    accuracy = true_count / pixel_count
    return accuracy


def format_accuracy(number):
    return format(number * 100, "3.2f")


if __name__ == "__main__":
    main()
