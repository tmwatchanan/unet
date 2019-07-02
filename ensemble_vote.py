import os
import csv
import natsort
import numpy as np
from termcolor import colored, cprint
from sklearn import metrics
import skimage
from scipy import stats

from check_accuracy import read_groundtruth
from keras.utils import to_categorical


experiment_names = ["s1", "s2", "s3", "s4", "s5"]


def evaluate():
    evaluation_csv_filename = "-".join(experiment_names) + ".csv"
    evaluation_csv_file = os.path.join("data", "evaluation", evaluation_csv_filename)

    data_path = "data"
    ensemble_path = os.path.join(data_path, "ensemble")

    TARGET_SIZE = (256, 256)

    output_summary_evaluation = []
    classes = ["iris", "sclera", "bg"]
    folds_label_image_pairs = {}
    for p_class in classes:
        folds_label_image_pairs[p_class] = {"label": np.empty(0), "image": np.empty(0)}
    fold_list = range(1, 4 + 1)
    test_accuracies = []
    for fold in fold_list:
        fold_dir = os.path.join(ensemble_path, f"fold_{fold}")

        groundtruths_dir = os.path.join(fold_dir, "labels")
        groundtruth_filenames = next(os.walk(groundtruths_dir))[2]
        groundtruth_filenames = natsort.natsorted(groundtruth_filenames, reverse=False)
        groundtruths = []
        for each_groundtruth_filename in groundtruth_filenames:
            groundtruth_path = os.path.join(groundtruths_dir, each_groundtruth_filename)
            groundtruth = read_groundtruth(groundtruth_path, TARGET_SIZE)
            groundtruths.append(groundtruth)

        segments_all = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], len(groundtruths), 0))
        for experiment_name in experiment_names:
            segments_dir = os.path.join(fold_dir, experiment_name)
            segment_filenames = next(os.walk(segments_dir))[2]
            segment_filenames = natsort.natsorted(segment_filenames, reverse=False)

            combined_segment = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 0))
            for each_segment_filename in segment_filenames:

                segment_path = os.path.join(segments_dir, each_segment_filename)
                segment = skimage.io.imread(segment_path)
                segment_class = np.argmax(segment, axis=2)
                segment_class = np.expand_dims(segment_class, axis=2)
                combined_segment = np.concatenate(
                    (combined_segment, segment_class), axis=2
                )
            combined_segment = np.expand_dims(combined_segment, axis=3)
            segments_all = np.concatenate((segments_all, combined_segment), axis=3)

        segment_modes = stats.mode(segments_all, axis=3)
        mode_values = segment_modes[0]
        mode_values = np.squeeze(mode_values, axis=3)

        categorical_mode_values = to_categorical(mode_values)
        voted_classes = [
            categorical_mode_values[:, :, i, :]
            for i in range(categorical_mode_values.shape[2])
        ]

        label_image_pairs = evaluate_classes(voted_classes, groundtruths)
        for p_class in classes:
            folds_label_image_pairs[p_class]["label"] = np.concatenate(
                (
                    folds_label_image_pairs[p_class]["label"],
                    label_image_pairs[p_class]["label"],
                ),
                axis=None,
            )
            folds_label_image_pairs[p_class]["image"] = np.concatenate(
                (
                    folds_label_image_pairs[p_class]["image"],
                    label_image_pairs[p_class]["image"],
                ),
                axis=None,
            )

        accuracies = []
        for groundtruth, segment in zip(groundtruths, voted_classes):
            acc_each_file = calculate_accuracy(groundtruth, segment, TARGET_SIZE)
            accuracies.append(acc_each_file)
        avg_accuracy = np.mean(accuracies)
        test_accuracies.append(avg_accuracy)

    for p_class in classes:
        precision = metrics.precision_score(
            folds_label_image_pairs[p_class]["label"],
            folds_label_image_pairs[p_class]["image"],
        )
        recall = metrics.recall_score(
            folds_label_image_pairs[p_class]["label"],
            folds_label_image_pairs[p_class]["image"],
        )
        f1 = metrics.f1_score(
            folds_label_image_pairs[p_class]["label"],
            folds_label_image_pairs[p_class]["image"],
        )
        output_summary_evaluation.append(precision)
        output_summary_evaluation.append(recall)
        output_summary_evaluation.append(f1)

    ordered_test_evaluation = []
    for test_acc in test_accuracies:
        ordered_test_evaluation.append("-")
        ordered_test_evaluation.append("-")
        ordered_test_evaluation.append(format_accuracy(test_acc))
        ordered_test_evaluation.append("-")

    output_summary_evaluation = np.concatenate(
        (output_summary_evaluation, ordered_test_evaluation), axis=None
    )

    with open(evaluation_csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(output_summary_evaluation)
    cprint(f"> evaluate succesfully, wrote file at ", end="")
    cprint(f"{evaluation_csv_file}", color="green", attrs=["bold"], end=".")


def evaluate_classes(images, groundtruths):
    label_image_pairs = {}
    classes = ["iris", "sclera", "bg"]
    for p_class in classes:
        label_image_pairs[p_class] = {"label": np.empty(0), "image": np.empty(0)}
    for image, label in zip(images, groundtruths):
        if image.shape != label.shape:
            print("Image's shape doesn't match with label's shape")
            exit(1)
        image_argmax = np.argmax(image, axis=2)
        label_argmax = np.argmax(label, axis=2)

        image = to_categorical(image_argmax)
        label = to_categorical(label_argmax)

        def extract_class_layers(image):
            iris = image[:, :, 0]
            sclera = image[:, :, 1]
            bg = image[:, :, 2]
            return iris, sclera, bg

        def flatten_class_layers(image):
            iris, sclera, bg = extract_class_layers(image)
            return iris.flatten(), sclera.flatten(), bg.flatten()

        flatten_label = {}
        flatten_image = {}

        flatten_label["iris"], flatten_label["sclera"], flatten_label[
            "bg"
        ] = flatten_class_layers(label)
        flatten_image["iris"], flatten_image["sclera"], flatten_image[
            "bg"
        ] = flatten_class_layers(image)

        for p_class in classes:
            label_image_pairs[p_class]["label"] = np.concatenate(
                (label_image_pairs[p_class]["label"], flatten_label[p_class]), axis=None
            )
            label_image_pairs[p_class]["image"] = np.concatenate(
                (label_image_pairs[p_class]["image"], flatten_image[p_class]), axis=None
            )
    return label_image_pairs


def format_accuracy(number):
    return format(number * 100, "3.2f")


def calculate_accuracy(groundtruth, segment, target_size):
    argmax_groundtruth = np.argmax(groundtruth, axis=-1)
    argmax_segment = np.argmax(segment, axis=-1)
    true_count = (argmax_groundtruth == argmax_segment).sum()
    pixel_count = target_size[0] * target_size[1]
    accuracy = true_count / pixel_count
    return accuracy


if __name__ == "__main__":
    evaluate()
