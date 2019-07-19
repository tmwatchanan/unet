import csv
import datetime
import importlib
import os
import textwrap
from enum import Enum

import natsort
import numpy as np
import skimage
import tensorflow as tf
from keras.utils import to_categorical
from scipy import stats
from sklearn import metrics
from termcolor import colored, cprint

from check_accuracy import (
    calculate_accuracy,
    format_accuracy,
    preprocess_image,
    read_groundtruth,
)
from utils import max_rgb_filter
from ensemble_vote import evaluate_classes


class EnsembleMode(Enum):
    summation = 1
    product = 2


experiment_pool = {
    "eye_v5-model_v12-rgb": {
        "abbreviation": "s1",
        "input_size": (256, 256, 3),
        "color_model": "rgb",
        "file": "model_v12_cv",
        "dataset": "eye_v5",
        "experiments": [
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                4138,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                4468,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                3396,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                3445,
            ),
        ],
    },
    "eye_v5-model_v12-hsv": {
        "abbreviation": "s2",
        "input_size": (256, 256, 3),
        "color_model": "hsv",
        "file": "model_v12_cv",
        "dataset": "eye_v5",
        "experiments": [
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                3722,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                4884,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                4971,
            ),
            (
                "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                4283,
            ),
        ],
    },
    "eye_v5-model_v15-hsv": {
        "abbreviation": "s3",
        "input_size": (256, 256, 2),
        "color_model": "hsv",
        "file": "model_v15_cv",
        "dataset": "eye_v5",
        "experiments": [
            (
                "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                2637,
            ),
            (
                "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                4811,
            ),
            (
                "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                4122,
            ),
            (
                "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                4388,
            ),
        ],
    },
    "eye_v5-model_v13-rgb": {
        "abbreviation": "s4",
        "input_size": (256, 256, 5),
        "color_model": "rgb",
        "file": "model_v13_cv",
        "dataset": "eye_v5",
        "experiments": [
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                4909,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                4619,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                4692,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                4399,
            ),
        ],
    },
    "eye_v5-model_v13-hsv": {
        "abbreviation": "s5",
        "input_size": (256, 256, 5),
        "color_model": "hsv",
        "file": "model_v13_cv",
        "dataset": "eye_v5",
        "experiments": [
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                4684,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                4513,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                4111,
            ),
            (
                "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                4886,
            ),
        ],
    },
    "eye_v5-model_v28-hsv": {
        "abbreviation": "s6",
        "input_size": (256, 256, 8),
        "color_model": None,
        "file": "model_v28_cv",
        "dataset": "eye_v5",
        "experiments": [
            ("eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_2-bn", 4186),
            ("eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_2-bn", 4717),
            ("eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_2-bn", 4737),
            ("eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_2-bn", 4185),
        ],
    },
}

create_model = None
get_test_data = None
get_train_data = None
test_generator = None
train_generator = None


def import_model_functions(filename):
    try:
        module = importlib.import_module(filename)
        global create_model
        global get_test_data
        global get_train_data
        global test_generator
        global train_generator
        create_model = module.create_model
        get_test_data = module.get_test_data
        get_train_data = module.get_train_data
        test_generator = module.test_generator
        train_generator = module.train_generator
    except ImportError:
        print("\n[ok]: I don't know how to - " + module + ".\n")
        raise ("EXIT_FAILURE")


def predict():
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    BATCH_NORMALIZATION = True
    PREDICT_VERBOSE = 1  # 0 = silent, 1

    experiment_names = ["s1", "s2", "s3", "s4", "s5", "s6"]
    ENSEMBLE_MODE = EnsembleMode.product

    data_path = "data"
    dataset_path = "datasets"

    output_summary_evaluation = []
    classes = ["iris", "sclera", "bg"]
    folds_label_image_pairs = initialize_fold_classes(classes)
    test_accuracies = []
    for fold in range(1, 4 + 1):

        test_dir_names = ["test"]
        for test_dir_name in test_dir_names:

            predicted_outputs = {}
            for NAME in experiment_pool:
                experiment = experiment_pool[NAME]
                if experiment["abbreviation"] not in experiment_names:
                    continue

                import_model_functions(experiment["file"])
                input_size = experiment["input_size"]
                color_model = experiment["color_model"]
                input_dataset_name = experiment["dataset"]
                experiment_epoch_pairs = experiment["experiments"][fold - 1]
                model_name = experiment_epoch_pairs[0]
                epoch = experiment_epoch_pairs[1]

                fold_dirname = f"fold_{fold}"
                fold_path = os.path.join(dataset_path, input_dataset_name, fold_dirname)
                set_path = os.path.join(fold_path, test_dir_name)
                set_labels_path = os.path.join(set_path, "labels")

                """
                load pretrained model
                """
                model_path = os.path.join(data_path, model_name)
                weights_dir = os.path.join(model_path, "weights")
                trained_weights_filename = f"{epoch:08d}.hdf5"
                trained_weights_file = os.path.join(
                    weights_dir, trained_weights_filename
                )

                model = create_model(
                    pretrained_weights=trained_weights_file,
                    input_size=input_size,
                    num_classes=NUM_CLASSES,
                    batch_normalization=BATCH_NORMALIZATION,
                    is_summary=False,
                )

                test_data_dict = dict(test_path=set_path, target_size=TARGET_SIZE)
                if "v28" not in NAME:
                    test_data_dict["image_color"] = color_model
                test_flow, test_files = get_test_data(**test_data_dict)
                if "v28" not in NAME:
                    test_gen = test_generator(test_flow, color_model)
                else:
                    test_gen = test_generator(test_flow)

                predict_steps = len(test_files)
                results = model.predict_generator(
                    test_gen, steps=predict_steps, verbose=PREDICT_VERBOSE
                )
                segment_results = results[0]
                predicted_outputs[NAME] = segment_results

            """
            Groundtruth
            """
            groundtruths = []
            for test_file in test_files:
                _, file = os.path.split(test_file)
                groundtruth_path = os.path.join(set_labels_path, file)
                groundtruth = read_groundtruth(groundtruth_path, TARGET_SIZE)
                groundtruths.append(groundtruth)

            predicted_confidences = ensemble_combine(
                len(groundtruths), ENSEMBLE_MODE, TARGET_SIZE, predicted_outputs
            )

            label_image_pairs = evaluate_classes(predicted_confidences, groundtruths)
            concat_classes(classes, folds_label_image_pairs, label_image_pairs)

            accuracies = []
            for groundtruth, segment in zip(groundtruths, predicted_confidences):
                acc_each_file = calculate_accuracy(groundtruth, segment, TARGET_SIZE)
                accuracies.append(acc_each_file)
            avg_accuracy = np.mean(accuracies)
            test_accuracies.append(avg_accuracy)

    for p_class in classes:
        precision = get_precision(folds_label_image_pairs, p_class)
        recall = get_recall(folds_label_image_pairs, p_class)
        f1 = get_f1(folds_label_image_pairs, p_class)
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

    evaluation_path = os.path.join(data_path, "evaluation")
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    evaluation_csv_filename = (
        "_".join(experiment_names) + f"-{ENSEMBLE_MODE.name}" + ".csv"
    )
    evaluation_csv_file = os.path.join("data", "evaluation", evaluation_csv_filename)
    with open(evaluation_csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(output_summary_evaluation)
    cprint(f"> evaluate succesfully, wrote file at ", end="")
    cprint(f"{evaluation_csv_file}", color="green", attrs=["bold"], end=".")


def get_f1(folds_label_image_pairs, p_class):
    f1 = metrics.f1_score(
        folds_label_image_pairs[p_class]["label"],
        folds_label_image_pairs[p_class]["image"],
    )
    return f1


def get_recall(folds_label_image_pairs, p_class):
    recall = metrics.recall_score(
        folds_label_image_pairs[p_class]["label"],
        folds_label_image_pairs[p_class]["image"],
    )
    return recall


def get_precision(folds_label_image_pairs, p_class):
    precision = metrics.precision_score(
        folds_label_image_pairs[p_class]["label"],
        folds_label_image_pairs[p_class]["image"],
    )
    return precision


def initialize_fold_classes(classes):
    folds_label_image_pairs = {}
    for p_class in classes:
        folds_label_image_pairs[p_class] = {"label": np.empty(0), "image": np.empty(0)}
    return folds_label_image_pairs


def concat_classes(classes, folds_label_image_pairs, label_image_pairs):
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


def ensemble_combine(num_files, ensemble_mode, TARGET_SIZE, predicted_outputs):
    predicted_confidences = []
    for g_index in range(num_files):
        if ensemble_mode == EnsembleMode.summation:
            image_confidences = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3))
        elif ensemble_mode == EnsembleMode.product:
            image_confidences = np.ones((TARGET_SIZE[0], TARGET_SIZE[1], 3))
        for model_output_name in predicted_outputs:
            output_confidence = predicted_outputs[model_output_name][g_index]
            if ensemble_mode == EnsembleMode.summation:
                image_confidences += output_confidence
            elif ensemble_mode == EnsembleMode.product:
                image_confidences *= output_confidence
        predicted_confidences.append(image_confidences)
    return predicted_confidences


if __name__ == "__main__":
    predict()
