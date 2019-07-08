import datetime
import importlib
import os
import textwrap

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
from termcolor import colored, cprint
from keras.utils import to_categorical

from check_accuracy import (
    calculate_accuracy,
    format_accuracy,
    preprocess_image,
    read_groundtruth,
)
from utils import max_rgb_filter
from ensemble_aggregation_predict import EnsembleMode, ensemble_combine

matplotlib.use("TKAgg", warn=False, force=True)


experiment_pool = {
    # "eye_v4-model_v12-rgb": {
    #     "input_size": (256, 256, 3),
    #     "color_model": "rgb",
    #     "file": "model_v12_cv",
    #     "dataset": "eye_v4",
    #     "experiments": [
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
    #             4991,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
    #             3974,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
    #             4853,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
    #             4345,
    #         ),
    #     ],
    # },
    # "eye_v4-model_v12-hsv": {
    #     "input_size": (256, 256, 3),
    #     "color_model": "hsv",
    #     "file": "model_v12_cv",
    #     "dataset": "eye_v4",
    #     "experiments": [
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
    #             2702,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
    #             4385,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
    #             3730,
    #         ),
    #         (
    #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
    #             4865,
    #         ),
    #     ],
    # },
    # "eye_v4-model_v15-hsv": {
    #     "input_size": (256, 256, 2),
    #     "color_model": "hsv",
    #     "file": "model_v15_cv",
    #     "dataset": "eye_v4",
    #     "experiments": [
    #         (
    #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
    #             3526,
    #         ),
    #         (
    #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
    #             4984,
    #         ),
    #         (
    #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
    #             4287,
    #         ),
    #         (
    #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
    #             4873,
    #         ),
    #     ],
    # },
    # "eye_v4-model_v13-rgb": {
    #     "input_size": (256, 256, 5),
    #     "color_model": "rgb",
    #     "file": "model_v13_cv",
    #     "dataset": "eye_v4",
    #     "experiments": [
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
    #             4925,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
    #             4829,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
    #             3061,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
    #             4808,
    #         ),
    #     ],
    # },
    # "eye_v4-model_v13-hsv": {
    #     "input_size": (256, 256, 5),
    #     "color_model": "hsv",
    #     "file": "model_v13_cv",
    #     "dataset": "eye_v4",
    #     "experiments": [
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
    #             2321,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
    #             2076,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
    #             4462,
    #         ),
    #         (
    #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
    #             2940,
    #         ),
    #     ],
    # },
    "eye_v5-model_v12-rgb": {
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
    # "eye_v5-model_v23-rgb": {
    #     "input_size": (256, 256, 9),
    #     "color_model": "rgb",
    #     "file": "model_v23_cv",
    #     "dataset": "eye_v5-s1",
    #     "experiments": [
    #         (
    #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
    #             2456,
    #         ),
    #         (
    #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
    #             4480,
    #         ),
    #         (
    #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
    #             4012,
    #         ),
    #         (
    #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
    #             3308,
    #         ),
    #     ],
    # },
    "sum_s1s2s3s4s5": {
        "dataset": "eye_v5",
        "ensemble_mode": EnsembleMode.summation,
        "models": [
            "eye_v5-model_v12-rgb",
            "eye_v5-model_v12-hsv",
            "eye_v5-model_v15-hsv",
            "eye_v5-model_v13-rgb",
            "eye_v5-model_v13-hsv",
        ],
    },
    "eye_v5-model_v23-rgb": {
        "input_size": (256, 256, 9),
        "color_model": "rgb",
        "file": "model_v23_cv",
        "dataset": "eye_v5-s1",
        "experiments": [
            (
                "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                3742,
            ),
            (
                "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                4903,
            ),
            (
                "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                3637,
            ),
            (
                "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                4617,
            ),
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

    OUTPUT_DIRNAME = "G_S1_S2_S3_S4_S5"
    # OUTPUT_DIRNAME = "G_S1_S2_S3_E5_P2"
    TARGET_EXPERIMENT_POOL = [
        "eye_v5-model_v12-rgb",
        "eye_v5-model_v12-hsv",
        "eye_v5-model_v15-hsv",
        "eye_v5-model_v13-rgb",
        "eye_v5-model_v13-hsv",
        # "sum_s1s2s3s4s5",
        # "eye_v5-model_v23-rgb",
    ]

    data_path = "data"
    dataset_path = "datasets"
    outputs_comparison_path = os.path.join(data_path, OUTPUT_DIRNAME)
    if not os.path.exists(outputs_comparison_path):
        os.makedirs(outputs_comparison_path)

    for fold in range(1, 4 + 1):

        test_dir_names = ["test", "train", "validation"]
        for test_dir_name in test_dir_names:

            predicted_outputs = {}
            for i, NAME in enumerate(TARGET_EXPERIMENT_POOL):
                experiment = experiment_pool[NAME]

                if "models" not in experiment:
                    import_model_functions(experiment["file"])
                    input_size = experiment["input_size"]
                    color_model = experiment["color_model"]
                    input_dataset_name = experiment["dataset"]
                    experiment_epoch_pairs = experiment["experiments"][fold - 1]
                    model_name = experiment_epoch_pairs[0]
                    epoch = experiment_epoch_pairs[1]

                    fold_dirname = f"fold_{fold}"
                    fold_path = os.path.join(
                        dataset_path, input_dataset_name, fold_dirname
                    )
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

                    results, test_files = predict_images(
                        trained_weights_file,
                        input_size,
                        NUM_CLASSES,
                        BATCH_NORMALIZATION,
                        set_path,
                        TARGET_SIZE,
                        color_model,
                        PREDICT_VERBOSE,
                    )
                    segment_results = results[0]

                    predicted_outputs[i] = segment_results
                else:
                    outputs_from_multiple_models = {}
                    model_names = experiment["models"]
                    ensemble_mode = experiment["ensemble_mode"]

                    for model_name in model_names:
                        each_experiment = experiment_pool[model_name]

                        import_model_functions(each_experiment["file"])
                        input_size = each_experiment["input_size"]
                        color_model = each_experiment["color_model"]
                        input_dataset_name = each_experiment["dataset"]
                        experiment_epoch_pairs = each_experiment["experiments"][
                            fold - 1
                        ]
                        model_name = experiment_epoch_pairs[0]
                        epoch = experiment_epoch_pairs[1]

                        fold_dirname = f"fold_{fold}"
                        fold_path = os.path.join(
                            dataset_path, input_dataset_name, fold_dirname
                        )
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

                        results, test_files = predict_images(
                            trained_weights_file,
                            input_size,
                            NUM_CLASSES,
                            BATCH_NORMALIZATION,
                            set_path,
                            TARGET_SIZE,
                            color_model,
                            PREDICT_VERBOSE,
                        )
                        num_files = len(test_files)
                        segment_results = results[0]

                        outputs_from_multiple_models[model_name] = segment_results

                    predicted_outputs[i] = ensemble_combine(
                        num_files,
                        ensemble_mode,
                        TARGET_SIZE,
                        outputs_from_multiple_models,
                    )

            """
            Groundtruth
            """
            groundtruths = []
            for test_file in test_files:
                _, file = os.path.split(test_file)
                groundtruth_path = os.path.join(set_labels_path, file)
                groundtruth = read_groundtruth(groundtruth_path, TARGET_SIZE)
                groundtruths.append((file, groundtruth))

            for g_index, (filename, groundtruth) in enumerate(groundtruths):
                concatenated_output = groundtruth
                for model_output_name in predicted_outputs:
                    output_segment = predicted_outputs[model_output_name][g_index]
                    output_segment = convert_confidence_to_image(output_segment)
                    concatenated_output = np.concatenate(
                        (concatenated_output, output_segment), axis=1
                    )

                concatenated_output = np.hsplit(concatenated_output, 2)
                concatenated_output = np.concatenate(concatenated_output, axis=0)

                output_path = os.path.join(
                    outputs_comparison_path, fold_dirname, test_dir_name
                )
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_filename = f"{filename}-{OUTPUT_DIRNAME}.png"
                output_file = os.path.join(output_path, output_filename)
                skimage.io.imsave(output_file, concatenated_output)


def predict_images(
    trained_weights_file,
    input_size,
    NUM_CLASSES,
    BATCH_NORMALIZATION,
    set_path,
    TARGET_SIZE,
    color_model,
    PREDICT_VERBOSE,
):
    model = create_model(
        pretrained_weights=trained_weights_file,
        input_size=input_size,
        num_classes=NUM_CLASSES,
        batch_normalization=BATCH_NORMALIZATION,
        is_summary=False,
    )

    test_data_dict = dict(
        test_path=set_path, target_size=TARGET_SIZE, image_color=color_model
    )
    test_flow, test_files = get_test_data(**test_data_dict)
    test_gen = test_generator(test_flow, color_model)

    predict_steps = len(test_files)
    results = model.predict_generator(
        test_gen, steps=predict_steps, verbose=PREDICT_VERBOSE
    )
    return results, test_files


def convert_confidence_to_image(array):
    argmax_class = np.argmax(array, axis=-1)
    image = to_categorical(argmax_class)
    return image


if __name__ == "__main__":
    predict()
