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
from utils import max_rgb_filter, get_weight_filename
from ensemble_aggregation_predict import ensemble_combine
from experiment_data import get_experiment_pool

matplotlib.use("TKAgg", warn=False, force=True)

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

    OUTPUT_DIRNAME = "S31_S32_S33_S34_S35_S36"
    TARGET_EXPERIMENT_POOL = [
        # "eye_v5-model_v12-rgb",
        # "eye_v5-model_v12-hsv",
        # "eye_v5-model_v15-hsv",
        # "eye_v5-model_v13-rgb",
        # "eye_v5-model_v13-hsv",
        # "eye_v5-model_v23-rgb",
        "eye_v5-model_v35-rgb",
        "eye_v5-model_v35-hsv",
        "eye_v5-model_v42",
        "eye_v5-model_v36-rgb",
        "eye_v5-model_v36-hsv",
        "eye_v5-model_v37",
        # "eye_v5-model_v39-rgb",
        # "eye_v5-model_v40-rgb",
        # "eye_v5-model_v41",
        # "eye_v5-unet_v2-rgb",
        # "eye_v5-unet_v3-rgb",
        # "eye_v5-unet_v4-rgb",
        # "eye_v5-unet-rgb",
        # "eye_v5-segnet-rgb",
        # "eye_v5-segnet_v3-rgb",
        # "sum_confidences_s1s2s3s4s5",
        # "eye_v5-model_v38",
        # "sum_confidences_s31u4sg1",
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
                print(">", NAME)
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
                    trained_weights_filename = get_weight_filename(
                        experiment, NAME, epoch
                    )
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
                        NAME,
                    )
                    if "model_v38" in NAME or "unet" in NAME or "segnet" in NAME:
                        segment_results = results
                    else:
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

                        trained_weights_filename = get_weight_filename(
                            each_experiment, model_name, epoch
                        )
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
                            model_name,
                        )
                        num_files = len(test_files)
                        if (
                            "model_v38" in model_name
                            or "unet" in model_name
                            or "segnet" in model_name
                        ):
                            segment_results = results
                        else:
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
                # concatenated_output = groundtruth
                first_time = True
                for model_output_name in predicted_outputs:
                    output_segment = predicted_outputs[model_output_name][g_index]
                    output_segment = convert_confidence_to_image(output_segment)
                    if first_time:
                        concatenated_output = output_segment
                    else:
                        concatenated_output = np.concatenate(
                            (concatenated_output, output_segment), axis=1
                        )
                    first_time = False

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
    model_name,
):
    model = create_model(
        pretrained_weights=trained_weights_file,
        input_size=input_size,
        num_classes=NUM_CLASSES,
        batch_normalization=BATCH_NORMALIZATION,
        is_summary=False,
    )

    test_data_dict = dict(test_path=set_path, target_size=TARGET_SIZE)
    if not ("v28" in model_name or "v37" in model_name or "v41" in model_name):
        test_data_dict["image_color"] = color_model
    test_flow, test_files = get_test_data(**test_data_dict)
    if not ("v28" in model_name or "v37" in model_name or "v41" in model_name):
        test_gen = test_generator(test_flow, color_model)
    else:
        test_gen = test_generator(test_flow)

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
    experiment_pool = get_experiment_pool()
    predict()
