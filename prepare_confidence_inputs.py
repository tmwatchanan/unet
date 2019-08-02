import csv
import datetime
import importlib
import os
import textwrap
import warnings
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
from utils import max_rgb_filter, get_weight_filename
from experiment_data import get_experiment_pool


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

    # experiment_names = ["s1", "s2", "s3", "s4", "s5"]
    experiment_names = ["s31", "u4", "sg1"]

    data_path = "data"
    dataset_path = os.path.join("datasets", "eye_v5-pass_1")

    for fold in range(4, 4 + 1):

        test_dir_names = ["test", "train", "validation"]
        for test_dir_name in test_dir_names:

            for NAME in experiment_pool:
                experiment = experiment_pool[NAME]
                if experiment.get("abbreviation") not in experiment_names:
                    continue

                import_model_functions(experiment["file"])
                input_size = experiment["input_size"]
                color_model = experiment["color_model"]
                experiment_epoch_pairs = experiment["experiments"][fold - 1]
                model_name = experiment_epoch_pairs[0]
                epoch = experiment_epoch_pairs[1]

                fold_dirname = f"fold_{fold}"
                fold_path = os.path.join(dataset_path, fold_dirname)
                set_path = os.path.join(fold_path, test_dir_name)

                """
                load pretrained model
                """
                model_path = os.path.join(data_path, model_name)
                weights_dir = os.path.join(model_path, "weights")
                trained_weights_filename = get_weight_filename(experiment, NAME, epoch)
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

                confidence_output_dir = os.path.join(set_path, "confidences")
                predicted_set_dir = os.path.join(
                    confidence_output_dir, experiment["abbreviation"]
                )
                if not os.path.exists(predicted_set_dir):
                    os.makedirs(predicted_set_dir)
                if "unet" in NAME or "segnet" in NAME:
                    multiple_outputs = False
                else:
                    multiple_outputs = True
                save_result(
                    predicted_set_dir,
                    results,
                    file_names=test_files,
                    weights_name=epoch,
                    target_size=TARGET_SIZE,
                    num_class=NUM_CLASSES,
                    multiple_outputs=multiple_outputs,
                )

    cprint(f"> predict confidence outputs succesfully")


def save_result(
    save_path,
    npyfile,
    file_names,
    weights_name,
    target_size=(256, 256),
    num_class=3,
    multiple_outputs=True,
):
    ol = 0
    if multiple_outputs:
        layer_output = npyfile[ol]
    else:
        layer_output = npyfile
    for i, item in enumerate(layer_output):
        file_name = os.path.split(file_names[i])[1]
        output_shape = (target_size[0], target_size[1], num_class)
        item = np.reshape(item, output_shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(
                os.path.join(
                    save_path, f"{file_name}-{weights_name}-{ol+1}-merged.png"
                ),
                item,
            )


if __name__ == "__main__":
    experiment_pool = get_experiment_pool()
    predict()
