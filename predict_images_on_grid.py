import datetime
import os
import textwrap

import matplotlib

matplotlib.use("TKAgg", warn=False, force=True)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage
from termcolor import colored, cprint
import tensorflow as tf

from utils import max_rgb_filter

from check_accuracy import preprocess_image, calculate_accuracy, format_accuracy


experiment_pool = {
    "eye_v4-model_v12-rgb": [
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn", 4991),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn", 3974),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn", 4853),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn", 4345),
    ],
    "eye_v4-model_v12-hsv": [
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 2702),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 4385),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 3730),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 4865),
    ],
    "eye_v4-model_v15-hsv": [
        ("eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 3526),
        ("eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 4984),
        ("eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 4287),
        ("eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 4873),
    ],
    "eye_v4-model_v13-rgb": [
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn", 4925),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn", 4829),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn", 3061),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn", 4808),
    ],
    "eye_v4-model_v13-hsv": [
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 2321),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 2076),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 4462),
        ("eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 2940),
    ],
    "eye_v5-model_v12-rgb": [
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn", 4138),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn", 4468),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn", 3396),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn", 3445),
    ],
    "eye_v5-model_v12-hsv": [
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 3722),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 4884),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 4971),
        ("eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 4283),
    ],
    "eye_v5-model_v15-hsv": [
        ("eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 3874),
        ("eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 3437),
        ("eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 4470),
        ("eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 4996),
    ],
    "eye_v5-model_v13-rgb": [
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn", 4842),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn", 4316),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn", 4698),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn", 4248),
    ],
    "eye_v5-model_v13-hsv": [
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn", 4514),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn", 4818),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn", 4218),
        ("eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn", 4416),
    ],
}

from model_v15_cv import (
    create_model,
    get_test_data,
    get_train_data,
    test_generator,
    train_generator,
)


def predict():
    INPUT_SIZE = (256, 256, 2)
    COLOR_MODEL = "hsv"
    BATCH_NORMALIZATION = True
    experiment_epoch_pairs = experiment_pool["eye_v5-model_v15-hsv"]
    test_dir_names = ["validation", "test"]
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    PREDICT_VERBOSE = 1  # 0 = silent, 1

    for test_dir_name in test_dir_names:
        fig = plt.figure(figsize=(28, 14))
        fig.tight_layout()
        figure_title = experiment_epoch_pairs[0][0].replace("fold_1", "4_folds")
        fig.suptitle(figure_title)
        num_folds = len(experiment_epoch_pairs)
        outer = gridspec.GridSpec(num_folds, 1, wspace=0.2, hspace=0.2)

        for fold, (experiment_name, epoch) in enumerate(experiment_epoch_pairs):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[fold], wspace=0, hspace=0.3
            )

            weight = f"{epoch:08d}"
            cprint(f"The weight at epoch#", color="green", end="")
            cprint(f"{weight}", color="green", attrs=["bold"], end="")
            cprint(f" will be used to predict the images in ", color="green", end="")
            cprint(f"{test_dir_name}", color="green", attrs=["bold"], end="")
            cprint(f" directory", color="green")

            dataset_path = os.path.join("data", experiment_name)
            weights_dir = os.path.join(dataset_path, "weights")
            test_set_dir = os.path.join(dataset_path, test_dir_name)
            predicted_set_dirname = f"{test_dir_name}-predicted"
            predicted_set_dir = os.path.join(dataset_path, predicted_set_dirname)

            if not os.path.exists(predicted_set_dir):
                os.makedirs(predicted_set_dir)

            trained_weights_filename = f"{weight}.hdf5"
            trained_weights_file = os.path.join(weights_dir, trained_weights_filename)

            # load pretrained model
            model = create_model(
                pretrained_weights=trained_weights_file,
                input_size=INPUT_SIZE,
                num_classes=NUM_CLASSES,
                batch_normalization=BATCH_NORMALIZATION,
                is_summary=False,
            )

            # test the model
            test_data_dict = dict(
                test_path=test_set_dir, target_size=TARGET_SIZE, image_color=COLOR_MODEL
            )
            test_flow, test_files = get_test_data(**test_data_dict)
            test_gen = test_generator(test_flow, COLOR_MODEL)

            predict_steps = len(test_files)
            results = model.predict_generator(
                test_gen, steps=predict_steps, verbose=PREDICT_VERBOSE
            )

            segment_results = results[0]
            # iris_results = results[1]

            """
            Groundtruth
            """
            groundtruths = []
            for test_file in test_files:
                _, file = os.path.split(test_file)
                label_path = os.path.join(test_set_dir, "labels", file)
                groundtruth = skimage.io.imread(label_path)
                groundtruths.append(groundtruth)

            row = gridspec.GridSpecFromSubplotSpec(
                1, len(test_files), subplot_spec=inner[0], wspace=0.2, hspace=0
            )
            plot_groundtruths(
                fig,
                row,
                groundtruths,
                filenames=test_files,
                weights_name=weight,
                target_size=TARGET_SIZE,
                num_class=NUM_CLASSES,
            )

            """
            Predicted Outputs
            """
            fig.add_subplot()
            row = gridspec.GridSpecFromSubplotSpec(
                1, len(test_files), subplot_spec=inner[1], wspace=0.2, hspace=0
            )
            plot_predicted_outputs(
                fig,
                row,
                segment_results,
                groundtruths,
                weights_name=weight,
                target_size=TARGET_SIZE,
                num_class=NUM_CLASSES,
            )

        plt.draw()
        plt.pause(0.001)
        # input("Press [enter] to continue.")
        plt.show()

        prediction_dir = os.path.join("data", "prediction")
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        figure_file = os.path.join(
            prediction_dir, f"{figure_title}-{test_dir_name}.png"
        )
        plt.savefig(figure_file, bbox_inches="tight")

        break


def plot_groundtruths(
    fig, axs, groundtruths, filenames, weights_name, target_size=(256, 256), num_class=3
):
    for i, groundtruth in enumerate(groundtruths):
        filename = os.path.split(filenames[i])[1]
        output_shape = (target_size[0], target_size[1], num_class)
        groundtruth = skimage.transform.resize(groundtruth, output_shape)

        ax = plt.Subplot(fig, axs[i])
        ax.set(ylabel="groundtruth")
        ax.imshow(groundtruth)
        ax.axis("off")
        title = f"{filename}"
        title = "\n".join(textwrap.wrap(title, 15))
        ax.set_title(title, fontdict={"fontsize": 8, "fontweight": "bold"})
        fig.add_subplot(ax)
    return axs


def cce_accuracy(groundtruth, segment, target_size):
    accuracy = tf.keras.metrics.categorical_accuracy(groundtruth, segment)
    accuracy = accuracy.eval(session=tf.Session())
    accuracy = (accuracy == 1).sum() / (target_size[0] * target_size[1])
    return accuracy


def plot_predicted_outputs(
    fig, axs, results, groundtruths, weights_name, target_size=(256, 256), num_class=3
):
    accuracies = []
    for i, (segment, groundtruth) in enumerate(zip(results, groundtruths)):
        output_shape = (target_size[0], target_size[1], num_class)
        groundtruth = preprocess_image(groundtruth, output_shape)
        segment = preprocess_image(segment, output_shape)
        accuracy = calculate_accuracy(groundtruth, segment, output_shape)
        # accuracy = cce_accuracy(groundtruth, segment, output_shape)
        accuracies.append(accuracy)
        accuracy = format_accuracy(accuracy)

        ax = plt.Subplot(fig, axs[i])
        title = f"{accuracy}%"
        ax.set_title(title, fontdict={"fontsize": 8})
        ax.set(ylabel="predicted")
        ax.imshow(segment)
        ax.axis("off")
        fig.add_subplot(ax)
    print("average accuracy =", np.mean(accuracies))
    return axs


if __name__ == "__main__":
    predict()
