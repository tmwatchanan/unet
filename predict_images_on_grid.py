from model_v12_cv import create_model, get_test_data, test_generator
import os
import datetime
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from termcolor import colored, cprint
from utils import max_rgb_filter


def predict():
    INPUT_SIZE = (256, 256, 3)
    COLOR_MODEL = "rgb"
    BATCH_NORMALIZATION = True
    experiment_epoch_pairs = [
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn", 4991),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn", 3974),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn", 4853),
        ("eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn", 4345),
    ]
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    SAVE_EACH_LAYER = False
    PREDICT_VERBOSE = 1  # 0 = silent, 1

    fig, axs = plt.subplots(len(experiment_epoch_pairs), 8 + 1 + 8)
    fig.tight_layout()
    # for ax in axs.flat:
    #     ax.set(xlabel="x-label", ylabel="y-label")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    for fold, (experiment_name, epoch) in enumerate(experiment_epoch_pairs):
        test_dir_names = ["validation", "test"]

        for row, test_dir_name in enumerate(test_dir_names):
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
            prediction_setting_file = os.path.join(
                predicted_set_dir, "prediction_settings.txt"
            )

            if not os.path.exists(predicted_set_dir):
                os.makedirs(predicted_set_dir)

            def save_prediction_settings_file():
                with open(prediction_setting_file, "w") as f:
                    current_datetime = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    f.write(f"{current_datetime}\n")
                    f.write(f"experiment_name={experiment_name}\n")
                    f.write(f"test_dir_name={test_dir_name}\n")
                    f.write(f"weight={weight}\n")
                    f.write(f"BATCH_SIZE=1\n")
                    f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
                    f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
                    f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
                    f.write(f"COLOR_MODEL={COLOR_MODEL}\n")
                    f.write(f"SAVE_EACH_LAYER={SAVE_EACH_LAYER}\n")
                    f.write(f"=======================\n")

            # save_prediction_settings_file()

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

            axs = get_images_from_batch(
                axs,
                fold,
                row,
                predicted_set_dir,
                segment_results,
                filenames=test_files,
                weights_name=weight,
                target_size=TARGET_SIZE,
                num_class=NUM_CLASSES,
            )

    plt.title("Eye predicted results")
    plt.draw()
    plt.pause(0.001)
    # input("Press [enter] to continue.")
    plt.show()


def get_images_from_batch(
    axs,
    row,
    set_number,
    save_path,
    results,
    filenames,
    weights_name,
    target_size=(256, 256),
    num_class=3,
):
    for i, segment in enumerate(results):
        filename = os.path.split(filenames[i])[1]
        output_shape = (target_size[0], target_size[1], num_class)
        segment = np.reshape(segment, output_shape)
        visualized_img = max_rgb_filter(segment)
        visualized_img[visualized_img > 0] = 1

        subplot_column_index = i + (set_number * 8) + (1 if set_number > 0 else 0)
        axs[row, subplot_column_index].imshow(visualized_img)
        axs[row, subplot_column_index].set_title(
            f"{filename}", fontdict={"fontsize": 9}
        )
        axs[row, subplot_column_index].axis("off")

        # ax = plt.subplot(gs1[i])
        # ax.imshow(visualized_img)
        # ax.axis("off")
        # ax.text(0.5, -0.1, f"{filename}", size=12, ha="center", transform=ax.transAxes)
    return axs


if __name__ == "__main__":
    predict()
