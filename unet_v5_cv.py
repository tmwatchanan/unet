import copy
import csv
import datetime
import io
import os
import shutil
import time
import warnings
from collections import Iterable, OrderedDict
from itertools import tee
import sys

import click
import cv2
import matplotlib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import six
import skimage
from sklearn import metrics
from scipy import ndimage
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback, CSVLogger, LambdaCallback, ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    Permute,
    Reshape,
    ThresholdedReLU,
    UpSampling2D,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard
from termcolor import colored, cprint

from utils import add_sobel_filters, max_rgb_filter
from datasets import Dataset

matplotlib.use("Agg")


@click.group()
def cli():
    #  click.echo("This is main function echoed by Click")
    pass


def get_color_convertion_function(color_model):
    if color_model == "hsv":
        color_convertion_function = skimage.color.rgb2hsv
    elif color_model == "ycbcr":
        color_convertion_function = skimage.color.rgb2ycbcr
    else:
        color_convertion_function = None
    return color_convertion_function


class PredictOutput(Callback):
    def __init__(
        self,
        test_set_dir,
        color_model,
        weights_dir,
        target_size,
        num_classes,
        predicted_set_dir,
        period,
        save_each_layer,
        fit_verbose,
    ):
        #  self.out_log = []
        self.test_set_dir = test_set_dir
        self.color_model = color_model
        self.weights_dir = weights_dir
        self.target_size = target_size
        self.num_classes = num_classes
        self.predicted_set_dir = predicted_set_dir
        self.period = period
        self.save_each_layer = save_each_layer
        self.fit_verbose = fit_verbose

    def on_epoch_end(self, epoch, logs=None):
        predict_epoch = epoch + 1
        if predict_epoch % self.period == 0:
            test_data_dict = dict(
                test_path=self.test_set_dir,
                target_size=self.target_size,
                image_color=self.color_model,
            )
            test_flow, test_files = get_test_data(**test_data_dict)
            test_gen = test_generator(test_flow, self.color_model)
            num_test_files = len(test_files)

            results = self.model.predict_generator(
                test_gen, steps=num_test_files, verbose=self.fit_verbose
            )
            last_weights_file = f"{predict_epoch:08d}"
            save_result(
                self.predicted_set_dir,
                results,
                file_names=test_files,
                weights_name=last_weights_file,
                target_size=self.target_size,
                num_class=self.num_classes,
                save_each_layer=self.save_each_layer,
            )
        #  self.out_log.append()


class TimeHistory(Callback):
    def __init__(self, filename, separator=",", append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = ["time_per_epoch"]
        self.append_header = True
        if six.PY2:
            self.file_flags = "b"
            self._open_args = {}
        else:
            self.file_flags = ""
            self._open_args = {"newline": "\n"}
        super(TimeHistory, self).__init__()

    def on_train_begin(self, logs={}):
        self.times = []
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, "r" + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = io.open(
            self.filename, mode + self.file_flags, **self._open_args
        )

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else "NA") for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({"epoch": epoch})
        row_dict.update({"time_per_epoch": self.times[-1]})
        self.writer.writerow(row_dict)
        self.csv_file.flush()


@cli.command()
@click.pass_context
def train(ctx):
    DATASET_NAME = "eye_v5"
    dataset = Dataset(DATASET_NAME)

    for fold in range(1, 4 + 1):
        cprint("> ", end="")
        cprint("`train`", color="green", end="")
        cprint(" function")
        COLOR_MODEL = "hsv" # rgb, hsv, ycbcr, gray
        MODEL_NAME = "unet_v5_multiclass"
        MODEL_INFO = f"softmax-cce-lw_1_0-{COLOR_MODEL}-fold_{fold}"
        BATCH_NORMALIZATION = True
        LEARNING_RATE = "1e_2"
        EXPERIMENT_NAME = (
            f"{DATASET_NAME}-{MODEL_NAME}-{MODEL_INFO}-lr_{LEARNING_RATE}"
            + ("-bn" if BATCH_NORMALIZATION else "")
        )
        TEST_DIR_NAME = "test"
        EPOCH_START = 0
        EPOCH_END = 5000
        MODEL_PERIOD = 1
        INPUT_SIZE = (256, 256, 2)
        TARGET_SIZE = (256, 256)
        NUM_CLASSES = 3
        SAVE_EACH_LAYER = False
        FIT_VERBOSE = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch

        dataset_path = os.path.join("data", EXPERIMENT_NAME)
        weights_dir = os.path.join(dataset_path, "weights")
        training_set_dir = os.path.join(dataset_path, "train")
        training_images_set_dir = os.path.join(training_set_dir, "images")
        training_labels_set_dir = os.path.join(training_set_dir, "labels")
        validation_set_dir = os.path.join(dataset_path, "validation")
        validation_images_set_dir = os.path.join(validation_set_dir, "images")
        validation_labels_set_dir = os.path.join(validation_set_dir, "labels")
        test_set_dir = os.path.join(dataset_path, TEST_DIR_NAME)
        predicted_set_dir = os.path.join(
            dataset_path, f"{TEST_DIR_NAME}-predicted-{COLOR_MODEL}"
        )
        training_log_file = os.path.join(dataset_path, "training.csv")
        training_time_log_file = os.path.join(dataset_path, "training_time.csv")
        loss_acc_file = os.path.join(dataset_path, "loss_acc.csv")
        experiments_setting_file = os.path.join(dataset_path, "experiment_settings.txt")
        model_file = os.path.join(dataset_path, "model.png")
        tensorboard_log_dir = os.path.join(dataset_path, "logs")

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        if not os.path.exists(predicted_set_dir):
            os.makedirs(predicted_set_dir)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)

        learning_rate = float(LEARNING_RATE.replace("_", "-"))

        def save_experiment_settings_file():
            with open(experiments_setting_file, "a") as f:
                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{current_datetime}\n")
                f.write(f"MODEL_NAME={MODEL_NAME}\n")
                f.write(f"MODEL_INFO={MODEL_INFO}\n")
                f.write(f"BATCH_NORMALIZATION={BATCH_NORMALIZATION}\n")
                f.write(f"EPOCH_START={EPOCH_START}\n")
                f.write(f"EPOCH_END={EPOCH_END}\n")
                f.write(f"MODEL_PERIOD={MODEL_PERIOD}\n")
                f.write(f"DATASET_NAME={DATASET_NAME}\n")
                f.write(f"TRAIN_BATCH_SIZE={dataset.train_batch_size}\n")
                f.write(f"VALIDATION_BATCH_SIZE={dataset.validation_batch_size}\n")
                f.write(f"TRAIN_STEPS_PER_EPOCH={dataset.train_steps_per_epoch}\n")
                f.write(f"VALIDATION_STEPS_PER_EPOCH={dataset.validation_steps_per_epoch}\n")
                f.write(f"LEARNING_RATE={LEARNING_RATE}\n")
                f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
                f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
                f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
                f.write(f"COLOR_MODEL={COLOR_MODEL}\n")
                f.write(f"SAVE_EACH_LAYER={SAVE_EACH_LAYER}\n")
                f.write(f"=======================\n")

        save_experiment_settings_file()

        model_filename = "{}.hdf5"
        if EPOCH_START == 0:
            trained_weights_file = None
        else:
            trained_weights_name = f"{EPOCH_START:08d}"
            trained_weights_file = model_filename.format(trained_weights_name)
            trained_weights_file = os.path.join(weights_dir, trained_weights_file)

        num_training = 0
        for _, _, files in os.walk(training_images_set_dir):
            num_training += len(files)
        num_validation = 0
        for _, _, files in os.walk(validation_images_set_dir):
            num_validation += len(files)
        print(f"num_training={num_training}")
        print(f"num_validation={num_validation}")

        model = create_model(
            pretrained_weights=trained_weights_file,
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
            batch_normalization=BATCH_NORMALIZATION,
        )  # load pretrained model

        # save model architecture figure
        plot_model(model, show_shapes=True, to_file=model_file)

        data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        train_data_dict = dict(
            batch_size=dataset.train_batch_size,
            train_path=training_set_dir,
            aug_dict=data_gen_args,
            image_color=COLOR_MODEL,
            mask_color=COLOR_MODEL,
            save_to_dir=None,
            target_size=TARGET_SIZE,
        )
        train_flow = get_train_data(**train_data_dict)
        train_gen = train_generator(train_flow, COLOR_MODEL)

        validation_data_dict = dict(
            batch_size=dataset.validation_batch_size,
            train_path=validation_set_dir,
            image_color=COLOR_MODEL,
            mask_color=COLOR_MODEL,
            save_to_dir=None,
            target_size=TARGET_SIZE,
            shuffle=False,
        )
        validation_flow = get_train_data(**validation_data_dict)
        validation_gen = train_generator(validation_flow, COLOR_MODEL)

        # train the model
        #  new_weights_name = '{epoch:08d}'
        #  new_weights_file = model_filename.format(new_weights_name)
        new_weights_file = "unet_v5_best.hdf5"
        new_weights_file = os.path.join(weights_dir, new_weights_file)
        model_checkpoint = ModelCheckpoint(
            filepath=new_weights_file,
            monitor="val_acc",
            mode="auto",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            period=MODEL_PERIOD,
        )
        # predict_output = PredictOutput(
        #     test_set_dir,
        #     COLOR_MODEL,
        #     weights_dir,
        #     TARGET_SIZE,
        #     NUM_CLASSES,
        #     predicted_set_dir,
        #     period=MODEL_PERIOD,
        #     save_each_layer=SAVE_EACH_LAYER,
        #     fit_verbose=FIT_VERBOSE,
        # )
        csv_logger = CSVLogger(training_log_file, append=True)
        tensorboard = TensorBoard(
            log_dir=os.path.join(tensorboard_log_dir, str(time.time())),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        last_epoch = []
        save_output_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: (last_epoch.append(epoch))
        )
        time_history_callback = TimeHistory(training_time_log_file, append=True)
        callbacks = [
            model_checkpoint,
            csv_logger,
            tensorboard,
            save_output_callback,
            # predict_output,
            time_history_callback,
        ]
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=dataset.train_steps_per_epoch,
            epochs=EPOCH_END,
            initial_epoch=EPOCH_START,
            callbacks=callbacks,
            validation_data=validation_gen,
            validation_steps=dataset.validation_steps_per_epoch,
            workers=0,
            use_multiprocessing=True,
            verbose=FIT_VERBOSE,
        )
        #  print(history.history.keys())  # show dict of metrics in history

        # plot([EXPERIMENT_NAME])
        ctx.invoke(plot, experiment_name=EXPERIMENT_NAME)


def diff_iris_area(y_true, y_pred):
    area_true = K.cast(K.sum(y_true, axis=[1, 2]), "float32")
    area_pred = K.sum(y_pred, axis=[1, 2])
    normalized_diff = (area_true - area_pred) / area_true
    return K.mean(K.square(normalized_diff), axis=0)


def create_model(
    pretrained_weights=None,
    input_size=(),
    num_classes=2,
    learning_rate=1e-4,
    batch_normalization=False,
    is_summary=True,
):
    inputs = Input(input_size)
    conv1 = Conv2D(
        24, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = Conv2D(
        24, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        48, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = Conv2D(
        48, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        96, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = Conv2D(
        96, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)

    up7 = Conv2D(
        96, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(
        96, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = Conv2D(
        96, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)

    up8 = Conv2D(
        48, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(
        48, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = Conv2D(
        48, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)

    up9 = Conv2D(
        24, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(
        24, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = Conv2D(
        24, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if is_summary:
        model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def preprocess_image_input(img, color_model):
    """preprocess input using image
       e.g., add feature layers, such as canny and gradient

       Normalization process must be done here after using image to calculate other features
    
    Arguments:
        img {3D array} -- size = (x, y, channels)
        color_model {string} -- color model of image (e.g., rgb and hsv)
    
    Returns:
        3D array -- preprocessed image
    """
    # create a duplicate object of image
    processed_img = copy.deepcopy(img)
    # add sobel feature layers
    processed_img = add_sobel_filters(img, color_model, processed_img)

    # normalize the values of gradient x and y
    gradients = processed_img[:,:, 3:5]
    max_absolute_gradient_value = max(gradients.min(), gradients.max(), key=abs)
    processed_img[:, :, 3:5] /= max_absolute_gradient_value

    # extract only the last `l` layers as this model we want only gradient layers
    last_layers = 2
    processed_img = processed_img[:, :, -last_layers:]
    return processed_img


def preprocess_mask_input(mask):
    # mask shape = (BATCH_SIZE, x, y, channels)
    mask = mask / 255
    mask_iris = mask[:, :, :, 0]
    return mask, mask_iris


def preprocess_images_in_batch(img_batch, img_color_model):
    processed_input_list = []
    for img in img_batch:
        processed_input = preprocess_image_input(img, img_color_model)
        processed_input_list.append(processed_input)
    processed_input_array = np.array(processed_input_list)
    return processed_input_array


def get_train_data(
    batch_size,
    train_path,
    aug_dict=dict(),
    image_folder="images",
    mask_folder="labels",
    image_color="rgb",
    mask_color="rgb",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
    shuffle=True,
):
    image_aug_dict = copy.deepcopy(aug_dict)
    image_aug_dict["preprocessing_function"] = get_color_convertion_function(
        image_color
    )
    image_datagen = ImageDataGenerator(**image_aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_color_mode = "grayscale" if image_color == "gray" else "rgb"
    mask_color_mode = "grayscale" if mask_color == "gray" else "rgb"
    image_flow = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        shuffle=shuffle,
    )
    mask_flow = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        shuffle=shuffle,
    )
    return zip(image_flow, mask_flow)


def get_test_data(
    test_path, image_folder="images", target_size=(256, 256), image_color="rgb", seed=1
):
    color_convertion_function = get_color_convertion_function(image_color)
    test_datagen = ImageDataGenerator(preprocessing_function=color_convertion_function)
    image_color_mode = "grayscale" if image_color == "gray" else "rgb"
    image_flow = test_datagen.flow_from_directory(
        test_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=1,
        shuffle=False,
        seed=seed,
    )
    return image_flow, image_flow.filenames


def train_generator(image_mask_pair_flow, image_color_model):
    for (img_batch, mask_batch) in image_mask_pair_flow:
        processed_img_array = preprocess_images_in_batch(img_batch, image_color_model)
        mask, mask_iris = preprocess_mask_input(mask_batch)
        yield ([processed_img_array], [mask, mask_iris])


def test_generator(test_flow, image_color_model):
    for img_batch in test_flow:
        processed_img_array = preprocess_images_in_batch(img_batch, image_color_model)
        yield [processed_img_array]


def save_result(
    save_path,
    npyfile,
    file_names,
    weights_name,
    target_size=(256, 256),
    num_class=3,
    save_each_layer=False,
    save_iris=False,
):
    for ol in range(len(npyfile)):
        layer_output = npyfile[ol]
        for i, item in enumerate(layer_output):
            file_name = os.path.split(file_names[i])[1]
            #  file_name=file_names[i]
            if ol == 0:
                output_shape = (target_size[0], target_size[1], num_class)
                item = np.reshape(item, output_shape)
                visualized_img = max_rgb_filter(item)
                visualized_img[visualized_img > 0] = 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skimage.io.imsave(
                        os.path.join(
                            save_path, f"{file_name}-{weights_name}-{ol+1}-merged.png"
                        ),
                        visualized_img,
                    )
                    if save_each_layer:
                        skimage.io.imsave(
                            os.path.join(
                                save_path, f"{file_name}-{weights_name}-{ol+1}-0.png"
                            ),
                            item[:, :, 0],
                        )
                        skimage.io.imsave(
                            os.path.join(
                                save_path, f"{file_name}-{weights_name}-{ol+1}-1.png"
                            ),
                            item[:, :, 1],
                        )
                        skimage.io.imsave(
                            os.path.join(
                                save_path, f"{file_name}-{weights_name}-{ol+1}-2.png"
                            ),
                            item[:, :, 2],
                        )
            elif ol == 1 and save_iris:
                item = np.reshape(item, target_size)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skimage.io.imsave(
                        os.path.join(
                            save_path, f"{file_name}-{weights_name}-{ol+1}-iris.png"
                        ),
                        item[:, :],
                    )


def plot_graph(
    figure_num, epoch_list, x, y, x_label, y_label, title, legend, save_name
):
    fig_acc = plt.figure(figure_num)
    plt.plot(epoch_list, x, "b")
    plt.plot(epoch_list, y, "g")
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color="k", linestyle="-", linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc="lower right")
    fig_acc.savefig(save_name, bbox_inches="tight")


@cli.command()
@click.argument("experiment_name", required=False)
def plot(experiment_name):
    # define paths
    dataset_path = os.path.join("data", experiment_name)
    training_log_file = os.path.join(dataset_path, "training.csv")

    graphs_dir = os.path.join(dataset_path, "graphs")
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    output1_acc_file = os.path.join(graphs_dir, "output1_acc.png")
    # output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    # output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")

    history_data = pd.read_csv(training_log_file)
    print(history_data.columns)

    # plot graphs
    plot_graph(
        1,
        history_data["epoch"],
        history_data["output1_acc"],
        history_data["val_output1_acc"],
        "Accuracy",
        "Epoch",
        f"{experiment_name} - Output 1 Model Accuracy",
        ["Train Accuracy", "Validation Accuracy"],
        output1_acc_file,
    )
    plot_graph(
        2,
        history_data["epoch"],
        history_data["output1_loss"],
        history_data["val_output1_loss"],
        "Loss",
        "Epoch",
        f"{experiment_name} - Output 1 Model Loss (cce)",
        ["Train Loss", "Validation Loss"],
        output1_loss_file,
    )
    # plot_graph(
    #     3,
    #     history_data["epoch"],
    #     history_data["output_iris_acc"],
    #     history_data["val_output_iris_acc"],
    #     "Accuracy",
    #     "Epoch",
    #     f"{experiment_name} - Output Iris Model Accuracy",
    #     ["Train Accuracy", "Validation Accuracy"],
    #     output_iris_acc_file,
    # )
    # plot_graph(
    #     4,
    #     history_data["epoch"],
    #     history_data["output_iris_loss"],
    #     history_data["val_output_iris_loss"],
    #     "Loss",
    #     "Epoch",
    #     f"{experiment_name} - Output Iris Model Loss (diff_iris_area)",
    #     ["Train Loss", "Validation Loss"],
    #     output_iris_loss_file,
    # )

    # immediately show plotted graphs
    #  plt.show()
    return


@cli.command()
@click.argument("experiment_name")
@click.argument("weight")
@click.argument("color_model")
@click.argument("batch_normalization")
@click.argument("test_dir_name")
def predict(experiment_name, weight, color_model, batch_normalization, test_dir_name):
    cprint(f"> Running `predict` command on ", color="green", end="")
    cprint(f"{experiment_name}", color="green", attrs=["bold"], end=", ")
    cprint(f"{color_model}", color="green", attrs=["bold"], end=", ")
    cprint(f"batch_normalization" if batch_normalization else "", color="grey", attrs=["bold"], end=", ")
    cprint(f" experiment", color="green")
    INPUT_SIZE = (256, 256, 2)
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    SAVE_EACH_LAYER = False
    PREDICT_VERBOSE = 1  # 0 = silent, 1

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
    prediction_setting_file = os.path.join(predicted_set_dir, "prediction_settings.txt")

    if not os.path.exists(predicted_set_dir):
        os.makedirs(predicted_set_dir)

    def save_prediction_settings_file():
        with open(prediction_setting_file, "w") as f:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_datetime}\n")
            f.write(f"experiment_name={experiment_name}\n")
            f.write(f"test_dir_name={test_dir_name}\n")
            f.write(f"weight={weight}\n")
            f.write(f"BATCH_SIZE=1\n")
            f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
            f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
            f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
            f.write(f"COLOR_MODEL={color_model}\n")
            f.write(f"SAVE_EACH_LAYER={SAVE_EACH_LAYER}\n")
            f.write(f"=======================\n")

    save_prediction_settings_file()

    trained_weights_filename = f"unet_v5_best.hdf5"
    trained_weights_file = os.path.join(weights_dir, trained_weights_filename)

    # load pretrained model
    model = create_model(
        pretrained_weights=trained_weights_file,
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        batch_normalization=batch_normalization,
        is_summary=False
    )

    # test the model
    test_data_dict = dict(
        test_path=test_set_dir,
        target_size=TARGET_SIZE,
        image_color=color_model,
    )
    test_flow, test_files = get_test_data(**test_data_dict)
    test_gen = test_generator(test_flow, color_model)

    predict_steps = len(test_files)
    results = model.predict_generator(
        test_gen, steps=predict_steps, verbose=PREDICT_VERBOSE
    )
    save_result(
        predicted_set_dir,
        results,
        file_names=test_files,
        weights_name=weight,
        target_size=TARGET_SIZE,
        num_class=NUM_CLASSES,
        save_each_layer=SAVE_EACH_LAYER,
    )
    cprint(
        f"> `test` command was successfully run, the predicted result will be in ",
        color="green",
        end="",
    )
    cprint(f"{predicted_set_dirname}", color="green", attrs=["bold"])

@cli.command()
@click.pass_context
def evaluate(ctx):
    DATASET_NAME = "eye_v5"
    dataset = Dataset(DATASET_NAME)
    MODEL_NAME = "unet_v5_multiclass"
    MODEL_INFO = "softmax-cce-lw_1_0"
    COLOR_MODEL = "hsv"  # rgb, hsv, ycbcr, gray
    BATCH_NORMALIZATION = True
    LEARNING_RATE = "1e_2"
    batch_size = dataset.validation_batch_size
    evaluate_steps = dataset.validation_steps_per_epoch
    INPUT_SIZE = (256, 256, 2)
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    fold_list = range(1, 4 + 1)
    batch_normalization_info = "-bn" if BATCH_NORMALIZATION else ""
    experiment_name_template = (
        DATASET_NAME
        + "-"
        + MODEL_NAME
        + "-"
        + MODEL_INFO
        + "-"
        + COLOR_MODEL
        + "-fold_{0}"
        + "-lr_"
        + LEARNING_RATE
        + batch_normalization_info
    )
    training_validation_evaluation = evaluate_training_and_validation(
        experiment_name_template, fold_list
    )

    data_path = "data"
    evaluation_dir = os.path.join(data_path, "evaluation")
    evaluation_csv_filename = (
        experiment_name_template.format(f"{fold_list[0]}-{fold_list[-1]}") + ".csv"
    )
    evaluation_csv_file = os.path.join(evaluation_dir, evaluation_csv_filename)

    output_summary_evaluation = []
    classes = ["iris", "sclera", "bg"]
    folds_label_image_pairs = {}
    for p_class in classes:
        folds_label_image_pairs[p_class] = {"label": np.empty(0), "image": np.empty(0)}
    model_epoch_list = [
        fold_evaluation["epoch"] for fold_evaluation in training_validation_evaluation
    ]
    for (fold, epoch) in zip(fold_list, model_epoch_list):
        experiment_name = experiment_name_template.format(fold)
        experiment_dir = os.path.join(data_path, experiment_name)
        weights_dir = os.path.join(experiment_dir, "weights")
        test_set_dir = os.path.join(experiment_dir, "test")
        test_set_images_dir = os.path.join(test_set_dir, "images")

        num_test = 0
        for _, _, files in os.walk(test_set_images_dir):
            num_test += len(files)
        print(f"num_test={num_test}")

        trained_weights_file = f"unet_v5_best.hdf5"
        trained_weights_file = os.path.join(weights_dir, trained_weights_file)

        learning_rate = float(LEARNING_RATE.replace("_", "-"))

        model = create_model(
            pretrained_weights=trained_weights_file,
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
            batch_normalization=BATCH_NORMALIZATION,
            is_summary=False,
        )  # load pretrained model

        test_data_dict = dict(
            batch_size=batch_size,
            train_path=test_set_dir,
            image_color=COLOR_MODEL,
            mask_color=COLOR_MODEL,
            save_to_dir=None,
            target_size=TARGET_SIZE,
            shuffle=False,
        )
        test_flow = get_train_data(**test_data_dict)
        test_gen = train_generator(test_flow, COLOR_MODEL)
        groundtruths = []
        step = 0
        for (_,), (mask_batch, _) in test_gen:
            for mask in mask_batch:
                groundtruths.append(mask)
            step += 1
            if step >= evaluate_steps:
                break
        predicted_results = model.predict_generator(
            generator=test_gen, steps=evaluate_steps, verbose=1
        )

        label_image_pairs = evaluate_classes(
            predicted_results[0], groundtruths
        )  # [0] images, [1] masks
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

        test_flow = get_train_data(**test_data_dict)
        test_gen = train_generator(test_flow, COLOR_MODEL)
        evaluation = model.evaluate_generator(
            generator=test_gen, steps=evaluate_steps, verbose=1
        )
        # print(model.metrics_names) # [3] output1_acc
        print(evaluation)
        training_validation_evaluation[fold - 1]["test"] = evaluation[3]

    for p_class in classes:
        precision = metrics.precision_score(
            folds_label_image_pairs[p_class]["label"], folds_label_image_pairs[p_class]["image"]
        )
        recall = metrics.recall_score(
            folds_label_image_pairs[p_class]["label"], folds_label_image_pairs[p_class]["image"]
        )
        f1 = metrics.f1_score(
            folds_label_image_pairs[p_class]["label"], folds_label_image_pairs[p_class]["image"]
        )
        output_summary_evaluation.append(precision)
        output_summary_evaluation.append(recall)
        output_summary_evaluation.append(f1)

    ordered_training_validation_evaluation = []
    for fold_evaluation in training_validation_evaluation:
        ordered_training_validation_evaluation.append(format_accuracy(fold_evaluation["training"]))
        ordered_training_validation_evaluation.append(format_accuracy(fold_evaluation["validation"]))
        ordered_training_validation_evaluation.append(format_accuracy(fold_evaluation["test"]))
        ordered_training_validation_evaluation.append(fold_evaluation["epoch"])

    output_summary_evaluation = np.concatenate(
        (output_summary_evaluation, ordered_training_validation_evaluation), axis=None
    )

    with open(evaluation_csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(output_summary_evaluation)
    cprint(f"> evaluate succesfully, wrote file at ", end="")
    cprint(f"{evaluation_csv_file}", color="green", attrs=["bold"], end=".")


def evaluate_training_and_validation(experiment_name_template, fold_list):
    data_dir = "data"

    output_values = []
    for fold in fold_list:
        experiment_name = experiment_name_template.format(fold)
        experiment_name_dir = os.path.join(data_dir, experiment_name)
        training_log_file = os.path.join(experiment_name_dir, "training.csv")

        training_log = pd.read_csv(training_log_file)
        output1_acc = training_log["output1_acc"]
        val_output1_acc = training_log["val_output1_acc"]

        def find_best_accuracy(accuracy_values):
            arg_max_output1_acc = np.argmax(np.array(accuracy_values))
            max_output1_acc = accuracy_values[arg_max_output1_acc]
            return max_output1_acc, arg_max_output1_acc

        max_val_output1_acc, arg_max_val_output1_acc = find_best_accuracy(
            val_output1_acc
        )
        max_val_output1_acc_epoch = arg_max_val_output1_acc + 1
        max_output1_acc = output1_acc[arg_max_val_output1_acc]
        output_values.append(
            {
                "training": max_output1_acc,
                "validation": max_val_output1_acc,
                "epoch": max_val_output1_acc_epoch,
            }
        )

        print(f"fold {fold}, max accuracy @ epoch # {max_val_output1_acc_epoch}")
        print(f"max training accuracy = {format_accuracy(max_output1_acc)}")
        print(f"max validation accuracy = {format_accuracy(max_val_output1_acc)}")

    return output_values


def evaluate_classes(images, groundtruths):
    label_image_pairs = {}
    classes = ["iris", "sclera", "bg"]
    for p_class in classes:
        label_image_pairs[p_class] = {"label": np.empty(0), "image": np.empty(0)}
    for image, label in zip(images, groundtruths):
        if image.shape != label.shape:
            print("Image's shape doesn't match with label's shape")
            exit(1)
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                predicted_class = np.argmax(image[x, y])
                image[x, y, :] = 0
                image[x, y, predicted_class] = 1

                predicted_class = np.argmax(label[x, y])
                label[x, y, :] = 0
                label[x, y, predicted_class] = 1

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

        flatten_label["iris"], flatten_label["sclera"], flatten_label["bg"] = flatten_class_layers(label)
        flatten_image["iris"], flatten_image["sclera"], flatten_image["bg"] = flatten_class_layers(image)

        for p_class in classes:
            label_image_pairs[p_class]["label"] = np.concatenate(
                (
                    label_image_pairs[p_class]["label"],
                    flatten_label[p_class],
                ),
                axis=None,
            )
            label_image_pairs[p_class]["image"] = np.concatenate(
                (
                    label_image_pairs[p_class]["image"],
                    flatten_image[p_class],
                ),
                axis=None,
            )
    return label_image_pairs


def format_accuracy(number):
    return format(number * 100, "3.2f")

if __name__ == "__main__":
    cli()
