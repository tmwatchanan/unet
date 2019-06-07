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

import click
import cv2
import matplotlib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import six
import skimage
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (Callback, CSVLogger, LambdaCallback,
                             ModelCheckpoint)
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, Lambda, MaxPooling2D, Permute,
                          Reshape, ThresholdedReLU, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from scipy import ndimage
from tensorflow.python.keras.callbacks import TensorBoard
from termcolor import colored, cprint

from utils import add_sobel_filters, max_rgb_filter

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
    for loo in range(1, 16 + 1):
        #  click.echo('> `train` function')
        cprint("> ", end="")
        cprint("`train`", color="green", end="")
        cprint(" function")
        DATASET_NAME = "eye_v3"
        COLOR_MODEL = "hsv"  # rgb, hsv, ycbcr, gray
        MODEL_NAME = "model_v25_multiclass"
        MODEL_INFO = f"softmax-cce-lw_1_0.01-{COLOR_MODEL}-loo_{loo}"
        BATCH_NORMALIZATION = True
        LEARNING_RATE = "1e_2"
        EXPERIMENT_NAME = (
            f"{DATASET_NAME}-{MODEL_NAME}-{MODEL_INFO}-lr_{LEARNING_RATE}"
            + ("-bn" if BATCH_NORMALIZATION else "")
        )
        TEST_DIR_NAME = "test"z
        EPOCH_START = 0
        EPOCH_END = 5000
        MODEL_PERIOD = 100
        BATCH_SIZE = 6  # 10
        STEPS_PER_EPOCH = 1  # None
        INPUT_SIZE = (256, 256)
        TARGET_SIZE = (256, 256)
        NUM_CLASSES = 3
        SAVE_EACH_LAYER = False
        FIT_VERBOSE = 2  # 0 = silent, 1 = progress bar, 2 = one line per epoch

        if BATCH_SIZE > 10:
            answer = input(
                f"Do you want to continue using BATCH_SIZE={BATCH_SIZE} [y/n] : "
            )
            if not answer or answer[0].lower() != "y":
                print("You can change the value of BATCH_SIZE in this file")
                exit(1)

        
        dataset_path = os.path.join("data", EXPERIMENT_NAME)
        weights_dir = os.path.join(dataset_path, "weights")
        training_set_dir = os.path.join(dataset_path, "train")
        training_images_set_dir = os.path.join(training_set_dir, "images")
        training_labels_set_dir = os.path.join(training_set_dir, "labels")
        training_aug_set_dir = os.path.join(training_set_dir, "augmentation")
        validation_set_dir = os.path.join(dataset_path, "validation")
        validation_images_set_dir = os.path.join(validation_set_dir, "images")
        validation_labels_set_dir = os.path.join(validation_set_dir, "labels")
        validation_aug_set_dir = os.path.join(validation_set_dir, "augmentation")
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
                f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
                f.write(f"STEPS_PER_EPOCH={STEPS_PER_EPOCH}\n")
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
        if STEPS_PER_EPOCH is None:
            STEPS_PER_EPOCH = num_training
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
            batch_size=BATCH_SIZE,
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
            batch_size=BATCH_SIZE,
            train_path=validation_set_dir,
            image_color=COLOR_MODEL,
            mask_color=COLOR_MODEL,
            save_to_dir=None,
            target_size=TARGET_SIZE,
        )
        validation_flow = get_train_data(**validation_data_dict)
        validation_gen = train_generator(validation_flow, COLOR_MODEL)

        # train the model
        #  new_weights_name = '{epoch:08d}'
        #  new_weights_file = model_filename.format(new_weights_name)
        new_weights_file = "{epoch:08d}.hdf5"
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
        predict_output = PredictOutput(
            test_set_dir,
            COLOR_MODEL,
            weights_dir,
            TARGET_SIZE,
            NUM_CLASSES,
            predicted_set_dir,
            period=MODEL_PERIOD,
            save_each_layer=SAVE_EACH_LAYER,
            fit_verbose=FIT_VERBOSE,
        )
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
            predict_output,
            time_history_callback,
        ]
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCH_END,
            initial_epoch=EPOCH_START,
            callbacks=callbacks,
            validation_data=validation_gen,
            validation_steps=num_validation,
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
):
    def create_structure(input_size, input_name, output_name):
        input_layer = Input(shape=input_size, name=input_name)

        layer1 = Conv2D(6, 3, padding="same", kernel_initializer="he_normal")(
            input_layer
        )

        if batch_normalization:
            layer1 = BatchNormalization()(layer1)
        layer1 = Activation("sigmoid")(layer1)

        layer2 = MaxPooling2D(pool_size=(2, 2))(layer1)

        layer3 = Conv2D(12, 3, padding="same", kernel_initializer="he_normal")(layer2)
        if batch_normalization:
            layer3 = BatchNormalization()(layer3)
        layer3 = Activation("sigmoid")(layer3)

        layer4 = MaxPooling2D(pool_size=(2, 2))(layer3)

        layer5 = Conv2D(24, 3, padding="same", kernel_initializer="he_normal")(layer4)
        if batch_normalization:
            layer5 = BatchNormalization()(layer5)
        layer5 = Activation("sigmoid")(layer5)

        layer6 = MaxPooling2D(pool_size=(2, 2))(layer5)

        layer7 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(layer6)
        if batch_normalization:
            layer7 = BatchNormalization()(layer7)
        layer7 = Activation("sigmoid")(layer7)

        layer8 = MaxPooling2D(pool_size=(2, 2))(layer7)

        layer9 = Conv2D(64, 1, padding="same", kernel_initializer="he_normal")(layer8)
        if batch_normalization:
            layer9 = BatchNormalization()(layer9)
        layer9 = Activation("sigmoid")(layer9)

        layer10 = UpSampling2D(size=(2, 2))(layer9)

        layer11 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(layer10)
        if batch_normalization:
            layer11 = BatchNormalization()(layer11)
        layer11 = Activation("sigmoid")(layer11)

        layer12 = UpSampling2D(size=(2, 2))(layer11)

        layer13 = Conv2D(24, 3, padding="same", kernel_initializer="he_normal")(layer12)
        if batch_normalization:
            layer13 = BatchNormalization()(layer13)
        layer13 = Activation("sigmoid")(layer13)

        layer14 = UpSampling2D(size=(2, 2))(layer13)

        layer15 = Conv2D(12, 3, padding="same", kernel_initializer="he_normal")(layer14)
        if batch_normalization:
            layer15 = BatchNormalization()(layer15)
        layer15 = Activation("sigmoid")(layer15)

        layer16 = UpSampling2D(size=(2, 2))(layer15)

        layer17 = Conv2D(6, 3, padding="same", kernel_initializer="he_normal")(layer16)
        if batch_normalization:
            layer17 = BatchNormalization()(layer17)
        layer17 = Activation("sigmoid")(layer17)

        layer18 = Conv2D(
            num_classes,
            3,
            activation="softmax",
            padding="same",
            kernel_initializer="he_normal",
            name=output_name,
        )(layer17)

        return input_layer, layer18

    input1_size = input_size + (3,)
    input2_size = input_size + (2,)
    input1, model_v12 = create_structure(input1_size, "input1", "model_v12")
    input2, model_v15 = create_structure(input2_size, "input2", "model_v15")

    concatenated_layer = Concatenate()([input1, model_v12, model_v15])

    layer19 = Conv2D(16, 3, padding="same", kernel_initializer="he_normal")(
        concatenated_layer
    )
    if batch_normalization:
        layer19 = BatchNormalization()(layer19)
    layer19 = Activation("sigmoid")(layer19)

    layer20 = Conv2D(16, 3, padding="same", kernel_initializer="he_normal")(layer19)
    if batch_normalization:
        layer20 = BatchNormalization()(layer20)
    layer20 = Activation("sigmoid")(layer20)

    layer21 = Conv2D(16, 3, padding="same", kernel_initializer="he_normal")(layer20)
    if batch_normalization:
        layer21 = BatchNormalization()(layer21)
    layer21 = Activation("sigmoid")(layer21)

    output1 = Conv2D(
        num_classes,
        3,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
        name="output1",
    )(layer21)
    output_iris = Lambda(lambda x: x[:, :, :, 0])(output1)
    output_iris = ThresholdedReLU(theta=0.5, name="output_iris")(output_iris)
    model = Model(
        inputs=[input1, input2], outputs=[output1, output_iris, model_v12, model_v15]
    )

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss={
            "output1": "categorical_crossentropy",
            "output_iris": diff_iris_area,
            "model_v12": "categorical_crossentropy",
            "model_v15": "categorical_crossentropy",
        },
        loss_weights={
            "output1": 1,
            "output_iris": 0.01,
            "model_v12": 1,
            "model_v15": 1,
        },
        metrics={
            "output1": ["accuracy"],
            "output_iris": ["accuracy"],
            "model_v12": ["accuracy"],
            "model_v15": ["accuracy"],
        },
    )

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

    # normalize the values of image to be in range [0, 1]
    if color_model == "hsv":
        # normalize V layer
        processed_img[:, :, 2] /= 255.0
    if color_model == "rgb":
        # normalize RGB layers => 0:3 = 0,1,2
        processed_img[:, :, 0:3] /= 255.0
    return processed_img


def compute_gradients_input(img, color_model):
    # create a duplicate object of image
    processed_img = copy.deepcopy(img)
    # add sobel feature layers
    processed_img = add_sobel_filters(img, color_model, processed_img)
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


def compute_gradients_in_batch(img_batch, img_color_model):
    processed_input_list = []
    for img in img_batch:
        processed_input = compute_gradients_input(img, img_color_model)
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
    )
    image_mask_pair_flow = zip(image_flow, mask_flow)
    return image_mask_pair_flow


def get_test_data(
    test_path, image_folder="images", target_size=(256, 256), image_color="rgb", seed=1
):
    color_convertion_function = get_color_convertion_function(image_color)
    test_datagen = ImageDataGenerator(preprocessing_function=color_convertion_function)
    image_color_mode = "grayscale" if image_color == "gray" else "rgb"
    test_flow = test_datagen.flow_from_directory(
        test_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=1,
        shuffle=False,
        seed=seed,
    )
    return test_flow, test_flow.filenames


def train_generator(image_mask_pair_flow, image_color_model):
    for (img_batch, mask_batch) in image_mask_pair_flow:
        processed_img_array = preprocess_images_in_batch(img_batch, image_color_model)
        gradients = compute_gradients_in_batch(img_batch, image_color_model)
        mask, mask_iris = preprocess_mask_input(mask_batch)
        yield ([processed_img_array, gradients], [mask, mask_iris, mask, mask])


def test_generator(test_flow, image_color_model):
    for img_batch in test_flow:
        processed_img_array = preprocess_images_in_batch(img_batch, image_color_model)
        gradients = compute_gradients_in_batch(img_batch, image_color_model)
        yield [processed_img_array, gradients]


def save_result(
    save_path,
    npyfile,
    file_names,
    weights_name,
    target_size=(256, 256),
    num_class=3,
    save_each_layer=False,
):
    for ol in range(len(npyfile)):
        layer_output = npyfile[ol]
        for i, item in enumerate(layer_output):
            file_name = os.path.split(file_names[i])[1]
            #  file_name=file_names[i]
            if ol in [0, 2, 3]:
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
            elif ol == 1:
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
    output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")

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
        history_data["output_iris_acc"],
        history_data["val_output_iris_acc"],
        "Accuracy",
        "Epoch",
        f"{experiment_name} - Output Iris Model Accuracy",
        ["Train Accuracy", "Validation Accuracy"],
        output_iris_acc_file,
    )
    plot_graph(
        3,
        history_data["epoch"],
        history_data["output1_loss"],
        history_data["val_output1_loss"],
        "Loss",
        "Epoch",
        f"{experiment_name} - Output 1 Model Loss (cce)",
        ["Train Loss", "Validation Loss"],
        output1_loss_file,
    )
    plot_graph(
        4,
        history_data["epoch"],
        history_data["output_iris_loss"],
        history_data["val_output_iris_loss"],
        "Loss",
        "Epoch",
        f"{experiment_name} - Output Iris Model Loss (diff_iris_area)",
        ["Train Loss", "Validation Loss"],
        output_iris_loss_file,
    )

    # immediately show plotted graphs
    #  plt.show()
    return


@cli.command()
@click.argument("experiment_name")
@click.argument("weight")
@click.argument("test_dir_name")
@click.argument("batch_normalization")
def test(experiment_name, weight, test_dir_name, batch_normalization):
    cprint(f"> Running `test` command on ", color="green", end="")
    cprint(f"{experiment_name}", color="green", attrs=["bold"], end=", ")
    cprint(f"{batch_normalization}", color="grey", attrs=["bold"], end=", ")
    cprint(f" experiment", color="green")
    #  experiment_name = "eye_v2-baseline_v8_multiclass-softmax-cce-lw_8421-lr_1e_3"
    #  weight = "98800"
    #  test_dir_name = 'blind_conj'
    BATCH_SIZE = 6  # 10
    INPUT_SIZE = (256, 256, 6)
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    COLOR_MODEL = "hsv"  # rgb, hsv, ycbcr, gray
    SAVE_EACH_LAYER = False
    PREDICT_VERBOSE = 1  # 0 = silent, 1

    cprint(f"The weight at epoch#", color="green", end="")
    cprint(f"{weight}", color="green", attrs=["bold"], end="")
    cprint(f" will be used to predict the images in ", color="green", end="")
    cprint(f"{test_dir_name}", color="green", attrs=["bold"], end="")
    cprint(f" directory", color="green")

    if BATCH_SIZE > 10:
        answer = input(
            f"Do you want to continue using BATCH_SIZE={BATCH_SIZE} [y/n] : "
        )
        if not answer or answer[0].lower() != "y":
            print("You can change the value of BATCH_SIZE in this file")
            exit(1)

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
            f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
            f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
            f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
            f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
            f.write(f"COLOR_MODEL={COLOR_MODEL}\n")
            f.write(f"SAVE_EACH_LAYER={SAVE_EACH_LAYER}\n")
            f.write(f"=======================\n")

    save_prediction_settings_file()

    trained_weights_filename = f"{weight}.hdf5"
    trained_weights_file = os.path.join(weights_dir, trained_weights_filename)

    # load pretrained model
    model = create_model(
        pretrained_weights=trained_weights_file,
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        batch_normalization=batch_normalization,
    )

    # test the model
    test_data_dict = dict(test_path=test_set_dir, target_size=TARGET_SIZE, color_model=COLOR_MODEL)
    test_flow, test_files = get_test_data(**test_data_dict)
    test_gen = test_generator(test_flow, COLOR_MODEL)
    num_test_files = len(test_files)

    results = model.predict_generator(
        test_gen, steps=num_test_files, verbose=PREDICT_VERBOSE
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
@click.argument("dataset")
@click.argument("experiment_name_replacement")
def create(dataset, experiment_name_replacement):
    # experiment_name_replacement = "eye_v3-model_v19_multiclass-softmax-cce-lw_1_0.01-hsv-p1-loo_{}-lr_1e_2-bn"

    eye_v3_dir = os.path.join("datasets", dataset)  # "eye_v3"
    eye_v3_images_dir = os.path.join(eye_v3_dir, "images")
    eye_v3_labels_dir = os.path.join(eye_v3_dir, "labels")

    filenames = next(os.walk(eye_v3_images_dir))[2]
    filenames = natsort.natsorted(filenames, reverse=False)
    print(filenames)

    for (idx, filename) in enumerate(filenames, 0):
        sub_exp_path = os.path.join("data", experiment_name_replacement.format(idx + 1))
        print(idx, filename, sub_exp_path)

        train_path = os.path.join(sub_exp_path, "train")
        validation_path = os.path.join(sub_exp_path, "validation")

        test_images_path = os.path.join(sub_exp_path, "test", "images")
        shutil.copytree(eye_v3_images_dir, test_images_path)

        train_images_path = os.path.join(train_path, "images")
        train_labels_path = os.path.join(train_path, "labels")

        validation_images_path = os.path.join(validation_path, "images")
        validation_labels_path = os.path.join(validation_path, "labels")

        def create_directory(path):
            try:
                os.makedirs(path)
            except FileExistsError:  # directory already exists
                print(f"{path} is already existed.")

        create_directory(train_images_path)
        create_directory(train_labels_path)
        create_directory(validation_images_path)
        create_directory(validation_labels_path)

        train_images_file_list = copy.deepcopy(filenames)
        del train_images_file_list[idx]
        validation_images_file_list = filenames[idx]

        def move_image_file(src_dir, dest_dir, filename):
            src = os.path.join(src_dir, filename)
            dest = os.path.join(dest_dir, filename)
            shutil.copy(src, dest)

        for img in train_images_file_list:
            move_image_file(eye_v3_images_dir, train_images_path, img)
            move_image_file(eye_v3_labels_dir, train_labels_path, img)

        src = os.path.join(eye_v3_images_dir, validation_images_file_list)
        dest = os.path.join(validation_images_path, validation_images_file_list)
        shutil.copy(src, dest)

        src = os.path.join(eye_v3_labels_dir, validation_images_file_list)
        dest = os.path.join(validation_labels_path, validation_images_file_list)
        shutil.copy(src, dest)


if __name__ == "__main__":
    cli()