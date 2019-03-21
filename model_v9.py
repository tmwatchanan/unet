import click
import warnings
import datetime
import numpy as np
import os
import csv
import cv2
from itertools import tee
from termcolor import colored, cprint
from utils import add_position_layers, max_rgb_filter
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LambdaCallback
from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, Flatten, MaxPooling2D, Concatenate, UpSampling2D, Dense, Lambda, ThresholdedReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import plot_model
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

Iris = [0, 255, 0]
Sclera = [255, 0, 0]
Background = [255, 255, 255]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Sclera, Iris, Unlabelled])


@click.group()
def cli():
    #  click.echo("This is main function echoed by Click")
    pass


class PredictOutput(Callback):
    def __init__(self, test_set_dir, target_size, color, weights_dir,
                 num_classes, predicted_set_dir, test_files, period):
        #  self.out_log = []
        self.test_set_dir = test_set_dir
        self.target_size = target_size
        self.color = color
        self.weights_dir = weights_dir
        self.num_classes = num_classes
        self.predicted_set_dir = predicted_set_dir
        self.test_files = test_files
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        predict_epoch = epoch + 1
        if (predict_epoch % self.period == 0):
            # test the model
            test_gen = test_generator(self.test_set_dir, self.target_size,
                                      self.color)
            num_test_files = 12  # sum(1 for _ in test_gen)
            results = self.model.predict_generator(
                test_gen, steps=num_test_files, verbose=1)
            #  print(test_files)
            last_weights_file = f"{predict_epoch:08d}"
            save_result(
                self.predicted_set_dir,
                results,
                file_names=self.test_files,
                weights_name=last_weights_file,
                flag_multi_class=True,
                num_class=self.num_classes)
        #  self.out_log.append()


@cli.command()
@click.pass_context
def train(ctx):
    #  click.echo('> `train` function')
    cprint("> ", end='')
    cprint("`train`", color='green', end='')
    cprint(" function")
    DATASET_NAME = 'eye_v2'
    MODEL_NAME = 'baseline_v9_multiclass'
    MODEL_INFO = 'softmax-cce-lw_1_0.1'
    LEARNING_RATE = "1e_3"
    EXPERIMENT_NAME = f"{DATASET_NAME}-{MODEL_NAME}-{MODEL_INFO}-lr_{LEARNING_RATE}"
    TEST_DIR_NAME = 'test'
    EPOCH_START = 0
    EPOCH_END = 9000
    MODEL_PERIOD = 100
    BATCH_SIZE = 6  # 10
    STEPS_PER_EPOCH = 1  # None
    INPUT_SIZE = (64, 64, 5)
    TARGET_SIZE = (64, 64)
    NUM_CLASSES = 3
    COLOR = 'rgb'  # rgb, grayscale

    if BATCH_SIZE > 10:
        answer = input(
            f"Do you want to continue using BATCH_SIZE={BATCH_SIZE} [y/n] : ")
        if not answer or answer[0].lower() != 'y':
            print("You can change the value of BATCH_SIZE in this file")
            exit(1)

    dataset_path = os.path.join('data', EXPERIMENT_NAME)
    weights_dir = os.path.join(dataset_path, 'weights')
    training_set_dir = os.path.join(dataset_path, 'train')
    training_images_set_dir = os.path.join(training_set_dir, 'images')
    training_labels_set_dir = os.path.join(training_set_dir, 'labels')
    training_aug_set_dir = os.path.join(training_set_dir, 'augmentation')
    validation_set_dir = os.path.join(dataset_path, 'validation')
    validation_images_set_dir = os.path.join(validation_set_dir, 'images')
    validation_labels_set_dir = os.path.join(validation_set_dir, 'labels')
    validation_aug_set_dir = os.path.join(validation_set_dir, 'augmentation')
    test_set_dir = os.path.join(dataset_path, TEST_DIR_NAME)
    predicted_set_dir = os.path.join(dataset_path,
                                     f"{TEST_DIR_NAME}-predicted-{COLOR}")
    training_log_file = os.path.join(dataset_path, 'training.csv')
    loss_acc_file = os.path.join(dataset_path, 'loss_acc.csv')
    experiments_setting_file = os.path.join(dataset_path,
                                            'experiment_settings.txt')
    model_file = os.path.join(dataset_path, 'model.png')
    tensorboard_log_dir = os.path.join(dataset_path, 'logs')

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(predicted_set_dir):
        os.makedirs(predicted_set_dir)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    learning_rate = float(LEARNING_RATE.replace("_", "-"))

    def save_experiment_settings_file():
        with open(experiments_setting_file, "a") as f:
            current_datetime = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            f.write(f"{current_datetime}\n")
            f.write(f"MODEL_NAME={MODEL_NAME}\n")
            f.write(f"MODEL_INFO={MODEL_INFO}\n")
            f.write(f"EPOCH_START={EPOCH_START}\n")
            f.write(f"EPOCH_END={EPOCH_END}\n")
            f.write(f"MODEL_PERIOD={MODEL_PERIOD}\n")
            f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
            f.write(f"STEPS_PER_EPOCH={STEPS_PER_EPOCH}\n")
            f.write(f"LEARNING_RATE={LEARNING_RATE}\n")
            f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
            f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
            f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
            f.write(f"COLOR={COLOR}\n")
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
    for root, dirs, files in os.walk(training_images_set_dir):
        num_training += len(files)
    num_validation = 0
    for root, dirs, files in os.walk(validation_images_set_dir):
        num_validation += len(files)
    if STEPS_PER_EPOCH is None:
        STEPS_PER_EPOCH = num_training
    print(f"num_traning={num_training}")
    print(f"num_validation={num_validation}")

    model = create_model(
        pretrained_weights=trained_weights_file,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        learning_rate=learning_rate)  # load pretrained model
    plot_model(model, show_shapes=True, to_file=model_file)

    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    train_gen = train_generator(
        BATCH_SIZE,
        training_set_dir,
        'images',
        'labels',
        data_gen_args,
        save_to_dir=None,
        image_color_mode=COLOR,
        mask_color_mode=COLOR,
        flag_multi_class=True,
        num_class=NUM_CLASSES)
    validation_gen = train_generator(
        BATCH_SIZE,
        validation_set_dir,
        'images',
        'labels',
        data_gen_args,
        save_to_dir=None,
        image_color_mode=COLOR,
        mask_color_mode=COLOR,
        flag_multi_class=True,
        num_class=NUM_CLASSES)
    test_gen = test_generator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)

    test_files = [
        name for name in os.listdir(test_set_dir)
        if os.path.isfile(os.path.join(test_set_dir, name))
    ]
    num_test_files = len(test_files)

    # train the model
    #  new_weights_name = '{epoch:08d}'
    #  new_weights_file = model_filename.format(new_weights_name)
    new_weights_file = '{epoch:08d}.hdf5'
    new_weights_file = os.path.join(weights_dir, new_weights_file)
    model_checkpoint = ModelCheckpoint(
        filepath=new_weights_file,
        monitor='val_acc',
        mode='auto',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        period=MODEL_PERIOD)
    predict_output = PredictOutput(
        test_set_dir,
        TARGET_SIZE,
        COLOR,
        weights_dir,
        NUM_CLASSES,
        predicted_set_dir,
        test_files,
        period=MODEL_PERIOD)
    csv_logger = CSVLogger(training_log_file, append=True)
    tensorboard = TensorBoard(
        log_dir=os.path.join(tensorboard_log_dir, str(time())),
        histogram_freq=0,
        write_graph=True,
        write_images=True)
    last_epoch = []
    save_output_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: (last_epoch.append(epoch)))
    callbacks = [
        model_checkpoint, csv_logger, tensorboard, save_output_callback,
        predict_output
    ]
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCH_END - EPOCH_START,
        initial_epoch=EPOCH_START,
        callbacks=callbacks,
        validation_data=validation_gen,
        validation_steps=num_validation,
        workers=0,
        use_multiprocessing=True)
    #  print(history.history.keys())  # show dict of metrics in history
    #  save_metrics(loss_acc_file=loss_acc_file, history=history, epoch=i)

    plot([EXPERIMENT_NAME])
    #  ctx.invoke(plot, experiment_name=EXPERIMENT_NAME)


def diff_iris_area(y_true, y_pred):
    area_true = K.cast(K.sum(y_true, axis=[1, 2]), 'float32')
    area_pred = K.sum(y_pred, axis=[1, 2])
    normalized_diff = (area_true - area_pred) / area_true
    return K.mean(K.square(normalized_diff), axis=0)


def create_model(pretrained_weights=None,
                 num_classes=2,
                 input_size=(64, 64, 5),
                 learning_rate=1e-4):
    input1_size = input_size
    input1 = Input(shape=input1_size, name='input1')
    conv1_1 = Conv2D(
        6,
        3,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    conv1_2 = Conv2D(
        12,
        3,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(pool1_1)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv1_3 = Conv2D(
        24,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(pool1_2)
    up1_4 = UpSampling2D(size=(2, 2))(conv1_3)
    conv1_5 = Conv2D(
        12,
        3,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(up1_4)
    up1_5 = UpSampling2D(size=(2, 2))(conv1_5)
    conv1_6 = Conv2D(
        6,
        3,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(up1_5)

    output1 = Conv2D(
        num_classes,
        5,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal',
        name='output1')(conv1_6)

    output_iris = Lambda(lambda x: x[:, :, :, 0])(output1)
    output_iris = ThresholdedReLU(theta=0.5, name='output_iris')(output_iris)

    model = Model(inputs=[input1], outputs=[output1, output_iris])

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss={
            'output1': 'categorical_crossentropy',
            'output_iris': diff_iris_area,
        },
        loss_weights={
            'output1': 1,
            'output_iris': 0.01,
        },
        metrics={
            'output1': ['accuracy'],
            'output_iris': ['accuracy'],
        })

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def adjust_data(img, mask, flag_multi_class, num_class, target_size):
    if (flag_multi_class):
        img = img / 255
        img = add_position_layers(img, -1)

        mask = mask / 255
        mask_iris = mask[:, :, :, 0]
    return [img], [mask, mask_iris]


def train_generator(batch_size,
                    train_path,
                    image_folder,
                    mask_folder,
                    aug_dict,
                    image_color_mode="grayscale",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class=False,
                    num_class=2,
                    save_to_dir=None,
                    target_size=(64, 64),
                    seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class,
                                target_size)
        yield (img, mask)


def test_generator(test_path, target_size=(64, 64), color='rgb'):
    file_list = [
        f for f in os.listdir(test_path)
        if os.path.isfile(os.path.join(test_path, f))
    ]
    for file_name in file_list:
        file_path = os.path.join(test_path, file_name)
        if color == 'rgb':
            imread_flag = cv2.IMREAD_COLOR
        elif color == 'grayscale':
            imread_flag = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(file_path, imread_flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1, ) + img.shape)
        img = add_position_layers(img, -1)
        yield [img]


def save_result(save_path,
                npyfile,
                file_names,
                weights_name,
                flag_multi_class=False,
                num_class=2):
    for ol in range(len(npyfile)):
        layer_output = npyfile[ol]
        for i, item in enumerate(layer_output):
            file_name = os.path.splitext(file_names[i])[0]
            if ol == 0:
                output_shape = (64, 64, num_class)
                item = np.reshape(item, output_shape)
                visualized_img = max_rgb_filter(item)
                visualized_img[visualized_img > 0] = 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(
                        os.path.join(
                            save_path,
                            f"{file_name}-{weights_name}-{ol+1}-merged.png"),
                        visualized_img)
                    io.imsave(
                        os.path.join(
                            save_path,
                            f"{file_name}-{weights_name}-{ol+1}-0.png"),
                        item[:, :, 0])
                    io.imsave(
                        os.path.join(
                            save_path,
                            f"{file_name}-{weights_name}-{ol+1}-1.png"),
                        item[:, :, 1])
                    io.imsave(
                        os.path.join(
                            save_path,
                            f"{file_name}-{weights_name}-{ol+1}-2.png"),
                        item[:, :, 2])
            elif ol == 1:
                output_shape = (64, 64)
                item = np.reshape(item, output_shape)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(
                        os.path.join(
                            save_path,
                            f"{file_name}-{weights_name}-{ol+1}-iris.png"),
                        item[:, :])


def save_metrics(loss_acc_file, history, epoch):
    if not os.path.exists(loss_acc_file):
        with open(loss_acc_file, "w") as f:
            f.write(
                'epoch,output1_acc,val_output1_acc,output_iris_acc,val_output_iris_acc,output1_loss,val_output1_loss,output_iris_loss,val_output_iris_loss\n'
            )
    output1_acc = history.history['output1_acc'][-1]
    val_output1_acc = history.history['val_output1_acc'][-1]
    output_iris_acc = history.history['output_iris_acc'][-1]
    val_output_iris_acc = history.history['val_output_iris_acc'][-1]

    output1_loss = history.history['output1_loss'][-1]
    val_output1_loss = history.history['val_output1_loss'][-1]
    output_iris_loss = history.history['output_iris_loss'][-1]
    val_output_iris_loss = history.history['val_output_iris_loss'][-1]

    loss_acc = ','.join(
        str(e) for e in [
            epoch,
            output1_acc,
            val_output1_acc,
            output_iris_acc,
            val_output_iris_acc,
            output1_loss,
            val_output1_loss,
            output_iris_loss,
            val_output_iris_loss,
        ])
    with open(loss_acc_file, "a") as f:
        f.write(f"{loss_acc}\n")


def plot_graph(figure_num, epoch_list, x, y, x_label, y_label, title, legend,
               save_name):
    fig_acc = plt.figure(figure_num)
    plt.plot(epoch_list, x, 'b')
    plt.plot(epoch_list, y, 'g')
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color='k', linestyle='-', linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc='upper left')
    fig_acc.savefig(save_name)


@cli.command()
@click.argument('experiment_name', required=False)
def plot(experiment_name):
    # define paths
    dataset_path = os.path.join('data', experiment_name)
    training_log_file = os.path.join(dataset_path, 'training.csv')

    graphs_dir = os.path.join(dataset_path, 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    output1_acc_file = os.path.join(graphs_dir, "output1_acc.png")
    output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")

    # prepare lists for storing histories
    epoch_list = []
    output1_acc_list = []
    val_output1_acc_list = []
    output_iris_acc_list = []
    val_output_iris_acc_list = []
    output1_loss_list = []
    val_output1_loss_list = []
    output_iris_loss_list = []
    val_output_iris_loss_list = []

    # read loss-acc csv file
    first_line = True
    with open(training_log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if first_line:
                print(f"Column names are {', '.join(row)}")
                first_line = False
            else:
                print(
                    f"\tepoch {row[0]} | {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}"
                )
                epoch_list.append(float(row[0]))
                output1_acc_list.append(float(row[2]))
                val_output1_acc_list.append(float(row[2]))
                output_iris_acc_list.append(float(row[4]))
                val_output_iris_acc_list.append(float(row[4]))
                output1_loss_list.append(float(row[3]))
                val_output1_loss_list.append(float(row[6]))
                output_iris_loss_list.append(float(row[5]))
                val_output_iris_loss_list.append(float(row[8]))

    # plot graphs
    plot_graph(1, epoch_list, output1_acc_list, val_output1_acc_list,
               'Accuracy', 'Epoch',
               f"{experiment_name} - Output 1 Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output1_acc_file)
    plot_graph(2, epoch_list, output_iris_acc_list, val_output_iris_acc_list,
               'Accuracy', 'Epoch',
               f"{experiment_name} - Output Iris Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output_iris_acc_file)
    plot_graph(3, epoch_list, output1_loss_list, val_output1_loss_list, 'Loss',
               'Epoch', f"{experiment_name} - Output 1 Model Loss (cce)",
               ['Train Loss', 'Validation Loss'], output1_loss_file)
    plot_graph(4, epoch_list, output_iris_loss_list, val_output_iris_loss_list,
               'Loss', 'Epoch',
               f"{experiment_name} - Output Iris Model Loss (diff_iris_area)",
               ['Train Loss', 'Validation Loss'], output_iris_loss_file)

    # immediately show plotted graphs
    plt.show()


@cli.command()
@click.argument('experiment_name')
@click.argument('weight')
@click.argument('test_dir_name')
def test(experiment_name, weight, test_dir_name):
    cprint(f"> Running `test` command on ", color='green', end='')
    cprint(f"{experiment_name}", color='green', attrs=['bold'], end='')
    cprint(f" experiment", color='green')
    #  experiment_name = "eye_v2-baseline_v8_multiclass-softmax-cce-lw_8421-lr_1e_3"
    #  weight = "98800"
    #  test_dir_name = 'blind_conj'
    BATCH_SIZE = 6  # 10
    INPUT_SIZE = (64, 64, 5)
    TARGET_SIZE = (64, 64)
    NUM_CLASSES = 3
    COLOR = 'rgb'  # rgb, grayscale

    cprint(f"The weight at epoch#", color='green', end='')
    cprint(f"{weight}", color='green', attrs=['bold'], end='')
    cprint(f" will be used to predict the images in ", color='green', end='')
    cprint(f"{test_dir_name}", color='green', attrs=['bold'], end='')
    cprint(f" directory", color='green')

    if BATCH_SIZE > 10:
        answer = input(
            f"Do you want to continue using BATCH_SIZE={BATCH_SIZE} [y/n] : ")
        if not answer or answer[0].lower() != 'y':
            print("You can change the value of BATCH_SIZE in this file")
            exit(1)

    dataset_path = os.path.join('data', experiment_name)
    weights_dir = os.path.join(dataset_path, 'weights')
    test_set_dir = os.path.join(dataset_path, test_dir_name)
    predicted_set_dirname = f"{test_dir_name}-predicted"
    predicted_set_dir = os.path.join(dataset_path, predicted_set_dirname)
    prediction_setting_file = os.path.join(predicted_set_dir,
                                           'prediction_settings.txt')

    if not os.path.exists(predicted_set_dir):
        os.makedirs(predicted_set_dir)

    def save_prediction_settings_file():
        with open(prediction_setting_file, "w") as f:
            current_datetime = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            f.write(f"{current_datetime}\n")
            f.write(f"experiment_name={experiment_name}\n")
            f.write(f"test_dir_name={test_dir_name}\n")
            f.write(f"weight={weight}\n")
            f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
            f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
            f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
            f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
            f.write(f"=======================\n")

    save_prediction_settings_file()

    trained_weights_filename = f"{weight}.hdf5"
    trained_weights_file = os.path.join(weights_dir, trained_weights_filename)

    # load pretrained model
    model = create_model(
        pretrained_weights=trained_weights_file,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE)

    test_files = [
        name for name in os.listdir(test_set_dir)
        if os.path.isfile(os.path.join(test_set_dir, name))
    ]
    #  print(test_files)
    num_test_files = len(test_files)

    # test the model
    test_gen = test_generator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    results = model.predict_generator(
        test_gen, steps=num_test_files, verbose=1)
    save_result(
        predicted_set_dir,
        results,
        file_names=test_files,
        weights_name=weight,
        flag_multi_class=True,
        num_class=NUM_CLASSES)
    cprint(
        f"> `test` command was successfully run, the predicted result will be in ",
        color='green',
        end='')
    cprint(f"{predicted_set_dirname}", color='green', attrs=['bold'])


if __name__ == '__main__':
    cli()
