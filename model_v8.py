import click
import datetime
import numpy as np
import os
import csv
import cv2
from utils import add_position_layers, max_rgb_filter
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, Flatten, MaxPooling2D, Concatenate, UpSampling2D, Dense, Lambda, ThresholdedReLU
from keras.optimizers import Adam
from keras import backend as K

Iris = [0, 255, 0]
Sclera = [255, 0, 0]
Background = [255, 255, 255]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Sclera, Iris, Unlabelled])


@click.group()
def cli():
    #  click.echo("This is main function echoed by Click")
    pass


@cli.command()
def train():
    click.echo('> `train` function')
    TRAIN_FLAG = True
    DATASET_NAME = 'eye_v2'
    MODEL_NAME = 'baseline_v8_multiclass'
    MODEL_INFO = 'softmax-cce-lw_8421'
    LEARNING_RATE = "1e_3"
    EXPERIMENT_NAME = f"{DATASET_NAME}-{MODEL_NAME}-{MODEL_INFO}-lr_{LEARNING_RATE}"
    TEST_DIR_NAME = 'test'
    CONTINUED_WEIGHT = None  # "14", None
    EPOCH_START = 1
    EPOCH_END = 9001
    BATCH_SIZE = 6  # 10
    STEPS_PER_EPOCH = 1  # None
    INPUT_SIZE = (256, 256)
    TARGET_SIZE = (256, 256)
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
    mask_set_dir = os.path.join(dataset_path, f"mask-{COLOR}")
    loss_acc_file = os.path.join(dataset_path, 'loss-acc.csv')
    experiments_setting_file = os.path.join(dataset_path,
                                            'experiment_settings.txt')

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(predicted_set_dir):
        os.makedirs(predicted_set_dir)
    if not os.path.exists(mask_set_dir):
        os.makedirs(mask_set_dir)

    def save_experiment_settings_file():
        with open(experiments_setting_file, "a") as f:
            current_datetime = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            f.write(f"{current_datetime}\n")
            f.write(f"MODEL_NAME={MODEL_NAME}\n")
            f.write(f"MODEL_INFO={MODEL_INFO}\n")
            f.write(f"CONTINUE_WEIGHT={CONTINUED_WEIGHT}\n")
            f.write(f"EPOCH_START={EPOCH_START}\n")
            f.write(f"EPOCH_END={EPOCH_END}\n")
            f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
            f.write(f"STEPS_PER_EPOCH={STEPS_PER_EPOCH}\n")
            f.write(f"LEARNING_RATE={LEARNING_RATE}\n")
            f.write(f"INPUT_SIZE={INPUT_SIZE}\n")
            f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
            f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
            f.write(f"COLOR={COLOR}\n")
            f.write(f"=======================\n")

    if TRAIN_FLAG:
        save_experiment_settings_file()

    model_filename = "{}.hdf5"
    if CONTINUED_WEIGHT:
        trained_weights_name = CONTINUED_WEIGHT
        trained_weights_file = model_filename.format(trained_weights_name)
        trained_weights_file = os.path.join(weights_dir, trained_weights_file)
    else:
        trained_weights_file = None

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

    if COLOR == 'rgb':
        input_size = INPUT_SIZE + (5, )
    elif COLOR == 'grayscale':
        input_size = INPUT_SIZE + (1, )
    learning_rate = float(LEARNING_RATE.replace("_", "-"))
    model = create_model(
        pretrained_weights=trained_weights_file,
        num_classes=NUM_CLASSES,
        input_size=input_size,
        learning_rate=learning_rate)  # load pretrained model

    #  model_file += "-{epoch:02d}-{val_acc:.2f}.hdf5"

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
        mask_set_dir,
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
        mask_set_dir,
        data_gen_args,
        save_to_dir=None,
        image_color_mode=COLOR,
        mask_color_mode=COLOR,
        flag_multi_class=True,
        num_class=NUM_CLASSES)

    test_files = [
        name for name in os.listdir(test_set_dir)
        if os.path.isfile(os.path.join(test_set_dir, name))
    ]
    num_test_files = len(test_files)

    # for each epoch
    for i in range(EPOCH_START, EPOCH_END):
        # train the model
        new_weights_name = str(i)
        if TRAIN_FLAG:
            new_weights_file = model_filename.format(new_weights_name)
            new_weights_file = os.path.join(weights_dir, new_weights_file)
            callbacks = None
            if (i == 1) or (i % 100 == 0):
                model_checkpoint = ModelCheckpoint(
                    filepath=new_weights_file,
                    monitor='val_acc',
                    mode='auto',
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=False,
                    period=1)
                callbacks = [model_checkpoint]
            history = model.fit_generator(
                train_gen,
                steps_per_epoch=STEPS_PER_EPOCH,
                epochs=1,
                callbacks=callbacks,
                validation_data=validation_gen,
                validation_steps=num_validation,
                workers=0,
                use_multiprocessing=True)
            #  print(history.history.keys())  # show dict of metrics in history
            save_metrics(loss_acc_file=loss_acc_file, history=history, epoch=i)

        # test the model
        test_gen = test_generator(
            test_set_dir, target_size=TARGET_SIZE, color=COLOR)
        results = model.predict_generator(
            test_gen, steps=num_test_files, verbose=1)
        #  print(test_files)
        print(f"EPOCH# {new_weights_name}")
        if (i == 1) or (i % 100 == 0):
            save_result(
                predicted_set_dir,
                results,
                file_names=test_files,
                weights_name=new_weights_name,
                flag_multi_class=True,
                num_class=NUM_CLASSES)
        if not TRAIN_FLAG:
            break

    plot_loss_acc(EXPERIMENT_NAME, loss_acc_file)


def diff_iris_area(y_true, y_pred):
    area_true = K.cast(K.sum(y_true, axis=[1, 2]), 'float32')
    area_pred = K.sum(y_pred, axis=[1, 2])
    normalized_diff = (area_true - area_pred) / area_true
    return K.mean(K.square(normalized_diff), axis=0)


def create_model(pretrained_weights=None,
                 num_classes=2,
                 input_size=(256, 256, 5),
                 learning_rate=1e-4):
    input3_size = (int(input_size[0] / 4), int(input_size[1] / 4), 5)
    input3 = Input(shape=input3_size, name='input3')
    conv3_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input3)
    conv3_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv3_1)
    output3 = Dense(num_classes, activation='softmax', name='output3')(conv3_2)
    up3_2 = UpSampling2D(size=(2, 2))(conv3_2)

    input2_size = (int(input_size[0] / 2), int(input_size[1] / 2), 5)
    input2 = Input(shape=input2_size, name='input2')
    conv2_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input2)
    conv2_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv2_1)
    concat2_3 = Concatenate()([conv2_2, up3_2])
    output2 = Dense(
        num_classes, activation='softmax', name='output2')(concat2_3)

    up2_3 = UpSampling2D(size=(2, 2))(concat2_3)

    input1_size = input_size
    input1 = Input(shape=input1_size, name='input1')
    conv1_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input1)
    conv1_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1_1)
    concat1_3 = Concatenate()([conv1_2, up2_3])
    output1 = Dense(
        num_classes, activation='softmax', name='output1')(concat1_3)

    output_iris = Lambda(lambda x: x[:, :, :, 0])(output1)
    output_iris = ThresholdedReLU(theta=0.5, name='output_iris')(output_iris)

    model = Model(
        inputs=[input1, input2, input3],
        outputs=[output1, output2, output3, output_iris])

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss={
            'output1': 'categorical_crossentropy',
            'output2': 'categorical_crossentropy',
            'output3': 'categorical_crossentropy',
            'output_iris': diff_iris_area,
        },
        loss_weights={
            'output1': 1,
            'output2': 2,
            'output3': 4,
            'output_iris': 4,
        },
        metrics={
            'output1': ['accuracy'],
            'output2': ['accuracy'],
            'output3': ['accuracy'],
            'output_iris': ['accuracy', diff_iris_area],
        })

    #  model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def adjust_data(img, mask, flag_multi_class, num_class, save_path,
                target_size):
    if (flag_multi_class):
        img = img / 255
        img1 = img
        img2 = img1[:, ::2, ::2, :]  # img1 / 2
        img3 = img2[:, ::2, ::2, :]  # img1 / 4
        #  img1 = trans.resize(img, target_size)
        #  img2_shape = (int(img1.shape[0] / 2), int(img1.shape[0] / 2), -1)
        #  img2 = trans.resize(img1, img2_shape)
        #  img3_shape = (int(img1.shape[0] / 4), int(img1.shape[0] / 4), -1)
        #  img3 = trans.resize(img1, img3_shape)
        img1 = add_position_layers(img1, -1)
        img2 = add_position_layers(img2, -1)
        img3 = add_position_layers(img3, -1)
        #  print(img1.shape, img2.shape, img3.shape)

        mask = mask / 255
        mask1 = mask
        mask2 = mask1[:, ::2, ::2, :]  # mask1 / 2
        mask3 = mask2[:, ::2, ::2, :]  # mask1 / 4
        mask_iris = mask[:, :, :, 0]
        #  print(mask1.shape, mask2.shape, mask3.shape)
        #  mask1 = trans.resize(mask, target_size)
        #  mask2_shape = (int(mask1.shape[0] / 2), int(mask1.shape[0] / 2), -1)
        #  mask2 = trans.resize(mask1, mask2_shape)
        #  mask3_shape = (int(mask1.shape[0] / 4), int(mask1.shape[0] / 4), -1)
        #  mask3 = trans.resize(mask1, mask3_shape)
    return [img1, img2, img3], [mask1, mask2, mask3, mask_iris]
    #  return {
    #  'input1': img1,
    #  'input2': img2,
    #  'input3': img3
    #  }, {
    #  'output1': mask1,
    #  'output2': mask2,
    #  'output3': mask3
    #  }


def train_generator(batch_size,
                    train_path,
                    image_folder,
                    mask_folder,
                    save_path,
                    aug_dict,
                    image_color_mode="grayscale",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class=False,
                    num_class=2,
                    save_to_dir=None,
                    target_size=(256, 256),
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
                                save_path, target_size)
        yield (img, mask)


def test_generator(test_path, target_size=(256, 256), color='rgb'):
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
        img1 = trans.resize(img, target_size)
        img2 = img1[::2, ::2, :]  # img1 / 2
        img3 = img2[::2, ::2, :]  # img1 / 4
        #  img1 = trans.resize(img, target_size)
        #  img2_shape = (int(img1.shape[0] / 2), int(img1.shape[0] / 2), -1)
        #  img2 = trans.resize(img1, img2_shape)
        #  img3_shape = (int(img1.shape[0] / 4), int(img1.shape[0] / 4), -1)
        #  img3 = trans.resize(img1, img3_shape)
        img1 = np.reshape(img1, (1, ) + img1.shape)
        img2 = np.reshape(img2, (1, ) + img2.shape)
        img3 = np.reshape(img3, (1, ) + img3.shape)

        img1 = add_position_layers(img1, -1)
        img2 = add_position_layers(img2, -1)
        img3 = add_position_layers(img3, -1)
        yield [img1, img2, img3]


def save_result(save_path,
                npyfile,
                file_names,
                weights_name,
                flag_multi_class=False,
                num_class=2):
    for l in range(3):
        layer_output = npyfile[l]
        for i, item in enumerate(layer_output):
            #  print(item.shape)
            if l == 0:
                output_shape = (256, 256, num_class)
            elif l == 1:
                output_shape = (128, 128, num_class)
            elif l == 2:
                output_shape = (64, 64, num_class)
            item = np.reshape(item, output_shape)
            #  print(item.shape)
            file_name = os.path.splitext(file_names[i])[0]

            visualized_img = max_rgb_filter(item)
            visualized_img[visualized_img > 0] = 1

            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-merged.png"),
                visualized_img)
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-0.png"),
                item[:, :, 0])
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-1.png"),
                item[:, :, 1])
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-2.png"),
                item[:, :, 2])


def save_metrics(loss_acc_file, history, epoch):
    if not os.path.exists(loss_acc_file):
        with open(loss_acc_file, "w") as f:
            f.write(
                'epoch,output1_acc,val_output1_acc,output2_acc,val_output2_acc,output3_acc,val_output3_acc,output_iris_acc,val_output_iris_acc,output1_loss,val_output1_loss,output2_loss,val_output2_loss,output3_loss,val_output3_loss,output_iris_loss,val_output_iris_loss\n'
            )
    output1_acc = history.history['output1_acc'][-1]
    val_output1_acc = history.history['val_output1_acc'][-1]
    output2_acc = history.history['output2_acc'][-1]
    val_output2_acc = history.history['val_output2_acc'][-1]
    output3_acc = history.history['output3_acc'][-1]
    val_output3_acc = history.history['val_output3_acc'][-1]
    output_iris_acc = history.history['output_iris_acc'][-1]
    val_output_iris_acc = history.history['val_output_iris_acc'][-1]

    output1_loss = history.history['output1_loss'][-1]
    val_output1_loss = history.history['val_output1_loss'][-1]
    output2_loss = history.history['output2_loss'][-1]
    val_output2_loss = history.history['val_output2_loss'][-1]
    output3_loss = history.history['output3_loss'][-1]
    val_output3_loss = history.history['val_output3_loss'][-1]
    output_iris_loss = history.history['output_iris_loss'][-1]
    val_output_iris_loss = history.history['val_output_iris_loss'][-1]

    loss_acc = ','.join(
        str(e) for e in [
            epoch,
            output1_acc,
            val_output1_acc,
            output2_acc,
            val_output2_acc,
            output3_acc,
            val_output3_acc,
            output_iris_acc,
            val_output_iris_acc,
            output1_loss,
            val_output1_loss,
            output2_loss,
            val_output2_loss,
            output3_loss,
            val_output3_loss,
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


def plot_loss_acc(dataset_name, loss_acc_file):
    # define paths
    dataset_path = os.path.join('data', dataset_name)
    loss_acc_file = os.path.join(dataset_path, f"loss-acc.csv")

    graphs_dir = os.path.join(dataset_path, 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    output1_acc_file = os.path.join(graphs_dir, "output1_acc.png")
    output2_acc_file = os.path.join(graphs_dir, "output2_acc.png")
    output3_acc_file = os.path.join(graphs_dir, "output3_acc.png")
    output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    output2_loss_file = os.path.join(graphs_dir, "output2_loss.png")
    output3_loss_file = os.path.join(graphs_dir, "output3_loss.png")
    output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")

    # prepare lists for storing histories
    epoch_list = []
    output1_acc_list = val_output1_acc_list = []
    output2_acc_list = val_output2_acc_list = []
    output3_acc_list = val_output3_acc_list = []
    output_iris_acc_list = val_output_iris_acc_list = []
    output1_loss_list = val_output1_loss_list = []
    output2_loss_list = val_output2_loss_list = []
    output3_loss_list = val_output3_loss_list = []
    output_iris_loss_list = val_output_iris_loss_list = []

    # read loss-acc csv file
    first_line = True
    with open(loss_acc_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if first_line:
                print(f"Column names are {', '.join(row)}")
                first_line = False
            else:
                print(
                    f"\tepoch {row[0]} | {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}, {row[9]}, {row[10]}, {row[11]}, {row[12]}, {row[13]}, {row[14]}, {row[15]}, {row[16]}"
                )
                epoch_list.append(float(row[0]))
                output1_acc_list.append(float(row[1]))
                val_output1_acc_list.append(float(row[2]))
                output2_acc_list.append(float(row[3]))
                val_output2_acc_list.append(float(row[4]))
                output3_acc_list.append(float(row[5]))
                val_output3_acc_list.append(float(row[6]))
                output_iris_acc_list.append(float(row[7]))
                val_output_iris_acc_list.append(float(row[8]))
                output1_loss_list.append(float(row[9]))
                val_output1_loss_list.append(float(row[10]))
                output2_loss_list.append(float(row[11]))
                val_output2_loss_list.append(float(row[12]))
                output3_loss_list.append(float(row[13]))
                val_output3_loss_list.append(float(row[14]))
                output_iris_loss_list.append(float(row[15]))
                val_output_iris_loss_list.append(float(row[16]))

    # plot graphs
    plot_graph(1, epoch_list, output1_acc_list, val_output1_acc_list,
               'Accuracy', 'Epoch',
               f"{dataset_name} - Output 1 Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output1_acc_file)
    plot_graph(2, epoch_list, output2_acc_list, val_output2_acc_list,
               'Accuracy', 'Epoch',
               f"{dataset_name} - Output 2 Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output2_acc_file)
    plot_graph(3, epoch_list, output3_acc_list, val_output3_acc_list,
               'Accuracy', 'Epoch',
               f"{dataset_name} - Output 3 Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output3_acc_file)
    plot_graph(4, epoch_list, output_iris_acc_list, val_output_iris_acc_list,
               'Accuracy', 'Epoch',
               f"{dataset_name} - Output Iris Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output_iris_acc_file)

    plot_graph(5, epoch_list, output1_loss_list, val_output1_loss_list, 'Loss',
               'Epoch', f"{dataset_name} - Output 1 Model Loss (cce)",
               ['Train Loss', 'Validation Loss'], output1_loss_file)
    plot_graph(6, epoch_list, output2_loss_list, val_output2_loss_list, 'Loss',
               'Epoch', f"{dataset_name} - Output 2 Model Loss (cce)",
               ['Train Loss', 'Validation Loss'], output2_loss_file)
    plot_graph(7, epoch_list, output3_loss_list, val_output3_loss_list, 'Loss',
               'Epoch', f"{dataset_name} - Output 3 Model Loss (cce)",
               ['Train Loss', 'Validation Loss'], output3_loss_file)
    plot_graph(8, epoch_list, output_iris_loss_list, val_output_iris_loss_list,
               'Loss', 'Epoch',
               f"{dataset_name} - Output Iris Model Loss (diff_iris_area)",
               ['Train Loss', 'Validation Loss'], output_iris_loss_file)

    # immediately show plotted graphs
    plt.show()


if __name__ == '__main__':
    cli()
