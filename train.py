import os
import datetime
#  from model import unet_v2
from model_baseline import baseline_v8_multiclass
from baseline_v8_data import train_generator, test_generator, save_result, save_metrics, plot_loss_acc
from keras.callbacks import ModelCheckpoint

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
model, mask_model = globals()[MODEL_NAME](
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
        print(history.history.keys())  # show dict of metrics in history
        save_metrics(loss_acc_file=loss_acc_file, history=history, epoch=i)

    # test the model
    test_gen_softmax = test_generator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    softmax_results = model.predict_generator(
        test_gen_softmax, steps=num_test_files, verbose=1)
    #  for idx, item in enumerate(softmax_results):
    #  print(item[16384, :])

    test_gen = test_generator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    results = mask_model.predict_generator(
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
