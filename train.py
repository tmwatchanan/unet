import os
from model import unet_v2, ModelCheckpoint
#  from model_baseline import baseline_v3_multiclass, ModelCheckpoint
from data import trainGenerator, testGenerator, saveResult
from keras.models import Model

DATASET_NAME = 'eye-multiclass-unet_v2-softmax-mse-64-lr1e_3'
dataset_path = os.path.join('data', DATASET_NAME)
COLOR = 'rgb'  # rgb, grayscale
CONTINUED_WEIGHT = None  # "14", None
weights_name = DATASET_NAME + "-{}"
loss_acc_filename = f"{DATASET_NAME}-loss-acc.csv"
loss_acc_file = os.path.join(dataset_path, loss_acc_filename)
EPOCH_START = 1
EPOCH_END = 4501
BATCH_SIZE = 6  # 10
STEPS_PER_EPOCH = 1  # None
LEARNING_RATE = 1e-3
INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)
NUM_CLASSES = 3

if BATCH_SIZE > 10:
    answer = input(
        f"Do you want to continue using BATCH_SIZE={BATCH_SIZE} [y/n] : ")
    if not answer or answer[0].lower() != 'y':
        print("You can change the value of BATCH_SIZE in this file")
        exit(1)

weights_dir = os.path.join(dataset_path, 'weights')
training_set_dir = os.path.join(dataset_path, 'train')
training_images_set_dir = os.path.join(training_set_dir, 'images')
training_labels_set_dir = os.path.join(training_set_dir, 'labels')
training_aug_set_dir = os.path.join(training_set_dir, 'augmentation')
validation_set_dir = os.path.join(dataset_path, 'validation')
validation_images_set_dir = os.path.join(validation_set_dir, 'images')
validation_labels_set_dir = os.path.join(validation_set_dir, 'labels')
validation_aug_set_dir = os.path.join(validation_set_dir, 'augmentation')
test_set_dir = os.path.join(dataset_path, 'test')
predicted_set_dir = os.path.join(dataset_path, f"test-predicted-{COLOR}")
mask_set_dir = os.path.join(dataset_path, f"mask-{COLOR}")

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
if not os.path.exists(predicted_set_dir):
    os.makedirs(predicted_set_dir)
if not os.path.exists(mask_set_dir):
    os.makedirs(mask_set_dir)

model_filename = "{}.hdf5"
if CONTINUED_WEIGHT:
    #  trained_weights_name = weights_name.format(CONTINUED_WEIGHT)
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
print(num_training, num_validation)

if COLOR == 'rgb':
    input_size = INPUT_SIZE + (3, )
elif COLOR == 'grayscale':
    input_size = INPUT_SIZE + (1, )
model, mask_model = unet_v2(
    pretrained_weights=trained_weights_file,
    num_classes=NUM_CLASSES,
    input_size=input_size,
    learning_rate=LEARNING_RATE)  # load pretrained model

#  model_file += "-{epoch:02d}-{val_acc:.2f}.hdf5"

data_gen_args = dict(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
train_gen = trainGenerator(
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
validation_gen = trainGenerator(
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

loss_acc_list = []
if not os.path.exists(loss_acc_file):
    with open(loss_acc_file, "w") as f:
        f.write('epoch,acc,val_acc,loss,val_loss,cce,val_cce,mae,val_mae,mape,val_mape\n')

# for each epoch
for i in range(EPOCH_START, EPOCH_END):
    # train the model
    #  new_weights_name = weights_name.format(str(i))
    new_weights_name = str(i)
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
    print(history.history.keys())
    trained_acc = history.history['acc'][-1]
    trained_val_acc = history.history['val_acc'][-1]
    trained_loss = history.history['loss'][-1]
    trained_val_loss = history.history['val_loss'][-1]
    trained_cce = history.history['categorical_crossentropy'][-1]
    trained_val_cce = history.history['val_categorical_crossentropy'][-1]
    trained_mae = history.history['mean_absolute_error'][-1]
    trained_val_mae = history.history['val_mean_absolute_error'][-1]
    trained_mape = history.history['mean_absolute_percentage_error'][-1]
    trained_val_mape = history.history['val_mean_absolute_percentage_error'][-1]
    loss_acc = ','.join(
        str(e) for e in
        [i, trained_acc, trained_val_acc, trained_loss, trained_val_loss, trained_cce, trained_val_cce, trained_mae, trained_val_mae, trained_mape, trained_val_mape])
    with open(loss_acc_file, "a") as f:
        f.write(f"{loss_acc}\n")
    # test the model
    test_gen_softmax = testGenerator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    softmax_results = model.predict_generator(
        test_gen_softmax, steps=num_test_files, verbose=1)
    #  for idx, item in enumerate(softmax_results):
        #  print(item[16384, :])

    test_gen = testGenerator(
        test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    results = mask_model.predict_generator(
        test_gen, steps=num_test_files, verbose=1)
    print(test_files)
    print(new_weights_name)
    if (i == 1) or (i % 100 == 0):
        saveResult(
            predicted_set_dir,
            results,
            file_names=test_files,
            weights_name=new_weights_name,
            flag_multi_class=True,
            num_class=NUM_CLASSES)

#  imgs_train,imgs_mask_train = geneTrainNpy("data/" + DATASET_NAME + "/train/aug/","data/" + DATASET_NAME + "/train/aug/")
#  model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
