import os
from model import *
from data import *

DATASET_NAME = 'eye'
CONTINUED_WEIGHT = None # "14", None
weights_name = "unet-eye-{}"
EPOCH_START = 1
EPOCH_END = 2
STEPS_PER_EPOCH = 500
COLOR = 'rgb' # rgb, gray
INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)
BATCH_SIZE = 2

model_filename = "{}.hdf5"
if CONTINUED_WEIGHT:
    trained_weights_name = weights_name.format(CONTINUED_WEIGHT)
    trained_weights_file = model_filename.format(trained_weights_name)
else:
    trained_weights_file = None

data_set_di = f"data/{DATASET_NAME}/dataset"
training_set_dir = f"data/{DATASET_NAME}/train"
training_images_set_dir = f"{training_set_dir}/images"
training_labels_set_dir = f"{training_set_dir}/labels"
training_aug_set_dir = f"{training_set_dir}/augmentation"
validation_set_dir = f"data/{DATASET_NAME}/validation"
validation_images_set_dir = f"{validation_set_dir}/images"
validation_labels_set_dir = f"{validation_set_dir}/labels"
validation_aug_set_dir = f"{validation_set_dir}/augmentation"
test_set_dir = f"data/{DATASET_NAME}/test"
predicted_set_dir = f"data/{DATASET_NAME}/test-predicted-{COLOR}"

if not os.path.exists(predicted_set_dir):
    os.makedirs(predicted_set_dir)

num_training = 0
for root, dirs, files in os.walk(training_images_set_dir):
    num_training += len(files)
num_validation = 0
for root, dirs, files in os.walk(validation_images_set_dir):
    num_validation += len(files)

if COLOR == 'rgb':
    input_size = INPUT_SIZE + (3,)
elif COLOR == 'gray':
    input_size = INPUT_SIZE + (1,)
model = unet(pretrained_weights=trained_weights_file, input_size=input_size) # load pretrained model

#  model_file += "-{epoch:02d}-{val_acc:.2f}.hdf5"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_gen = trainGenerator(BATCH_SIZE, training_set_dir, 'images', 'labels', data_gen_args, save_to_dir = None, image_color_mode=COLOR)
validation_gen = trainGenerator(BATCH_SIZE, validation_set_dir, 'images', 'labels', data_gen_args, save_to_dir = None, image_color_mode=COLOR)
test_gen = testGenerator(test_set_dir, target_size=TARGET_SIZE, color=COLOR)

# for each epoch
for i in range(EPOCH_START, EPOCH_END):
    # train the model
    new_weights_name = weights_name.format(str(i))
    new_weights_file = model_filename.format(new_weights_name)
    model_checkpoint = ModelCheckpoint(filepath=new_weights_file, monitor='val_acc', mode='auto', verbose=1, save_best_only=False, save_weights_only=False, period=1)
    model.fit_generator(train_gen,steps_per_epoch=num_training,epochs=2,callbacks=[model_checkpoint], validation_data=validation_gen, validation_steps=num_validation)
    # test the model
    test_files = [name for name in os.listdir(test_set_dir) if os.path.isfile(os.path.join(test_set_dir, name))]
    num_test_files = len(test_files)
    results = model.predict_generator(test_gen, steps=num_test_files, verbose=1)
    saveResult(predicted_set_dir, results, file_names=test_files, weights_name=new_weights_name)

#  imgs_train,imgs_mask_train = geneTrainNpy("data/" + DATASET_NAME + "/train/aug/","data/" + DATASET_NAME + "/train/aug/")
#  model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

