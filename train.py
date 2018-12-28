import os
from model import *
from data import *

DATASET_NAME = 'eye'
CONTINUED_WEIGHT = "14" # None
weights_name = "unet_eye-{}"
EPOCH_START = 15
EPOCH_END = 19
STEPS_PER_EPOCH = 500
COLOR = 'rgb' # rgb, gray
INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)

model_filename = "{}.hdf5"
trained_weights_name = weights_name.format(CONTINUED_WEIGHT)
trained_weights_file = model_filename.format(trained_weights_name)

training_set_dir = f"data/{DATASET_NAME}/train"
test_set_dir = f"data/{DATASET_NAME}/test"
predicted_set_dir = f"data/{DATASET_NAME}/test-predicted-rgb"
if not os.path.exists(predicted_set_dir):
    os.makedirs(predicted_set_dir)

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
myGene = trainGenerator(2, training_set_dir, 'image', 'label', data_gen_args, save_to_dir = None, image_color_mode=COLOR)

# for each epoch
for i in range(EPOCH_START, EPOCH_END):
    # train the model
    new_weights_name = weights_name.format(str(i))
    new_weights_file = model_filename.format(new_weights_name)
    model_checkpoint = ModelCheckpoint(filepath=new_weights_file, monitor='val_acc', mode='auto', verbose=1, save_best_only=False, save_weights_only=False, period=1)
    model.fit_generator(myGene,steps_per_epoch=STEPS_PER_EPOCH,epochs=1,callbacks=[model_checkpoint])
    # test the model
    testGene = testGenerator(test_set_dir, target_size=TARGET_SIZE, color=COLOR)
    test_files = [name for name in os.listdir(test_set_dir) if os.path.isfile(os.path.join(test_set_dir, name))]
    num_test_files = len(test_files)
    results = model.predict_generator(testGene, steps=num_test_files, verbose=1)
    saveResult(predicted_set_dir, results, file_names=test_files, weights_name=new_weights_name)

#  imgs_train,imgs_mask_train = geneTrainNpy("data/" + DATASET_NAME + "/train/aug/","data/" + DATASET_NAME + "/train/aug/")
#  model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

