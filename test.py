from model import *
from data import *
from keras.utils import plot_model

import os

DATASET_NAME = 'eye'
WEIGHT_NAME = "unet_eye-13"
TEST_SET = "test"
SAVE_NAME = "test-predicted-rgb"
test_dir = os.path.join('data', DATASET_NAME, TEST_SET)
save_dir = os.path.join('data', DATASET_NAME, SAVE_NAME)
COLOR = 'rgb' # rgb, gray
INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)
FLAG_PLOT_MODEL = True

weights_file = f"{WEIGHT_NAME}.hdf5"
model_figure_file = f"{WEIGHT_NAME}-model.png"

## Run ------------------------------------------------------------------------

testGene = testGenerator(test_dir, target_size=TARGET_SIZE, color=COLOR)
if COLOR == 'rgb':
    input_size = INPUT_SIZE + (3,)
elif COLOR == 'gray':
    input_size = INPUT_SIZE + (1,)
model = unet(pretrained_weights=weights_file, input_size=input_size) # load pretrained model
if FLAG_PLOT_MODEL:
    plot_model(model, to_file=model_figure_file)
test_files = [name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))]
num_test_files = len(test_files)
results = model.predict_generator(testGene, steps=num_test_files, verbose=1)
saveResult(save_dir, results, file_names=test_files, weights_name=WEIGHT_NAME)

