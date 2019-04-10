import os
#  from model import unet_v2, ModelCheckpoint
from model_baseline import baseline_v2_multiclass, ModelCheckpoint
from data import trainGenerator, testGenerator, saveResult
from keras.models import Model

DATASET_NAME = 'eye-multiclass-baseline_v2-softmax-cce-lr1e_2'
dataset_path = os.path.join('data', DATASET_NAME)
weights_name = DATASET_NAME + "-{}"
CONTINUED_WEIGHT = '4500'
NUM_CLASSES = 3
COLOR = 'rgb'
BATCH_SIZE = 1

weights_dir = os.path.join(dataset_path, 'weights')
eval_set_dir = os.path.join(dataset_path, 'eval')
mask_set_dir = os.path.join(dataset_path, f"mask-{COLOR}")

model_filename = "{}.hdf5"
if CONTINUED_WEIGHT:
    #  trained_weights_name = weights_name.format(CONTINUED_WEIGHT)
    trained_weights_name = CONTINUED_WEIGHT
    trained_weights_file = model_filename.format(trained_weights_name)
    trained_weights_file = os.path.join(weights_dir, trained_weights_file)
else:
    trained_weights_file = None

INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)
input_size = INPUT_SIZE + (3, )
model, mask_model = baseline_v2_multiclass(
    pretrained_weights=trained_weights_file,
    num_classes=NUM_CLASSES,
    input_size=input_size)

data_gen_args = dict()
eval_gen = trainGenerator(
    BATCH_SIZE,
    eval_set_dir,
    'images',
    'labels',
    mask_set_dir,
    data_gen_args,
    save_to_dir=None,
    image_color_mode=COLOR,
    mask_color_mode=COLOR,
    flag_multi_class=True,
    num_class=NUM_CLASSES)
score = model.evaluate_generator(generator=eval_gen, steps=1)
print(model.metrics_names)
print(score)
