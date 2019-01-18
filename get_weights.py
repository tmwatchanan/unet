import os
from model_baseline import *
from data import *

DATASET_NAME = 'multi-eye'
TRAINED_MODEL_EPOCH = 100

#  model = unet() # new model
#  trained_weights_name = "unet_" + DATASET_NAME + "-baseline-%d.hdf5" % (TRAINED_MODEL_EPOCH)
#  trained_weights_name = "unet_01-grey.hdf5"
trained_weights_name = f"{DATASET_NAME}-{TRAINED_MODEL_EPOCH}.hdf5"
trained_weights_file = os.path.join('data', DATASET_NAME, 'weights',
                                    trained_weights_name)

model = baseline_v2_multiclass(
    pretrained_weights=trained_weights_file)  # load pretrained model
print(model.get_weights())
