import os
from model_baseline import *
from data import *

DATASET_NAME = 'eye'
TRAINED_MODEL_EPOCH = 50

model = unet() # new model
#  trained_weights_name = "unet_" + DATASET_NAME + "-baseline-%d.hdf5" % (TRAINED_MODEL_EPOCH)
#  trained_weights_name = "unet_01-grey.hdf5"
trained_weights_name = "unet_eye-01-ok.hdf5"

model = unet(pretrained_weights=trained_weights_name) # load pretrained model
print(model.get_weights())
