import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def baseline(pretrained_weights=None,
             input_size=(256, 256, 3),
             learning_rate=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(
        3,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(
        1,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)

    model = Model(input=inputs, output=conv2)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mean_squared_error',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def baseline_v2(pretrained_weights=None,
                input_size=(256, 256, 3),
                learning_rate=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(
        1,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv2)

    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mean_squared_error',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def baseline_v2_multiclass(pretrained_weights=None,
                           input_size=(256, 256, 3),
                           learning_rate=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(
        3,
        1,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal')(conv2)

    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mean_squared_error',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
