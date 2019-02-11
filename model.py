from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def unet(pretrained_weights=None, input_size=(256, 256, 3),
         learning_rate=1e-4):
    inputs = Input(input_size)

    gray0 = inputs
    if (input_size == (256, 256, 3)):
        gray0 = Conv2D(
            1,
            1,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(gray0)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_v2(pretrained_weights=None,
            num_classes=2,
            input_size=(256, 256, 3),
            learning_rate=1e-4):
    inputs = Input(input_size)

    gray0 = inputs
    if (input_size == (256, 256, 3)):
        gray0 = Conv2D(
            1,
            1,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(gray0)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_v3(pretrained_weights=None,
            num_classes=2,
            input_size=(256, 256, 3),
            learning_rate=1e-4):
    input_width = input_size[0]
    input_height = input_size[1]

    inputs = Input(input_size)

    gray0 = inputs
    if (input_size == (256, 256, 3)):
        gray0 = Conv2D(
            1,
            1,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(gray0)
    conv1 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(
        1024,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(
        512,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(
        256,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(
        128,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(
        64,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

    conv11 = Reshape((input_height * input_width, num_classes))(conv10)
    #  conv11 = Permute((2, 1))(conv11)
    conv11 = Activation('softmax')(conv11)

    mask_model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=conv11)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model, mask_model
