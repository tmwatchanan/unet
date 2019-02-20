from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, Flatten, MaxPooling2D, Concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


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

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def baseline_v2_multiclass(pretrained_weights=None,
                           num_classes=2,
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
        num_classes,
        1,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal')(conv2)

    mask_model = Model(input=inputs, output=conv3)
    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_crossentropy', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_error'
        ])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model


def baseline_v3_multiclass(pretrained_weights=None,
                           num_classes=2,
                           input_size=(256, 256, 3),
                           learning_rate=1e-4):
    input_width = input_size[0]
    input_height = input_size[1]

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
        num_classes,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv2)

    conv4 = Reshape((input_height * input_width, num_classes))(conv3)
    #  conv4 = Permute((2, 1))(conv4)
    conv4 = Activation('softmax')(conv4)

    mask_model = Model(inputs=inputs, outputs=conv3)
    model = Model(input=inputs, output=conv4)
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model


def baseline_v4_multiclass(pretrained_weights=None,
                           num_classes=2,
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
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(
        num_classes,
        1,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal')(conv2)

    mask_model = Model(input=inputs, output=conv3)
    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_crossentropy', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_error'
        ])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model


def baseline_v5_multiclass(pretrained_weights=None,
                           num_classes=2,
                           input_size=(256, 256, 3),
                           learning_rate=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(pool1)
    conv2_2 = UpSampling2D(size=(2, 2))(conv2_2)

    conv2_1 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)

    concat1 = Concatenate()([conv2_1, conv2_2])

    conv3 = Conv2D(
        num_classes,
        1,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal')(concat1)

    mask_model = Model(input=inputs, output=conv3)
    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_crossentropy', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_error'
        ])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model


def baseline_v6_multiclass(pretrained_weights=None,
                           num_classes=2,
                           input_size=(256, 256, 5),
                           learning_rate=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(pool1)
    conv2_2 = UpSampling2D(size=(2, 2))(conv2_2)

    conv2_1 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1)

    concat1 = Concatenate()([conv2_1, conv2_2])

    conv3 = Conv2D(
        num_classes,
        1,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal')(concat1)

    mask_model = Model(input=inputs, output=conv3)
    model = Model(input=inputs, output=conv3)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_crossentropy', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_error'
        ])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model
