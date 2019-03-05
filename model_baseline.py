from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, Flatten, MaxPooling2D, Concatenate, UpSampling2D, Dense, Lambda
from keras.activations import relu
from keras.optimizers import Adam
from keras import backend as K


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


def baseline_v7_multiclass(pretrained_weights=None,
                           num_classes=2,
                           input_size=(256, 256, 5),
                           learning_rate=1e-4):
    input3_size = (int(input_size[0] / 4), int(input_size[1] / 4), 3)
    input3 = Input(shape=input3_size, name='input3')
    conv3_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input3)
    conv3_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv3_1)
    output3 = Dense(num_classes, activation='softmax', name='output3')(conv3_2)

    up3_2 = UpSampling2D(size=(2, 2))(conv3_2)

    input2_size = (int(input_size[0] / 2), int(input_size[1] / 2), 3)
    input2 = Input(shape=input2_size, name='input2')
    conv2_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input2)
    conv2_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv2_1)
    concat2_3 = Concatenate()([conv2_2, up3_2])
    output2 = Dense(
        num_classes, activation='softmax', name='output2')(concat2_3)

    up2_3 = UpSampling2D(size=(2, 2))(concat2_3)

    input1_size = input_size
    input1 = Input(shape=input1_size, name='input1')
    conv1_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input1)
    conv1_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1_1)
    concat1_3 = Concatenate()([conv1_2, up2_3])
    output1 = Dense(
        num_classes, activation='softmax', name='output1')(concat1_3)

    model = mask_model = Model(
        inputs=[input1, input2, input3], outputs=[output1, output2, output3])

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss={
            'output1': 'categorical_crossentropy',
            'output2': 'categorical_crossentropy',
            'output3': 'categorical_crossentropy',
        },
        loss_weights={
            'output1': 1,
            'output2': 2,
            'output3': 4,
        },
        metrics={
            'output1': 'accuracy',
            'output2': 'accuracy',
            'output3': 'accuracy'
        })

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model


def diff_iris_area(y_true, y_pred):
    area = K.cast(K.sum(y_true), 'float32')
    return K.pow((K.sum(y_pred) - area) / area, 2)


def baseline_v8_multiclass(pretrained_weights=None,
                           num_classes=2,
                           input_size=(256, 256, 5),
                           learning_rate=1e-4):
    input3_size = (int(input_size[0] / 4), int(input_size[1] / 4), 5)
    input3 = Input(shape=input3_size, name='input3')
    conv3_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input3)
    conv3_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv3_1)
    output3 = Dense(num_classes, activation='softmax', name='output3')(conv3_2)

    up3_2 = UpSampling2D(size=(2, 2))(conv3_2)

    input2_size = (int(input_size[0] / 2), int(input_size[1] / 2), 5)
    input2 = Input(shape=input2_size, name='input2')
    conv2_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input2)
    conv2_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv2_1)
    concat2_3 = Concatenate()([conv2_2, up3_2])
    output2 = Dense(
        num_classes, activation='softmax', name='output2')(concat2_3)

    up2_3 = UpSampling2D(size=(2, 2))(concat2_3)

    input1_size = input_size
    input1 = Input(shape=input1_size, name='input1')
    conv1_1 = Conv2D(
        6,
        1,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(input1)
    conv1_2 = Conv2D(
        6,
        5,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal')(conv1_1)
    concat1_3 = Concatenate()([conv1_2, up2_3])
    output1 = Dense(
        num_classes, activation='softmax', name='output1')(concat1_3)

    output_iris = Lambda(lambda x: x[:, :, :, 0])(output1)
    output_iris = Activation(
        lambda x: relu(x, threshold=0.2), name='output_iris')(output_iris)

    model = mask_model = Model(
        inputs=[input1, input2, input3],
        outputs=[output1, output2, output3, output_iris])

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss={
            'output1': 'categorical_crossentropy',
            'output2': 'categorical_crossentropy',
            'output3': 'categorical_crossentropy',
            'output_iris': diff_iris_area,
        },
        loss_weights={
            'output1': 1,
            'output2': 2,
            'output3': 4,
            'output_iris': 8,
        },
        metrics={
            'output1': 'accuracy',
            'output2': 'accuracy',
            'output3': 'accuracy',
            'output_iris': 'accuracy',
        })

    #  model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, mask_model
