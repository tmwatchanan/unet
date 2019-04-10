from model import *
from data import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from model_baseline import baseline_v7_multiclass

DATASET_NAME = 'eye-multiclass-baseline_v7-softmax-cce-lr1e_2'
TEST_DIR_NAME = 'blind_test'
WEIGHT_NAME = "9000"
COLOR = 'rgb'  # rgb, grayscale
INPUT_SIZE = (256, 256, 5)
TARGET_SIZE = (256, 256)
NUM_CLASSES = 3
FLAG_PLOT_MODEL = True

dataset_dir = os.path.join('data', DATASET_NAME)
test_dir = os.path.join(dataset_dir, TEST_DIR_NAME)
save_dir = os.path.join(dataset_dir, f"{TEST_DIR_NAME}-predicted-{COLOR}")
graylayer_dir = os.path.join(dataset_dir, "graylayer")
weight_dir = os.path.join(dataset_dir, 'weights')
image_dir = os.path.join(dataset_dir, 'dataset')

images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

weights_file = os.path.join(weight_dir, f"{WEIGHT_NAME}.hdf5")
model_figure_file = os.path.join(dataset_dir, f"{WEIGHT_NAME}-model.png")

# unet_check

check_test1 = os.path.join(images_dir, 'E-1-1.jpg')
check_label1 = os.path.join(labels_dir, 'M-1-1.jpg')

# Run ------------------------------------------------------------------------


def layer_to_visualize(model, layer_name, image, save=None):
    layer = model.get_layer(layer_name)
    inputs = [keras.learning_phase()] + model.inputs

    _convout1_f = keras.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(image)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    cv2.imshow('foo', convolutions)
    cv2.waitKey(0)

    #  fig = plt.figure(figsize=(12,8))
    #  ax = fig.add_subplot(n,n,1)
    #  ax.imshow(convolutions, cmap='gray')

    #  if save:
    #  cv2.imwrite(os.path.join(graylayer_dir, f"{save}-graylayer.jpg"), convolutions)
    #  else:
    #  cv2.imshow('foo', convolutions)
    #  cv2.waitKey(0)

    # Visualization of each filter of the layer
    #  fig = plt.figure(figsize=(12,8))
    #  for i in range(len(convolutions)):
    #  print(convolutions[i].shape)
    #  print(convolutions[i])
    #  ax = fig.add_subplot(n,n,i+1)
    #  ax.imshow(convolutions[i], cmap='gray')


def test(model):
    testGene = testGenerator(test_dir, target_size=TARGET_SIZE, color=COLOR)
    if FLAG_PLOT_MODEL:
        plot_model(model, to_file=model_figure_file)
    test_files = [
        name for name in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, name))
    ]
    num_test_files = len(test_files)
    results = model.predict_generator(
        testGene, steps=num_test_files, verbose=1)
    saveResult(
        save_dir, results, file_names=test_files, weights_name=WEIGHT_NAME)


def read_image(file_path):
    imread_flag = None
    if COLOR == 'rgb':
        imread_flag = cv2.IMREAD_COLOR
    elif COLOR == 'grayscale':
        imread_flag = cv2.IMREAD_GRAYSCALE
    #  img = mpimg.imread(file_path)
    img = cv2.imread(file_path, imread_flag)
    img = img / 255
    img = trans.resize(img, TARGET_SIZE)
    # Keras requires the image to be in 4D
    #  img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img, img.shape + (1, )) if COLOR == 'grayscale' else img
    img = np.reshape(img, (1, ) + img.shape)
    #  img = np.expand_dims(img, axis=0)
    return img


def import_image(filename, color):
    #  img = Image.open(filename).convert("LA")
    #  img_np = np.array(img)[:,:,0]
    #  return img_np
    if color == 'rgb':
        #  imread_flag = cv2.IMREAD_color
        imread_flag = 1
    elif color == 'grayscale':
        #  imread_flag = cv2.IMREAD_GRAYSCALE
        imread_flag = 0
    img = cv2.imread(filename, imread_flag)
    img = img / 255
    img = trans.resize(img, TARGET_SIZE)
    #  img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img, img.shape + (1, )) if color == 'grayscale' else img
    #  img = np.reshape(img,(1,)+img.shape)
    return img


def read_images(images_dir, color):
    file_list = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]
    images = []
    for file_name in file_list:
        file_path = os.path.join(images_dir, file_name)
        img = import_image(file_path, color)
        images.append(img)
    return images


def bw_eval():
    model = create_model(model_name='unet_check')

    #  x = import_image(check_test1)
    #  y = import_image(check_label1)
    #  x_predicted = model.predict(x)
    #  print(x_predicted)

    images = read_images(images_dir, color='rgb')
    labels = read_images(labels_dir, color='grayscale')
    images = np.array(images)
    labels = np.array(labels)
    print("Shape of train images is: ", images.shape)
    print("Shape of labels is: ", labels.shape)

    #  print(images)
    #  print(labels)

    #  evaluation = model.evaluate(x=x, y=y, batch_size=2, verbose=1)
    evaluation = model.evaluate(x=images, y=labels, batch_size=2, verbose=1)
    print(evaluation)

    #  test_files = [name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))]
    #  num_test_files = len(test_files)
    #  results = model.predict_generator(testGene, steps=num_test_files, verbose=1)
    #  saveResult(save_dir, results, file_names=test_files, weights_name='zeros')


def create_model(model_name):
    if COLOR == 'rgb':
        input_size = INPUT_SIZE + (3, )
    elif COLOR == 'gray':
        input_size = INPUT_SIZE + (1, )
    if model_name == 'unet':
        model = unet(
            pretrained_weights=weights_file,
            input_size=input_size)  # load pretrained model
    elif model_name == 'unet_check':
        model = unet_check(input_size=input_size)
    return model


def main():
    #  model = create_model(model_name='unet')
    model, _ = baseline_v6_multiclass(
        pretrained_weights=weights_file,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE)
    #  test()
    file_list = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    for file_name in file_list:
        file_path = os.path.join(test_dir, file_name)
        img = read_image(file_path)
        layer_to_visualize(
            model, layer_name='conv2d_1', image=img, save=file_name)


def run_predict():
    model, _ = baseline_v7_multiclass(
        pretrained_weights=weights_file,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE)

    test_files = [
        name for name in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, name))
    ]
    num_test_files = len(test_files)
    test_gen = testGenerator(test_dir, target_size=TARGET_SIZE, color=COLOR)
    results = model.predict_generator(
        test_gen, steps=num_test_files, verbose=1)
    saveResult(
        save_dir,
        results,
        file_names=test_files,
        weights_name=WEIGHT_NAME,
        flag_multi_class=True,
        num_class=NUM_CLASSES)


if __name__ == "__main__":
    #  main()
    #  bw_eval()
    run_predict()
