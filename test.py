from model import *
from data import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

image_filename = 'E-1-1.jpg'
DATASET_NAME = 'eye_graylayer'
WEIGHT_NAME = "unet-eye-10"
COLOR = 'rgb' # rgb, grayscale
INPUT_SIZE = (256, 256)
TARGET_SIZE = (256, 256)
FLAG_PLOT_MODEL = True

dataset_dir = os.path.join('data', DATASET_NAME)
test_dir = os.path.join(dataset_dir, "test")
save_dir = os.path.join(dataset_dir, f"test-predicted-{COLOR}")
graylayer_dir = os.path.join(dataset_dir, "graylayer")
weight_dir = os.path.join(dataset_dir, 'weights')
image_dir = os.path.join(dataset_dir, 'dataset')

weights_file = os.path.join(weight_dir, f"{WEIGHT_NAME}.hdf5")
model_figure_file = os.path.join(dataset_dir, f"{WEIGHT_NAME}-model.png")
image_test_file = os.path.join(image_dir, image_filename)

## Run ------------------------------------------------------------------------

def layer_to_visualize(model, layer_name, image, save=None):
    layer = model.get_layer(layer_name)
    inputs = [keras.learning_phase()] + model.inputs

    _convout1_f = keras.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(image)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
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
    test_files = [name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))]
    num_test_files = len(test_files)
    results = model.predict_generator(testGene, steps=num_test_files, verbose=1)
    saveResult(save_dir, results, file_names=test_files, weights_name=WEIGHT_NAME)


def create_model():
    if COLOR == 'rgb':
        input_size = INPUT_SIZE + (3,)
    elif COLOR == 'gray':
        input_size = INPUT_SIZE + (1,)
    model = unet(pretrained_weights=weights_file, input_size=input_size) # load pretrained model
    return model


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
    img = np.reshape(img,img.shape+(1,)) if COLOR == 'grayscale' else img
    img = np.reshape(img,(1,)+img.shape)
    #  img = np.expand_dims(img, axis=0)
    return img


def main():
    model = create_model()
    #  test()
    #  img = read_image(image_test_file)
    #  layer_to_visualize(model, layer_name='conv2d_1', image=img) 
    file_list = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    for file_name in file_list:
        file_path = os.path.join(test_dir, file_name)
        img = read_image(file_path)
        layer_to_visualize(model, layer_name='conv2d_1', image=img, save=file_name)


if __name__ == "__main__":
    main()


