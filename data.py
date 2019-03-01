from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

Background = [255, 255, 255]
Sclera = [255, 0, 0]
Iris = [0, 255, 0]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Sclera, Iris, Unlabelled])


def add_position_layers(img, size, axis):
    x_range = np.arange(size[0]) / (size[0] - 1)
    y_range = x_range.reshape(-1, 1) / (size[0] - 1)
    X = Y = np.zeros(shape=[size[0], size[1]])
    X[:, :] = x_range
    Y[:, :] = y_range
    img = np.insert(img, -1, X, axis=axis)
    img = np.insert(img, -1, Y, axis=axis)
    return img


def adjustData(img, mask, flag_multi_class, num_class, save_path, target_size):
    if (flag_multi_class):
        img = img / 255
        img = add_position_layers(img, target_size, 3)

        #  img[0, :, :, 3] = X
        #  img[0, :, :, 4] = Y

        #  mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        #  new_mask = np.zeros(mask.shape + (num_class,))
        new_mask = mask / 255

        # for model with reshaped outputs
        #  mask_shape = new_mask.shape
        #  new_mask = np.reshape(
        #  new_mask,
        #  (mask_shape[0], mask_shape[1] * mask_shape[2], mask_shape[3]))

        #  new_mask[0, :, :, 0] = 1
        #  new_mask[0, :, :, 1] = 1 - new_mask[0, :, :, 0]
        #  new_mask[0, :, :, 2] = 0

        #  io.imsave(os.path.join(save_path, f"merged.png"), new_mask[0])
        #  io.imsave(os.path.join(save_path, f"0.png"), new_mask[0, :, :, 0])
        #  io.imsave(os.path.join(save_path, f"1.png"), new_mask[0, :, :, 1])
        #  io.imsave(os.path.join(save_path, f"2.png"), new_mask[0, :, :, 2])

        #  new_mask = np.zeros(mask.shape + (1,))
        #  for i in range(num_class):
        # for one pixel in the image, find the class in mask and convert it into one-hot vector
        # index = np.where(mask == i)
        # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
        # new_mask[index_mask] = 1
        #  new_mask[mask == i,i] = 1
        #  new_mask[mask == i,0] = i

        #  new_mask = np.reshape(
        #  new_mask,
        #  (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
        #  new_mask.shape[3])) if flag_multi_class else np.reshape(
        #  new_mask,
        #  (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))

        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def baseline_v7_adjust_data(img, mask, flag_multi_class, num_class, save_path,
                            target_size):
    if (flag_multi_class):
        img = img / 255
        img1 = img
        img2 = img1[:, ::2, ::2, :]  # img1 / 2
        img3 = img2[:, ::2, ::2, :]  # img1 / 4
        #  img1 = trans.resize(img, target_size)
        #  img2_shape = (int(img1.shape[0] / 2), int(img1.shape[0] / 2), -1)
        #  img2 = trans.resize(img1, img2_shape)
        #  img3_shape = (int(img1.shape[0] / 4), int(img1.shape[0] / 4), -1)
        #  img3 = trans.resize(img1, img3_shape)
        img1 = add_position_layers(img1, target_size, 3)
        #  print(img1.shape, img2.shape, img3.shape)

        mask = mask / 255
        mask1 = mask
        mask2 = mask1[:, ::2, ::2, :]  # mask1 / 2
        mask3 = mask2[:, ::2, ::2, :]  # mask1 / 4
        #  print(mask1.shape, mask2.shape, mask3.shape)
        #  mask1 = trans.resize(mask, target_size)
        #  mask2_shape = (int(mask1.shape[0] / 2), int(mask1.shape[0] / 2), -1)
        #  mask2 = trans.resize(mask1, mask2_shape)
        #  mask3_shape = (int(mask1.shape[0] / 4), int(mask1.shape[0] / 4), -1)
        #  mask3 = trans.resize(mask1, mask3_shape)
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return [img1, img2, img3], [mask1, mask2, mask3]
    #  return {
    #  'input1': img1,
    #  'input2': img2,
    #  'input3': img3
    #  }, {
    #  'output1': mask1,
    #  'output2': mask2,
    #  'output3': mask3
    #  }


def baseline_v7_test_generator(test_path, target_size=(256, 256), color='rgb'):
    file_list = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    for file_name in file_list:
        file_path = os.path.join(test_path, file_name)
        if color == 'rgb':
            imread_flag = cv2.IMREAD_COLOR
        elif color == 'grayscale':
            imread_flag = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(file_path, imread_flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img1 = trans.resize(img, target_size)
        img2 = img1[::2, ::2, :]  # img1 / 2
        img3 = img2[::2, ::2, :]  # img1 / 4
        #  img1 = trans.resize(img, target_size)
        #  img2_shape = (int(img1.shape[0] / 2), int(img1.shape[0] / 2), -1)
        #  img2 = trans.resize(img1, img2_shape)
        #  img3_shape = (int(img1.shape[0] / 4), int(img1.shape[0] / 4), -1)
        #  img3 = trans.resize(img1, img3_shape)
        img1 = np.reshape(img1, (1, ) + img1.shape)
        img2 = np.reshape(img2, (1, ) + img2.shape)
        img3 = np.reshape(img3, (1, ) + img3.shape)

        img1 = add_position_layers(img1, target_size, -1)
        yield [img1, img2, img3]


def trainGenerator(batch_size,
                   train_path,
                   image_folder,
                   mask_folder,
                   save_path,
                   aug_dict,
                   image_color_mode="grayscale",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   flag_multi_class=False,
                   num_class=2,
                   save_to_dir=None,
                   target_size=(256, 256),
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = baseline_v7_adjust_data(img, mask, flag_multi_class,
                                            num_class, save_path, target_size)
        yield (img, mask)


def testGenerator(test_path, target_size=(256, 256), color='rgb'):
    file_list = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    for file_name in file_list:
        file_path = os.path.join(test_path, file_name)
        if color == 'rgb':
            imread_flag = cv2.IMREAD_COLOR
        elif color == 'grayscale':
            imread_flag = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(file_path, imread_flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = trans.resize(img, target_size)
        #  img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        #  img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        img = np.reshape(img,
                         img.shape + (1, )) if color == 'grayscale' else img
        img = np.reshape(img, (1, ) + img.shape)
        img = add_position_layers(img, target_size, 3)
        yield img


def geneTrainNpy(image_path,
                 mask_path,
                 flag_multi_class=False,
                 num_class=2,
                 image_prefix="image",
                 mask_prefix="mask",
                 image_as_gray=True,
                 mask_as_gray=True):
    image_name_arr = glob.glob(
        os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1, )) if image_as_gray else img
        mask = io.imread(
            item.replace(image_path, mask_path).replace(
                image_prefix, mask_prefix),
            as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1, )) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    #  img = img[:,:,0] if len(img.shape) == 3 else img
    img_dim = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img_dim.shape + (3, ))

    #  mask = ((img[:, :, 0] == 1) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    #  img_out[mask, :] = color_dict[0]
    #  mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 1) & (img[:, :, 2] == 0))
    #  img_out[mask, :] = color_dict[1]
    #  mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 1))
    #  img_out[mask, :] = color_dict[2]
    #  mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    #  img_out[mask, :] = color_dict[3]

    #  for i in range(num_class):
    #  img_out[img == i,:] = color_dict[i]

    #  mask = ((img[:,:,0] == 1) & (img[:,:,1] == 0) & (img[:,:,2] == 0))
    #  img_out[mask, :] = color_dict[i]

    #  img_out[img == i, 0] = color_dict[i][0]
    #  img_out[img == i, 1] = color_dict[i][1]
    #  img_out[img == i, 2] = color_dict[i][2]
    print(np.unique(img_out))
    return img_out / 255


def max_rgb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


def saveResult(save_path,
               npyfile,
               file_names,
               weights_name,
               flag_multi_class=False,
               num_class=2):
    for i, item in enumerate(npyfile):
        item = np.reshape(item, (256, 256, num_class))
        file_name = os.path.splitext(file_names[i])[0]

        img = labelVisualize(num_class, COLOR_DICT,
                             item) if flag_multi_class else item[:, :, 0]
        #  cv2.imshow('img', img)
        #  cv2.waitKey(0)
        #  io.imsave(
        #  os.path.join(save_path, f"{file_name}-{weights_name}.png"),
        #  img)
        #  print(
        #  np.min(item[:, :, 0]), np.min(item[:, :, 1]),
        #  np.min(item[:, :, 2]))
        #  print(
        #  np.max(item[:, :, 0]), np.max(item[:, :, 1]),
        #  np.max(item[:, :, 2]))

        visualized_img = max_rgb_filter(item)
        visualized_img[visualized_img > 0] = 1

        io.imsave(
            os.path.join(save_path, f"{file_name}-{weights_name}-merged.png"),
            visualized_img)
        io.imsave(
            os.path.join(save_path, f"{file_name}-{weights_name}-0.png"),
            item[:, :, 0])
        io.imsave(
            os.path.join(save_path, f"{file_name}-{weights_name}-1.png"),
            item[:, :, 1])
        io.imsave(
            os.path.join(save_path, f"{file_name}-{weights_name}-2.png"),
            item[:, :, 2])


def baseline_v7_save_result(save_path,
                            npyfile,
                            file_names,
                            weights_name,
                            flag_multi_class=False,
                            num_class=2):
    for l in range(3):
        layer_output = npyfile[l]
        for i, item in enumerate(layer_output):
            print(item.shape)
            if l == 0:
                output_shape = (256, 256, num_class)
            elif l == 1:
                output_shape = (128, 128, num_class)
            elif l == 2:
                output_shape = (64, 64, num_class)
            item = np.reshape(item, output_shape)
            print(item.shape)
            file_name = os.path.splitext(file_names[i])[0]

            img = labelVisualize(num_class, COLOR_DICT,
                                 item) if flag_multi_class else item[:, :, 0]
            visualized_img = max_rgb_filter(item)
            visualized_img[visualized_img > 0] = 1

            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-merged.png"),
                visualized_img)
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-0.png"),
                item[:, :, 0])
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-1.png"),
                item[:, :, 1])
            io.imsave(
                os.path.join(save_path,
                             f"{file_name}-{weights_name}-{l+1}-2.png"),
                item[:, :, 2])
