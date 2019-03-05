import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import cv2
from utils import add_position_layers, max_rgb_filter
from keras.preprocessing.image import ImageDataGenerator

Iris = [0, 255, 0]
Sclera = [255, 0, 0]
Background = [255, 255, 255]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Sclera, Iris, Unlabelled])


def adjust_data(img, mask, flag_multi_class, num_class, save_path,
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
        img1 = add_position_layers(img1, -1)
        img2 = add_position_layers(img2, -1)
        img3 = add_position_layers(img3, -1)
        #  print(img1.shape, img2.shape, img3.shape)

        mask = mask / 255
        mask1 = mask
        mask2 = mask1[:, ::2, ::2, :]  # mask1 / 2
        mask3 = mask2[:, ::2, ::2, :]  # mask1 / 4
        mask_iris = mask[:, :, :, 0]
        #  print(mask1.shape, mask2.shape, mask3.shape)
        #  mask1 = trans.resize(mask, target_size)
        #  mask2_shape = (int(mask1.shape[0] / 2), int(mask1.shape[0] / 2), -1)
        #  mask2 = trans.resize(mask1, mask2_shape)
        #  mask3_shape = (int(mask1.shape[0] / 4), int(mask1.shape[0] / 4), -1)
        #  mask3 = trans.resize(mask1, mask3_shape)
    return [img1, img2, img3], [mask1, mask2, mask3, mask_iris]
    #  return {
    #  'input1': img1,
    #  'input2': img2,
    #  'input3': img3
    #  }, {
    #  'output1': mask1,
    #  'output2': mask2,
    #  'output3': mask3
    #  }


def train_generator(batch_size,
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
        img, mask = adjust_data(img, mask, flag_multi_class, num_class,
                                save_path, target_size)
        yield (img, mask)


def test_generator(test_path, target_size=(256, 256), color='rgb'):
    file_list = [
        f for f in os.listdir(test_path)
        if os.path.isfile(os.path.join(test_path, f))
    ]
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

        img1 = add_position_layers(img1, -1)
        img2 = add_position_layers(img2, -1)
        img3 = add_position_layers(img3, -1)
        yield [img1, img2, img3]


def save_result(save_path,
                npyfile,
                file_names,
                weights_name,
                flag_multi_class=False,
                num_class=2):
    for l in range(3):
        layer_output = npyfile[l]
        for i, item in enumerate(layer_output):
            #  print(item.shape)
            if l == 0:
                output_shape = (256, 256, num_class)
            elif l == 1:
                output_shape = (128, 128, num_class)
            elif l == 2:
                output_shape = (64, 64, num_class)
            item = np.reshape(item, output_shape)
            #  print(item.shape)
            file_name = os.path.splitext(file_names[i])[0]

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


def save_metrics(loss_acc_file, history, epoch):
    if not os.path.exists(loss_acc_file):
        with open(loss_acc_file, "w") as f:
            f.write(
                'epoch,output1_acc,val_output1_acc,output2_acc,val_output2_acc,output3_acc,val_output3_acc,output_iris_acc,val_output_iris_acc\n'
            )
    trained_output1_acc = history.history['output1_acc'][-1]
    trained_val_output1_acc = history.history['val_output1_acc'][-1]
    trained_output2_acc = history.history['output2_acc'][-1]
    trained_val_output2_acc = history.history['val_output2_acc'][-1]
    trained_output3_acc = history.history['output3_acc'][-1]
    trained_val_output3_acc = history.history['val_output3_acc'][-1]
    trained_output_iris_acc = history.history['output_iris_acc'][-1]
    trained_val_output_iris_acc = history.history['val_output_iris_acc'][-1]
    loss_acc = ','.join(
        str(e) for e in [
            epoch,
            trained_output1_acc,
            trained_val_output1_acc,
            trained_output2_acc,
            trained_val_output2_acc,
            trained_output3_acc,
            trained_val_output3_acc,
            trained_output_iris_acc,
            trained_val_output_iris_acc,
        ])
    with open(loss_acc_file, "a") as f:
        f.write(f"{loss_acc}\n")
