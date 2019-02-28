import os
import json
import urllib.request
import cv2
import numpy as np
from keras.utils import to_categorical

Background = [255, 255, 255]
Sclera = [255, 0, 0]
Iris = [0, 255, 0]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Sclera, Iris, Unlabelled])

DATASET_NAME = 'eye_v2'
DATASET_JSON_FILENAME = 'eye_v2-labeled_data'
labeled_data_file_path = os.path.join('datasets',
                                      f"{DATASET_JSON_FILENAME}.json")
dataset_path = os.path.join('datasets', DATASET_NAME)
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(images_path):
    os.makedirs(images_path)
if not os.path.exists(labels_path):
    os.makedirs(labels_path)


def download_from_url(url, external_id, file_ext, dir_name, class_id=None):
    if class_id:
        file_name = f"{external_id}-{str(class_id)}.{file_ext}"
    else:
        file_name = external_id
    file_name = os.path.join(dir_name, file_name)
    u = urllib.request.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta["Content-Length"])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl,
                                       file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        print(status)

    f.close()
    return file_name


def main():
    with open(labeled_data_file_path) as f:
        data = json.load(f)

    for item in data:
        if item.get('Label') == "Skip":
            print(item.get('Label'))
            continue
        external_id = item['External ID']
        masks_by_name = item['Label']['segmentationMasksByName']
        img_url = item['Labeled Data']
        class_0_url = masks_by_name['Background']
        class_1_url = masks_by_name['Sclera']
        class_2_url = masks_by_name['Iris']
        img_path = download_from_url(img_url, external_id, 'jpg', images_path,
                                     0)
        img_0_path = download_from_url(class_0_url, external_id, 'png',
                                       dataset_path, 0)
        img_1_path = download_from_url(class_1_url, external_id, 'png',
                                       dataset_path, 1)
        img_2_path = download_from_url(class_2_url, external_id, 'png',
                                       dataset_path, 2)
        img_0 = cv2.imread(img_0_path)
        img_1 = cv2.imread(img_1_path)
        img_2 = cv2.imread(img_2_path)
        #  print(img_1.shape)

        background_color = [255, 255, 255]
        sclera_color = [0, 0, 255]  # BGR
        iris_color = [0, 255, 0]  # BGR
        img_mask = np.zeros((img_0.shape[0], img_0.shape[1], 1))
        mask = ((img_0[:, :, 0] == background_color[0]) &
                (img_0[:, :, 1] == background_color[1]) &
                (img_0[:, :, 2] == background_color[2]))
        img_mask[mask, 0] = 0
        #  img_mask[mask] = (255,0,0)
        mask = ((img_1[:, :, 0] == sclera_color[0]) &
                (img_1[:, :, 1] == sclera_color[1]) &
                (img_1[:, :, 2] == sclera_color[2]))
        img_mask[mask, 0] = 1
        #  img_mask[mask] = (0,255,0)
        mask = ((img_2[:, :, 0] == iris_color[0]) &
                (img_2[:, :, 1] == iris_color[1]) &
                (img_2[:, :, 2] == iris_color[2]))
        img_mask[mask, 0] = 2
        #  img_mask_vis = labelVisualize(3, COLOR_DICT, img_mask)
        #  img_mask[mask] = (0,0,255)
        #  print(img_mask.shape)

        #  print(img_mask[300, 600])

        reshaped_img_mask = to_categorical(img_mask)
        #  print(reshaped_img_mask[300, 600, 0], reshaped_img_mask[300, 600, 1],
        #  reshaped_img_mask[300, 600, 2])
        #  print(reshaped_img_mask.shape)
        #  print(reshaped_img_mask)

        mask_path = os.path.join(labels_path, external_id)
        cv2.imwrite(mask_path, reshaped_img_mask * 255)

        #  print(np.unique(img_mask))


def labelVisualize(num_class, color_dict, img):
    #  img = img[:,:,0] if len(img.shape) == 3 else img
    img_dim = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img_dim.shape + (3, ))
    print(img.shape)
    print(img_out.shape)
    mask = ((img[:, :, 0] == 1) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    img_out[mask, :] = color_dict[0]
    mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 1) & (img[:, :, 2] == 0))
    img_out[mask, :] = color_dict[1]
    mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 1))
    img_out[mask, :] = color_dict[2]
    mask = ((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    img_out[mask, :] = color_dict[3]
    #  for i in range(num_class):
    #  img_out[img == i,:] = color_dict[i]

    #  mask = ((img[:,:,0] == 1) & (img[:,:,1] == 0) & (img[:,:,2] == 0))
    #  img_out[mask, :] = color_dict[i]

    #  img_out[img == i, 0] = color_dict[i][0]
    #  img_out[img == i, 1] = color_dict[i][1]
    #  img_out[img == i, 2] = color_dict[i][2]
    print(np.unique(img_out))
    return img_out / 255


if __name__ == "__main__":
    main()
