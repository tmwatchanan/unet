import os
import cv2
import skimage.transform as trans
import numpy as np

TARGET_SIZE = (256, 256)


def import_image(filename, color):
    #  img = Image.open(filename).convert("LA")
    #  img_np = np.array(img)[:,:,0]
    #  return img_np
    imread_flag = None
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
    img = np.reshape(img,img.shape+(1,)) if color == 'grayscale' else img
    #  img = np.reshape(img,(1,)+img.shape)
    return img


def main():
    mask_file_path = os.path.join('data', 'unet_check', 'labels', 'M-1-1.jpg')
    mask = import_image(mask_file_path, 'rgb')
    #  print(mask)
    print(mask.shape)
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    #  print(mask)
    print(mask.shape)

    num_class = 5
    new_mask = np.zeros(mask.shape + (num_class,))
    print(new_mask.shape)

    i = 1
    new_mask[mask == i,i] = 1
    print(new_mask.shape)
    #  cv2.imshow('2', new_mask[:,:,1])
    #  cv2.waitKey(0)

    flag_multi_class = True
    new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
    print(new_mask.shape)

if __name__ == '__main__':
    main()

