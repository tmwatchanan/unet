import numpy as np
import cv2

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def labelVisualize(num_class,color_dict,img):
    print(img.shape)
    print(img[270, 650])
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    print(img)
    print(np.unique(img))
    for i in range(num_class):
        #  img_out[img == i,:] = color_dict[i]
        img_out[img == i, 0] = color_dict[i][0]
        img_out[img == i, 1] = color_dict[i][1]
        img_out[img == i, 2] = color_dict[i][2]
    print(np.unique(img_out))
    return img_out / 255


def main():
    img = cv2.imread('data/multi-eye/E-1-1.jpg')
    v_img = labelVisualize(num_class=3, color_dict=COLOR_DICT, img=img)
    cv2.imshow('123', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
