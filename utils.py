import numpy as np
import cv2
from skimage import filters, feature, color


def add_position_layers(img, axis):
    size = (img.shape[1], img.shape[2])
    x_range = np.arange(size[0]) / (size[0] - 1)
    y_range = x_range.reshape(-1, 1) / (size[0] - 1)
    X = Y = np.zeros(shape=[size[0], size[1], 1])
    X[:, :, 0] = x_range
    Y[:, :, 0] = y_range
    img = np.concatenate((img, X), axis=-1)
    img = np.concatenate((img, Y), axis=-1)
    return img


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


def color_to_gray(img, color_model):
    if color_model.lower() == "hsv":
        # extract only the V layer
        img_gray = img[:, :, 2]
    elif color_model.lower() in (
        "rgb",
        "cie",
        "xyz",
        "yuv",
        "yiq",
        "ypbpr",
        "ycbcr",
        "ydbdr",
    ):
        # convert image into RGB first
        img_rgb = color.convert_colorspace(img, color_model, "RGB")
        # then convert RGB to gray
        img_gray = color.rgb2gray(img_rgb)
    else:
        raise Exception(f"color_model {color_model} is invalid.")
    return img_gray


def add_sobel_filters(img, color_model, added_img):
    img_gray = color_to_gray(img, color_model)
    # extract sobel gradient feature layers
    im_sobel_h = filters.sobel_h(img_gray)
    im_sobel_v = filters.sobel_v(img_gray)
    # expand dimensions from (X, Y) to (X, Y, 1)
    im_sobel_h = np.expand_dims(im_sobel_h, axis=-1)
    im_sobel_v = np.expand_dims(im_sobel_v, axis=-1)
    # concatenate feature layers into image
    # (X, Y, 3) then becomes (X, Y, 5)
    added_img = np.concatenate((added_img, im_sobel_h), axis=-1)
    added_img = np.concatenate((added_img, im_sobel_v), axis=-1)
    return added_img


def add_canny_filter(img, color_model, added_img, sigma):
    img_gray = color_to_gray(img, color_model)
    # extract canny edge feature layer = array of booleans
    im_canny = feature.canny(img_gray, sigma=sigma)
    # convert from boolean values to int values (0 or 1)
    im_canny = im_canny.astype(int)
    # expand dimensions from (X, Y) to (X, Y, 1)
    im_canny = np.expand_dims(im_canny, axis=-1)
    # concatenate a canny feature layer into image
    # (X, Y, l) then becomes (X, Y, l+1)
    added_img = np.concatenate((added_img, im_canny), axis=-1)
    return added_img
