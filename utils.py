import numpy as np
import cv2


def add_position_layers(img, axis):
    size = (img.shape[1], img.shape[2])
    x_range = np.arange(size[0]) / (size[0] - 1)
    y_range = x_range.reshape(-1, 1) / (size[0] - 1)
    X = Y = np.zeros(shape=[size[0], size[1]])
    X[:, :] = x_range
    Y[:, :] = y_range
    img = np.insert(img, -1, X, axis=axis)
    img = np.insert(img, -1, Y, axis=axis)
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
