import skimage
import os
import numpy as np

img_list = [
    "2238864_20170428_c3_L_01.png",
    "2323956_20161223__R_a1.bmp",
    "2976988_20161215_a3.bmp",
    "E-18-3.bmp",
    "E-21-2.bmp",
    "E-21-4.bmp",
]


def create():
    img_dir = os.path.join("paper", "selected")
    processed_dir = os.path.join("paper", "processed")

    for img_full_filename in img_list:
        img_name, _ = os.path.splitext(img_full_filename)
        actual_img_path = os.path.join(
            "paper", "processed", "groundtruths", f"{img_name}.jpg"
        )
        actual_img = skimage.io.imread(actual_img_path)
        actual_shape = actual_img.shape
        print(img_name, actual_shape)

        img_path = os.path.join(img_dir, img_full_filename)
        img = skimage.io.imread(img_path)
        print(img_full_filename, img.shape)

        h_img = np.array_split(img, 3, axis=0)
        for r_idx, row in enumerate(h_img):
            v_img = np.array_split(row, 4, axis=1)

            for c_idx, column in enumerate(v_img):

                filename = None

                if r_idx == 0 and c_idx == 2:
                    filename = os.path.join(processed_dir, "mask_groundtruths")
                elif r_idx == 0 and c_idx == 3:
                    filename = os.path.join(processed_dir, "mask_predicted")
                elif r_idx == 1 and c_idx == 0:
                    filename = os.path.join(processed_dir, "rgb")
                elif r_idx == 1 and c_idx == 1:
                    filename = os.path.join(processed_dir, "hsv")
                elif r_idx == 1 and c_idx == 2:
                    filename = os.path.join(processed_dir, "gxgy")
                elif r_idx == 1 and c_idx == 3:
                    filename = os.path.join(processed_dir, "rgbhsvgxgy")
                elif r_idx == 2 and c_idx == 0:
                    filename = os.path.join(processed_dir, "unet")
                elif r_idx == 2 and c_idx == 1:
                    filename = os.path.join(processed_dir, "segnet")
                elif r_idx == 2 and c_idx == 2:
                    filename = os.path.join(processed_dir, "rs")
                elif r_idx == 2 and c_idx == 3:
                    filename = os.path.join(processed_dir, "final")

                if filename != None:
                    filename = os.path.join(filename, f"{img_name}.jpg")
                    transformed_img = skimage.transform.resize(column, actual_shape)
                    skimage.io.imsave(filename, transformed_img)


if __name__ == "__main__":
    create()
