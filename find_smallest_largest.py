import os
import glob
from PIL import Image
import numpy as np

dir_path = os.path.join("datasets", "eye_v5", "images")

i = 1
pixel_list = []
size_list = []
for imgfile in glob.glob((dir_path + "/*")):
    img = Image.open(imgfile)
    pixel_count = img.size[0] * img.size[1]
    print(
        i,
        ") ",
        imgfile,
        ", resolution: ",
        img.size[0],
        "x",
        img.size[1],
        " =",
        pixel_count,
        "pixels",
    )
    pixel_list.append(pixel_count)
    size_list.append(img.size)
    i += 1

min_idx = np.argmin(pixel_list)
max_idx = np.argmax(pixel_list)

print("MIN =", size_list[min_idx], " =", pixel_list[min_idx], "pixels")
print("MAX =", size_list[max_idx], " =", pixel_list[max_idx], "pixels")
