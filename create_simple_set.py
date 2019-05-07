import os
import numpy as np
import copy
from PIL import Image

simple_set_dir = os.path.join('datasets', 'simple_set')
if not os.path.exists(simple_set_dir):
    os.makedirs(simple_set_dir)

width = height = 256

im = np.zeros([height,width,3],dtype=np.uint8)

black_np = copy.deepcopy(im)

white_np = copy.deepcopy(im)
white_np.fill(255) # or img[:] = 255

red_np = copy.deepcopy(im)
red_np[:, :, 0] = 255

noise_np = np.random.random((width,height))

gray_np = np.zeros([height,width], dtype=np.uint8)
gray_np.fill(128)

black_im = Image.fromarray(black_np, 'RGB')
black_im_filepath = os.path.join(simple_set_dir, 'black.jpg')
black_im.save(black_im_filepath)

white_im = Image.fromarray(white_np, 'RGB')
white_im_filepath = os.path.join(simple_set_dir, 'white.jpg')
white_im.save(white_im_filepath)

red_im = Image.fromarray(red_np, 'RGB')
red_im_filepath = os.path.join(simple_set_dir, 'red.jpg')
red_im.save(red_im_filepath)

noise_im = Image.fromarray(noise_np, 'RGB')
noise_im_filepath = os.path.join(simple_set_dir, 'noise.jpg')
noise_im.save(noise_im_filepath)

gray_im = Image.fromarray(gray_np, 'L')
gray_im_filepath = os.path.join(simple_set_dir, 'gray.jpg')
gray_im.save(gray_im_filepath)
