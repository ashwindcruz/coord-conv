# -*- coding: utf-8 -*-
#!/bin/python3

import os

import numpy as np
import tensorflow as tf


# Create the one-hot images and square images
oneshots = np.pad(
            np.eye(3136).reshape((3136, 56, 56, 1)),
            ((0,0), (4,4), (4,4), (0,0)), 'constant')
images = tf.nn.conv2d(oneshots, np.ones((9, 9, 1, 1)), [1]*4, 'SAME')
sess = tf.Session()
images_ = sess.run(images)

# Create the cartesian coordinate set
row_nums = np.linspace(0, 55, 56, dtype=np.uint8) + 4
col_nums = np.linspace(0, 55, 56, dtype=np.uint8) + 4
row_inds, col_inds = np.meshgrid(row_nums, col_nums, indexing='ij')
row_inds = np.reshape(row_inds, [3136])
col_inds = np.reshape(col_inds, [3136])

# Create the 3 fields of the datasets
cartesian_coordinates = np.stack((row_inds, col_inds), axis=-1)
pixel_centers = np.reshape(oneshots, (3136, 64, 64))
image_squares = np.reshape(images_, (3136, 64, 64))

# Save the data
if not os.path.exists('../data'):
    os.makedirs('../data')

np.save('../data/cartesian_coordinates', cartesian_coordinates)
np.save('../data/pixel_centers', pixel_centers)
np.save('../data/image_squares', image_squares)
