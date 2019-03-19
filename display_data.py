# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:57:02 2019

@author: Ashwin
"""

import matplotlib.pyplot as plt
import numpy as np

cartesian_coordinates = np.load('data/cartesian_coordinates.npy')
pixel_centers = np.load('data/pixel_centers.npy')
image_squares = np.load('data/image_squares.npy')

cartesian_coordinate_single = cartesian_coordinates[0]
pixel_center_single = np.repeat(pixel_centers[0, :, :, None], 3, axis=-1)
image_square_single = np.repeat(image_squares[0, :, :, None], 3, axis=-1)

print('Display data for point 0')

print('Cartesian coordinate is: {}'.format(cartesian_coordinate_single))

plt.figure()
plt.imshow(pixel_center_single)

plt.figure()
plt.imshow(image_square_single)


#image = np.repeat(images_[3000], 3, axis=-1)
#plt.figure()
#plt.imshow(image, cmap=plt.cm.gray)
#

pc_reshape = np.reshape(image_squares, [56,56,64,64])

bottom_quad = pc_reshape[28:, 28:, :,:]

