"""
Unpack the dataset's numpy arrays.
"""

import numpy as np

split = 'uniform'

cartesian_coordinates = np.load('../../data/cartesian_coordinates.npy')
pixel_centers = np.load('../../data/pixel_centers.npy')
image_squares = np.load('../../data/image_squares.npy')


if split == 'uniform':
	indices = np.random.permutations(len(cartesian_coordinates))
	training_indices = indices[0:2352]
	testing_indices = indices[2352:]
elif split == 'quadrant':


def get_train_data(split):
	"""
	Get training data for the networks. 

	Args:
		split: The type of split imposed on the dataset. 
			Can be either uniform or quadrant. 
	Returns:
		training_data: Numpy array containing the training data.
	"""

	if split == 'uniform':
		

