"""
Unpack the dataset's numpy arrays.
"""

import numpy as np

split = 'uniform'

cartesian_coordinates = \
        np.load('../data/cartesian_coordinates.npy').astype(np.float32)
pixel_centers = np.load('../data/pixel_centers.npy')
image_squares = np.load('../data/image_squares.npy')

num_examples = len(cartesian_coordinates)
num_rows = 56
num_cols = 56

def get_indices(split='quadrant'):
    """
    Get training and testing indices.

    Args:
        split: Type of split applied to the data.
            Can be either uniform or quadrant.
    Returns:
        training_indices: Integer array of training indices.
        testing_indices: Integer array of testing indices.
    """
    num_examples = len(cartesian_coordinates)
    num_rows = 56
    num_cols = 56

    if split == 'uniform':
        indices = np.random.permutation(num_examples)
        training_indices = indices[0:2352]
        testing_indices = indices[2352:]
    elif split == 'quadrant':
        indices_vector = np.linspace(0, num_examples-1, num_examples, dtype=np.int64)
        indices = np.reshape(indices_vector, [num_cols, num_cols])
        testing_indices = np.reshape(indices[28:,28:], 784)
        training_indices = np.array(list(set(indices_vector)-set(testing_indices)))

    return training_indices, testing_indices

def get_data(start_index, indices, batch_size):
    """
    Fetch data.

    Args:
        start_index: Start index of data we want.
        indices: List of complete indices from which we will be selecting.
        batch_size: How large a batch of data we are getting.
    Returns:
        cartesian_coordinates: xy coordinates of data.
        pixel_centers: Subset of data with pixel centers highlighted.
        image_squares: Subset of data with squares rendered around a specific
            pixel.
    """
    indices_to_return = indices[start_index:(start_index+batch_size)]
    cartesian_coordinates_batch = cartesian_coordinates[indices_to_return]
    pixel_centers_batch = pixel_centers[indices_to_return]
    image_squares_batch = image_squares[indices_to_return]

    # Edit the dimensionality of the data so it fits properly into placeholders
    cartesian_coordinates_batch = np.expand_dims(
            cartesian_coordinates_batch, axis=1)
    cartesian_coordinates_batch = np.expand_dims(
            cartesian_coordinates_batch, axis=1)
    pixel_centers_batch = np.expand_dims(pixel_centers_batch, axis=-1)
    image_squares_batch = np.expand_dims(image_squares_batch, axis=-1)

    return cartesian_coordinates_batch, pixel_centers_batch, image_squares_batch

