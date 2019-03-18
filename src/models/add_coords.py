import tensorflow as tf

def add_coords_layers(input_tensor):
	"""
	For a given tensor, add additional layers specifying the xy coordinates. 

	Args:
		input_tensor: An input tensor not containining coordinate layers. 
	Returns:
		output_tensor: Similar to input tensor but with two additional layers
			specifying xy coordinates. 
	"""
	row_dim = input_tensor.get_shape().as_list()[1]
	col_dim = input_tensor.get_shape().as_list()[2]

	row_inds = tf.linspace(0.0, row_dim-1, row_dim)
	col_inds = tf.linspace(0.0, col_dim-1, col_dim)

	row_matrix, col_matrix = tf.meshgrid(row_inds, col_inds)

	row_matrix = row_matrix[None, :, :, None]
	col_matrix = col_matrix[None, :, :, None]

	output_tensor = tf.concat([input_tensor, row_matrix, col_matrix], axis=-1)

	return output_tensor
