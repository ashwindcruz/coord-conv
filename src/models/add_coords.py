import tensorflow as tf

def add_coords_layers(input_tensor):
	"""
	For a given tensor, add additional layers specifying the xy coordinates. 

	Args:
		input_tensor: An input tensor not containing coordinate layers. 
	Returns:
		output_tensor: Similar to input tensor but with two additional layers
			specifying xy coordinates. 
	"""
	batch_size_tensor = tf.shape(input_tensor)[0]
	x_dim = tf.shape(input_tensor)[1]
	y_dim = tf.shape(input_tensor)[2]
	xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.int32)
	xx_ones = tf.expand_dims(xx_ones, -1)
	xx_range = tf.tile( 
		tf.expand_dims(tf.range(y_dim), 0), [batch_size_tensor, 1])
	xx_range = tf.expand_dims(xx_range, 1)
	xx_channel = tf.matmul(xx_ones, xx_range)
	xx_channel = tf.expand_dims(xx_channel, -1)
	yy_ones = tf.ones([batch_size_tensor, y_dim], dtype=tf.int32)
	yy_ones = tf.expand_dims(yy_ones, 1)
	yy_range = tf.tile(
		tf.expand_dims(tf.range(x_dim), 0), [batch_size_tensor, 1])
	yy_range = tf.expand_dims(yy_range, -1)
	yy_channel = tf.matmul(yy_range, yy_ones)
	yy_channel = tf.expand_dims(yy_channel, -1)

	x_dim = tf.cast(x_dim, tf.float32)
	y_dim = tf.cast(y_dim, tf.float32)

	xx_channel = tf.cast(xx_channel, tf.float32) / (x_dim - 1)
	yy_channel = tf.cast(yy_channel, tf.float32) / (y_dim - 1)
	xx_channel = xx_channel*2 - 1
	yy_channel = yy_channel*2 - 1



	ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

	output_tensor = ret
	

	return output_tensor
