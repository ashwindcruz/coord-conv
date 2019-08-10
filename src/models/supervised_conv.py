import tensorflow as tf

def model_classification(input_tensor):
	"""
	Highlight the location of a pixel, represented in the input tensor,
	on a 64x64 map.
	Use convolutions to achieve this.
	Using CoordConv is optional.

	Args:
		input_tensor: Tensor representation of a pixel coordinate.
	Returns:
		pixel_map: 64x64 grid highligting the pixel specified in the input.
	"""

	with tf.variable_scope('conv_model_classification', reuse=tf.AUTO_REUSE):
		conv_1 = tf.layers.conv2d(
			input_tensor, 32, 1, activation='relu', name='conv_1')
		conv_2 = tf.layers.conv2d(
			conv_1, 32, 1, activation='relu', name='conv_2')
		conv_3 = tf.layers.conv2d(
			conv_2, 64, 1, activation='relu', name='conv_3')
		conv_4 = tf.layers.conv2d(
			conv_3, 64, 1, activation='relu', name='conv_4')
		conv_5 = tf.layers.conv2d(
			conv_4, 1, 1, name='conv_5')

	return conv_5

def model_rendering(input_tensor):
	"""
	Render a 9x9 square on a 64x64 grid.

	Use convolutions to achieve this.
	Using CoordConv is optional.

	Args:
		input_tensor: 64x64 grid highlighting a single pixel.
	Returns:
		pixel_map: 64x64 grid highligting a 9x9 square centered on single
			pixel.
	"""

	# Specify the filter size and number of channels
	filter_size = 2
	channels = 2

	with tf.variable_scope('conv_model_rendering', reuse=tf.AUTO_REUSE):
		conv_1 = tf.layers.conv2d(
			input_tensor, 8*channels, filter_size,
			padding='same', activation='relu', name='conv_1')
		conv_2 = tf.layers.conv2d(
			conv_1, 8*channels, filter_size,
			padding='same', activation='relu', name='conv_2')
		conv_3 = tf.layers.conv2d(
			conv_2, 16*channels, filter_size,
			padding='same', activation='relu', name='conv_3')
		conv_4 = tf.layers.conv2d(
			conv_3, 16*channels, filter_size,
			padding='same', activation='relu', name='conv_4')
		conv_5 = tf.layers.conv2d(
			conv_4, 1, filter_size,
			padding='same', name='conv_5')

	return conv_5

def model_regression(input_tensor):
	"""
	Highlight the location of a pixel, represented in the input tensor,
	on a 64x64 map.
	Given a pixel map with a single highlighted points, regress the Cartesian
	coordinate's of that point.

	Args:
		input_tensor: Tensor representation of a pixel coordinate.
	Returns:
		coordinates: Pair of coordinates marking the pixel specified in the input.
	"""

	with tf.variable_scope('conv_model_regression', reuse=tf.AUTO_REUSE):

		conv_1 = tf.layers.conv2d(
			input_tensor, 8, 1,  name='conv_1')
		conv_2 = tf.layers.conv2d(
			conv_1, 8, 1,  name='conv_2')
		conv_3 = tf.layers.conv2d(
			conv_2, 8, 1,  name='conv_3')
		conv_4 = tf.layers.conv2d(
			conv_3, 8, 3,  name='conv_4')
		conv_5 = tf.layers.conv2d(
			conv_4, 2, 3, activation=None, name='conv_5')
                conv_5 = tf.layers.conv2d(
			conv_4, 2, 3, activation=None, name='conv_5')

                max_pooling_1 = tf.compat.v1.layers.max_pooling2d(conv_5, pool_size=64, strides=64)
		#global_pooling_1 = tf.reduce_mean(
	#		conv_5, axis=[1,2], name='global_pooling_1')


	return max_pooling_1
