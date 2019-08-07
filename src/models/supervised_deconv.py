import tensorflow as tf

def model(input_tensor):
	"""
	Highlight the location of a pixel, represented in the input tensor,
	on a 64x64 map.
	Use deconvolutions (aka transposed convolutions) to achieve this.

	Args:
		input_tensor: Tensor representation of a Cartesian pixel coordinate.
	Returns:
		pixel_map: 64x64 grid highligting the pixel specified in the input.
	"""

	# Specify the filter size and number of channels
	filter_size = 2
	channels = 3


	with tf.variable_scope('deconv_model', reuse=tf.AUTO_REUSE):
		deconv_1 = tf.layers.conv2d_transpose(
			input_tensor, 64*channels, filter_size,
			strides=2,  activation='relu', name='deconv_1')
		deconv_2 = tf.layers.conv2d_transpose(
			deconv_1, 64*channels, filter_size,
			strides=2, activation='relu', name='deconv_2')
		deconv_3 = tf.layers.conv2d_transpose(
			deconv_2, 64*channels, filter_size,
			strides=2, activation='relu', name='deconv_3')
		deconv_4 = tf.layers.conv2d_transpose(
			deconv_3, 32*channels, filter_size,
			strides=2, activation='relu', name='deconv_4')
		deconv_5 = tf.layers.conv2d_transpose(
			deconv_4, 32*channels, filter_size,
			strides=2, activation='relu', name='deconv_5')
		deconv_6 = tf.layers.conv2d_transpose(
			deconv_5, 1, filter_size, strides=2, name='deconv_6')

	return deconv_6

def model_regression_uniform(input_tensor):
	"""
	Highlight the location of a pixel, represented in the input tensor,
	on a 64x64 map.
	Given a pixel map with a single highlighted points, regress the Cartesian
	coordinate's of that point.

	Args:
		input_tensor: Tensor representation of a pixel coordinate.
	Returns:
		coordinates: Pair of coordinates marking the pixel specified in the
			input.
	"""

	with tf.variable_scope('conv_regression', reuse=tf.AUTO_REUSE):
		conv_1 = tf.layers.conv2d(
			input_tensor, 16, 3, activation='relu', name='conv_1')
		max_pool_1 = tf.layers.max_pooling2d(
			conv_1, 2, 1, name='mp_1')
		conv_2 = tf.layers.conv2d(
			max_pool_1, 16, 3, activation='relu', name='conv_2')
		max_pool_2 = tf.layers.max_pooling2d(
			conv_2, 2, 1, name='mp_2')
		conv_3 = tf.layers.conv2d(
			max_pool_2, 16, 3, activation='relu', name='mp_2')
		max_pool_3 = tf.layers.max_pooling2d(
			conv_3, 2, 1, name='mp_3')
		conv_4 = tf.layers.conv2d(
			max_pool_3, 16, 3, activation='relu', name='conv_4')
		conv_shape = conv_4.get_shape().as_list()
		conv_vector = tf.reshape(
			conv_4, [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]])
		fc_1 = tf.contrib.layers.fully_connected(conv_vector, 64)
		fc_2 = tf.contrib.layers.fully_connected(fc_1, 2, activation_fn=None)

	return fc_2

def model_regression_quadrant(input_tensor, training=False):
	"""
	Highlight the location of a pixel, represented in the input tensor,
	on a 64x64 map.
	Given a pixel map with a single highlighted points, regress the Cartesian
	coordinate's of that point.

	Args:
	    input_tensor: Tensor representation of a pixel coordinate.
            training: (bool) Whether we are using training or not for the
                purpose of batch normalization.
	Returns:
	    global_pooling_1: Pair of coordinates marking the pixel specified in the
			input.
	"""

	with tf.variable_scope('conv_regression', reuse=tf.AUTO_REUSE):
            conv_1 = tf.layers.conv2d(
                input_tensor, 16, 5, 2, activation='relu', name='conv_1')
            conv_2 = tf.layers.conv2d(
                conv_1, 16, 1, activation='relu', name='conv_2')
            bn_1 = tf.layers.batch_normalization(
                conv_2, training=training, name='bn_1')
            conv_3 = tf.layers.conv2d(
                bn_1, 16, 3, activation='relu', name='conv_3')
            conv_4 = tf.layers.conv2d(
                conv_3, 16, 3, 2, activation='relu', name='conv_4')
            conv_5 = tf.layers.conv2d(
                conv_4, 16, 3, 2, activation='relu', name='conv_5')
            bn_2 = tf.layers.batch_normalization(
                conv_5, training=training, name='bn_2')
            conv_6 = tf.layers.conv2d(
                bn_2, 16, 3, 2, activation='relu', name='conv_6')
            conv_7 = tf.layers.conv2d(
                conv_6, 16, 1, activation='relu', name='conv_7')
            conv_8 = tf.layers.conv2d(
                conv_7, 2, 2, activation='relu', name='conv_8')
            reshaped_coordinates = tf.reshape(
                conv_8, [-1, 2])

	return reshaped_coordinates
