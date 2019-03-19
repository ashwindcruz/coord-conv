import tensorflow as tf

def model(input_tensor):
	"""
	Highlight the location of a pixel, represented in the input tensor, 
	on a 64x64 map. 
	Use deconvolutions (aka transposed convolutions) to achieve this. 

	Args:
		input_tensor: Tensor representation of a pixel coordinate. 
	Returns:
		pixel_map: 64x64 grid highligting the pixel specified in the input. 
	"""

	# Specify the filter size and number of channels
	filter_size = 2
	channels = 2


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
			deconv_5, 1, filter_size, 
			strides=2, name='deconv_6')

	return deconv_6
