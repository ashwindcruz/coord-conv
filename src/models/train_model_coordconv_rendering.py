
"""
Training script for CoordConv Rendering on Not-So-Clevr dataset.
"""

import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf

import config as cfg
sys.path.insert(0, cfg.DIR_PATH)
import add_coords, supervised_conv
from data import read_dataset

# Set TF debugging to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if cfg.RESET_SAVES is True:
    # Ensure tensorboard is not running when you try to delete
    # this directory
    if os.path.exists(cfg.TENSORBOARD_DIR):
        shutil.rmtree(cfg.TENSORBOARD_DIR)

# Set the seeds to provide consistency between runs
# Can also comment out for variability between runs
np.random.seed(cfg.NP_SEED)
tf.set_random_seed(cfg.TF_SEED)

# Coordinates placeholder
coordinates_input = tf.placeholder(
    tf.float32, shape=(None, 64, 64, 2), name='coordinates_input')

# Carry out the rendering
coordinates_input_coord_conv = \
    add_coords.add_coords_layers(coordinates_input)
classification_map = supervised_conv.model_classification(
    coordinates_input_coord_conv)
classification_vector = tf.reshape(classification_map, [-1, 4096])
classification_vector_softmax = tf.contrib.layers.softmax(classification_vector)
classification_pixel_map = tf.reshape(
    classification_vector_softmax, [-1, 64, 64, 1])
output_map = supervised_conv.model_rendering(classification_pixel_map)

# Reshaping required for sigmoid cross entropy
output_vector = tf.reshape(output_map, [-1, 4096])

# Loss placeholder
expected_output = tf.placeholder(
    tf.float32, shape=(None, 4096), name='expected_output')

# Calculate the loss
training_loss = tf.losses.sigmoid_cross_entropy(expected_output, output_vector)

# Set up the final loss, optimizer, and summaries
optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)
train_op = optimizer.minimize(training_loss)
    
init_op = tf.group(
    tf.global_variables_initializer(), 
    tf.local_variables_initializer(),
    name='initialize_all')

# Tensorboard summaries
train_loss_summary = tf.summary.scalar('train_loss', training_loss)
train_merged_summaries = tf.summary.merge(
    [
        train_loss_summary
    ],
    name='train_merged_summaries')

# Set up images for tensorboard
output_vector_sigmoid = tf.nn.sigmoid(output_vector)
images_output = tf.reshape(output_vector_sigmoid, [-1, 64, 64, 1])
images_summary = tf.summary.image('output_images', images_output)
images_test_summary = tf.summary.image('output_images_test', images_output)


# Training
with tf.Session() as sess:
    # Initialize all variables and then
    # restore weights for feature extractor
    sess.run(init_op)

    # Set up summary writer for tensorboard, saving graph as well
    train_writer = tf.summary.FileWriter(
        cfg.TENSORBOARD_DIR, sess.graph)

    # Get training and testing indices
    training_idx, testing_idx = read_dataset.get_indices(cfg.SPLIT)

    # Set aside some images for tensorboard
    first_idx = np.random.permutation(len(training_idx))[0]
    tensorboard_batch, _, _ = read_dataset.get_data(
        first_idx, training_idx, cfg.BATCH_SIZE, 'coordconv')

    first_idx = np.random.permutation(len(testing_idx))[0]
    tensorboard_test_batch, _, _ = read_dataset.get_data(
        first_idx, testing_idx, cfg.BATCH_SIZE, 'coordconv')

    # Begin training
    num_train_batches = read_dataset.num_training_examples // cfg.BATCH_SIZE
    num_test_batches = read_dataset.num_testing_examples // cfg.BATCH_SIZE

    training_step = 0
    for i in range(cfg.TRAINING_EPOCHS):
        # Shuffle the training indices for every epoch
        training_idx = np.random.permutation(training_idx)
        # Go through one pass of the training data
        for j in range(num_train_batches):


            start_index = j * cfg.BATCH_SIZE
            # coord_batch, pixel_batch, _  = read_dataset.get_data(
            coord_batch, _, image_square_batch = read_dataset.get_data(
                start_index, training_idx, cfg.BATCH_SIZE, 'coordconv')

            summaries, _ = sess.run(
                [train_merged_summaries, train_op],
                feed_dict={
                    coordinates_input:coord_batch,
                    expected_output:image_square_batch
                })

            train_writer.add_summary(
                summaries, training_step)
            

            # Print losses to screen
            if training_step % cfg.DISPLAY_STEPS == 0:
                training_loss_ = sess.run(
                    training_loss, feed_dict={
                        coordinates_input:coord_batch,
                        expected_output:image_square_batch
                        }
                    )
                print('Batch: {:7.1f}, Training Loss: {:12.7f}'.format(
                    training_step,
                    training_loss_))

            # Save image summary to tensorboard
            if training_step % cfg.TENSORBOARD_STEPS == 0:
                images_summary_ = sess.run(
                    images_summary, 
                    feed_dict={coordinates_input:tensorboard_batch}
                    )

                train_writer.add_summary(images_summary_, training_step)

            training_step += 1

            
        print('Epoch {} done'.format(i+1))

    # After training is done, check how well the model renders test images
    images_summary_ = sess.run(
                    images_test_summary, 
                    feed_dict={coordinates_input:tensorboard_test_batch}
                    )

    train_writer.add_summary(images_summary_, training_step)
