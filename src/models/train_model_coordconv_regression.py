
"""
Training script for CoordConv Regression on Not-So-Clevr dataset.
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
pixel_input = tf.placeholder(
    tf.float32, shape=(None, 64, 64, 1), name='pixel_input')

# Set up supervised regression using convolutions with coord convs
pixel_input_coord_conv = \
    add_coords.add_coords_layers(pixel_input)
output_map, conv_5 = supervised_conv.model_regression(pixel_input_coord_conv)

output_vector = output_map

# Loss placeholder
expected_output = tf.placeholder(
    tf.float32, shape=(None, 2), name='expected_output')

# Calculate the loss
training_loss_sq = tf.square(expected_output-output_vector)
training_loss = tf.reduce_mean(training_loss_sq)
#training_loss = tf.losses.mean_squared_error(
#    expected_output, output_vector)

# Set up accuracy calculations
# This will just be for the test set after training is complete
acc, acc_op = tf.metrics.accuracy(expected_output, output_vector)

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
    [train_loss_summary], name='train_merged_summaries')

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
            coord_batch, pixel_batch, _ = read_dataset.get_data(
                start_index, training_idx, cfg.BATCH_SIZE, 'regression')

            
            output_vector_, training_loss_sq_, summaries, _ = sess.run(
                [output_vector, training_loss_sq, train_merged_summaries, train_op],
                feed_dict={
                    pixel_input:pixel_batch,
                    expected_output:coord_batch
                })
            if j == 50:
                import pdb; pdb.set_trace()

            train_writer.add_summary(
                summaries, training_step)
            
            # Print losses to screen
            if training_step % cfg.DISPLAY_STEPS == 0:
                training_loss_ = sess.run(
                    training_loss, feed_dict={
                        pixel_input:pixel_batch,
                        expected_output:coord_batch
                        }
                    )
                print('Batch: {:7.1f}, Training Loss: {:12.7f}'.format(
                    training_step,
                    training_loss_))

            training_step += 1

            
        print('Epoch {} done'.format(i+1))

    #After training, calculate the accuracy on the test set
    total_accuracy = 0
    for i in range(num_test_batches):
        start_index = i * cfg.BATCH_SIZE
        coord_batch, pixel_batch, _ = read_dataset.get_data(
            start_index, training_idx, cfg.BATCH_SIZE, 'regression')

        acc_, acc_op_ = sess.run(
            [acc, acc_op],
            feed_dict={
                pixel_input:pixel_batch, expected_output:coord_batch}
            )
        print(acc_)


    total_accuracy = sess.run(acc)

    print('Test set accuracy: {}'.format(total_accuracy*100))