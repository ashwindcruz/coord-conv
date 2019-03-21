# Path to folder
DIR_PATH = 'C:/Users/Ashwin/Documents/Projects/coord-conv/src/'

# Seed for initializing numpy and tf
NP_SEED = 0
TF_SEED = 0

# Location of tensorboard summaries
TENSORBOARD_DIR = '../results/tensorboard_logs/'

# Path to directory used for storing images
DEBUG_DIR = '../results/debug/'

# Type of split to apply to dataset
SPLIT = 'uniform'

# Dimensions desired for input, channels must be kept as 3
BATCH_SIZE = 32

# Learning rate for optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

# Number of training and validation step
# In this instance, validation refers to when we would like to examine:
# save currently optimized image and loss
TRAINING_EPOCHS = 100
TENSORBOARD_STEPS = 10

# Offline debugging refers to images that will be saved to folder using plt,
# every validation step
DEBUG_OFFLINE = True

# This is how often training information will be printed to screen
DISPLAY_STEPS = 10

# Determines whether information is saved between runs
# for tensorboard
RESET_SAVES = True