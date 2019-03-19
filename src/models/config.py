# Path to images uused during training and validation
STYLE_IMAGE_PATH = './images/starry_night.jpg'
TRAIN_IMAGES_DIR = './train2014/train2014/'
VAL_IMAGES_DIR = './val2014/val2014/'

# Seed for initializing numpy and tf
NP_SEED = 0
TF_SEED = 0

# Location of tensorboard summaries
TENSORBOARD_DIR = '../results/tensorboard_logs/'

# Path to directory used for storing images
DEBUG_DIR = '../results/debug/'

# Dimensions desired for input, channels must be kept as 3
BATCH_SIZE = 32

# Learning rate for optimizer
LEARNING_RATE = 1e-3

# Number of training and validation step
# In this instance, validation refers to when we would like to examine:
# save currently optimized image and loss
TRAINING_EPOCHS = 200
TENSORBOARD_STEPS = 10

# Offline debugging refers to images that will be saved to folder using plt,
# every validation step
DEBUG_OFFLINE = True

# This is how often training information will be printed to screen
DISPLAY_STEPS = 10

# Determines whether information is saved between runs
# for tensorboard
RESET_SAVES = True

### OVERFITTING MODE
# In this mode, we train on a smaller batch of data, treat that set as the 
# validation data, since it's unlikely the network will generalize, and 
# we also view information more frequently
OVERFITTING_MODE = False
if OVERFITTING_MODE:
	TRAINING_EPOCHS = 500
	VALIDATION_STEPS = 50
	DISPLAY_STEPS = 1
