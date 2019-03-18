import os
import sys
import time

import cv2
import cmath
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import interpolate
import svgwrite

dir_path = os.path.dirname(os.path.realpath(__file__))
# The following will fix paths on a Windows OS
# Doesn't affect other OSs
dir_path = dir_path.replace('\\', '/') 
folders_along_path = dir_path.split('/')
project_index = folders_along_path.index('fourier-line-art')
base_path = '/'.join(folders_along_path[:project_index+1])

# This allows us to source local modules
sys.path.insert(0, base_path)

from src.features import build_features

results_folder = base_path + '/results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Select an input image to work on
icon_path = base_path + '/data/'
file_name = "deer2"
file_extension = ".png"
img_path = icon_path + file_name + file_extension

if os.path.isfile(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED # cv2.IMREAD_GRAYSCALE

else:
    print ("The file " + img_path + " does not exist.")

if len(image.shape) == 3:
    original_height, original_width, original_depth = image.shape
    print("Current width (before any changes): " + str(original_width))
    print("Current height (before any changes): " + str(original_height))
    print("Current depth (before any changes): " + str(original_depth))
elif len(image.shape) == 2:
    original_height, original_width = image.shape
    print("Current width (before any changes): " + str(original_width))
    print("Current height (before any changes): " + str(original_height))

# Reduce the image size if it's too big
# TODO: Use the number of pixels instead of the width
max_width = 800

if original_width > max_width:

    img_scale = max_width/original_width

    print("Scaling factor: " + str(max_width) + "/"
        + str(original_width) + "=" + str(img_scale))

    new_x, new_y = image.shape[1]*img_scale, image.shape[0]*img_scale
    image = cv2.resize(image,(int(new_x),int(new_y)))

if len(image.shape) == 3:
    new_height, new_width, new_depth = image.shape
    print("Current width (after): " + str(new_width))
    print("Current height (after): " + str(new_height))
    print("Current depth (after): " + str(new_depth))
elif len(image.shape) == 2:
    new_height, new_width = image.shape
    print("Current width (after): " + str(new_width))
    print("Current height (after): " + str(new_height))

# Load numpy array into Pillow as image
im = Image.fromarray(image)
contrast = ImageEnhance.Contrast(im)

# Increase contrast of image (1.0 = leave as is; 0 = all grey)
# im = contrast.enhance(2)

# Image filter
# TODO: The option should be in a config file instead of comments
# https://pillow.readthedocs.io/en/3.1.x/reference/ImageFilter.html
# im = im.filter(ImageFilter.CONTOUR)
# im = im.filter(ImageFilter.EDGE_ENHANCE)
# im = im.filter(ImageFilter.EMBOSS)
# im = im.filter(ImageFilter.BLUR)
# im = im.filter(ImageFilter.SMOOTH)
im = im.filter(ImageFilter.SMOOTH_MORE)
# im = im.filter(ImageFilter.FIND_EDGES) # like Canny edge detection

image = np.asarray(im)

# Carry out edge detection
# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
# TODO: Different options should be in a config file
blurred = cv2.GaussianBlur(image, (3, 3), 0)
#wide = cv2.Canny(blurred, 10, 200)
#tight = cv2.Canny(blurred, 225, 250)
new = cv2.Canny(blurred, 120, 120)
#auto = auto_canny(blurred)
image_to_be_used = new

# Save canny image
canny_file_name = base_path + "/results/" + file_name + "_canny" + ".png"
cv2.imwrite(canny_file_name, image_to_be_used)

# Store all edge points (i.e. non-zero output from Canny edge detector)
# TODO: There is prolly a numpy function to do this
edge_points = []

total_number_of_pixels_to_be_processed = image_to_be_used.shape[0] * image_to_be_used.shape[1]
print("Total number of pixels to be processed: " + str(total_number_of_pixels_to_be_processed))

for y in range(0, image_to_be_used.shape[0]):

    # Progress report
    if (y % 100 == 0):
        print("Processing " + str(y) + " out of " + str(image_to_be_used.shape[0]) + " (y values)")

    for x in range(0, image_to_be_used.shape[1]):
        if image_to_be_used[y, x] != 0:
            edge_points = edge_points + [[x, -y]]

print("Done reading pixels... We have " + str(len(edge_points)) + " points.")

# Convert list into a two-dimensional np array
edge_points = np.array(edge_points)

print("Done storing " + str(len(edge_points)) + " edge points in a NumPy array.")
print("The dimension of the edge_points array is: " + str(edge_points.shape))

print("A selection of the first 3 edge points looks like this: \n" + str(edge_points[0:3, :]))

# Potentially use a subset of edge points rather than the entire set
"""
Placeholder: Idea: Maybe we can reduce the edgePoint array here
by clustering pixels that are close to each other and replacing them
by a single one
"""
# TODO: Sampling parameter should be in a config file
sampling = False
proportion_to_keep_in_percent = 90 # if we sample

len_edge_points_old = len(edge_points)

# print(type(edge_points)) # numpy array

if sampling:

    # edge_points is a Numpy array and needs to be converted into a list first before we can sample from it
    edge_points = random.sample(edge_points.tolist(), round(len(edge_points) * proportion_to_keep_in_percent // 100, 1))

    # print(type(edge_points)) # list

    # Now, we have to convert it back into a Numpy array

    edge_points = np.asarray(edge_points)

    # print(type(edge_points)) # nunmpy array

    print("Done sampling from " + str(len_edge_points_old) + " edge points down to " + str(len(edge_points)) + ".")



### Converting edge points into line segments (this is the problematic step)
# Convert edge points into line segments
# WARNING: This takes a very long time
#This process takes the longest... About 2-10 seconds per point that needs to be processed.
# TODO: Replace current inaccurate percentage counter with a correct
# one (e.g. progressbar or https://github.com/tqdm/tqdm)

line_segments = build_features.pointListToLines(edge_points)

# Use the line segments as input to the Fourier Series approximation
# The higher, the more accurate
# Reasonable values are roughly between 30 and 180
approximation_accuracy = 300
print("Number of line segements:", len(line_segments))
all_list=[]

list_of_list_of_tuples_to_be_exported = []

for line in line_segments:

    # TODO: These two lines can likely be replaced with a better
    # for loop above
    x=[x[0] for x in line]
    y=[y[1] for y in line]

    # Make them closed paths To do: How to handle open paths?
    x = np.append(x,x[0])
    y = np.append(y,y[0])

    # TODO: Figure out what epsilon is and what it is used for
    eps = 10**-3

    # Set mu = the total number of points to represent
    # mu = 2^14
    mu = 2**14

    tck,u = interpolate.splprep([x,y],k=2,s=0)
    u = np.linspace(0,1,num = mu,endpoint = True)

    # Results of the spine interpolated points
    spline_interpolated = interpolate.splev(u,tck)

    # https://stackoverflow.com/questions/8680909/fft-in-matlab-and-numpy-scipy-give-different-results
    x_fourier = np.fft.fft(spline_interpolated[0]).T
    y_fourier = np.fft.fft(spline_interpolated[1]).T
    #x_fourier=(2*math.pi/math.sqrt(mu))*x_fourier
    #y_fourier=(2*math.pi/math.sqrt(mu))*y_fourier

    for inx,val in enumerate(x_fourier):
        x_fourier[inx] = (2*math.pi/math.sqrt(mu))*(np.real(cmath.exp((inx+1)*math.pi*complex(0,1)))*val)

    for inx,val in enumerate(y_fourier):
        y_fourier[inx] = (2*math.pi/math.sqrt(mu))*(np.real(cmath.exp((inx+1)*math.pi*complex(0,1)))*val)

    # This is the maximum number that can be used for approximation
    max_order = 180

    x_fourier = x_fourier[:max_order+1]
    y_fourier = y_fourier[:max_order+1]

    points_space = np.linspace(-np.pi, np.pi, 1000)

    approximation_value_we_add_per_figure = approximation_accuracy / 5


    approx_param = int(0.5*approximation_value_we_add_per_figure)
    x=[];
    for i in range(len(points_space)):
        x.append(tuple([-x for x in build_features.FourierSeriesApprox(x_fourier, y_fourier,points_space[i],approx_param)]))#Hack to invert, multiply by -1
    # Here, we export the tuples - BUT ONLY FOR THIS APPROX.
    list_of_list_of_tuples_to_be_exported.append(x)

"""
If the lines start to trace individual pixels
(step-like appearance), the degree of approximation was set
too high for a good output.

Placeholder / idea: Optimise the paths by removing unnecessary points
without changing the shape too much
"""

# Generate and display SVG

# Create a drawing
dwg = svgwrite.Drawing(base_path + '/results/' + file_name + '.svg', profile='tiny')

# Find the maximum x and y value and the minimum x and y value
# Hoping that we can transform the whole thing somehow

tuple_array = np.array(list_of_list_of_tuples_to_be_exported)

max_x = np.max(tuple_array[:, :, 0])
max_y = -1 * np.min(tuple_array[:, :, 1])
min_x = np.min(tuple_array[:, :, 0])
min_y = -1 * np.max(tuple_array[:, :, 1])

max_desired_width = 300

# Each list of tuples corresponds to one contour
for list_of_tuples in list_of_list_of_tuples_to_be_exported:

    # print(len(list_of_tuples))
    # Why are all lists exactly a thousand elements large?

    new_list_of_tuples = []

    for tuple_item in list_of_tuples:

        # We need to flip the image (negate the y coordinates)
        new_tuple = (tuple_item[0], (-1)*tuple_item[1])
        new_list_of_tuples.append(new_tuple)


#         print(new_list_of_tuples)
#         print('\n')
#         print('----------------------')

    # Normalise tuples

    list_of_normalised_tuples = []

    for tuple_item in new_list_of_tuples:

        additive_x = -1 * min_x
        additive_y = -1 * min_y

        factor_x = max_desired_width / max_x
        factor_y = factor_x

        new_x_value = (tuple_item[0] + additive_x) * factor_x
        new_y_value = (tuple_item[1] + additive_y) * factor_y

        new_tuple = (new_x_value, new_y_value)
        list_of_normalised_tuples.append(new_tuple)


    polyline_obj = svgwrite.shapes.Polyline(list_of_normalised_tuples, stroke='black', stroke_width=1, fill="none")

    dwg.add(polyline_obj)

# dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))


print("max_x: " + str(max_x))
print("min_x: " + str(min_x))
print("max_y: " + str(max_y))
print("min_y: " + str(min_y))

dwg.save()

#display(SVG(filename = base_path + '/results/' + file_name + '.svg'))

print('done')
