import time

import math
import numpy as np
from scipy import interpolate, spatial


# Parameter used in the nearest neighbour calculation
neighbourhoodsize = 6000

def auto_canny(image, sigma=5.33):
    """
    Apply canny edge detection.

    Args:
        image: Original input image.
        sigma: 
    Returns:
        edged: Array with edges being high-valued pixels, and all other pixels
            being zero.
    """

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def nearest(nearestTo, pointsList):
    """
    Finds the nearest points based on a neighbourhood size parameter.

    Args:
        nearestTo: Central point.
        pointsList: List of points that might be potential neighbours.
    Returns:
        nearest_neighbours (implicit): Nearest points to the central point.
    """
    # Find the nearest points
    # This method returns a list of neigbouring points given a point and a list of all points

    # The variable neighbourhoodsize is taken from the beginning
    tree = spatial.cKDTree(pointsList)
    pointsIndex = tree.query_ball_point(nearestTo, neighbourhoodsize)
    return pointsList[pointsIndex]

def removeItem(source_array, points_to_remove):
    """
    Remove one set of points from another.
    Args:
        source_array: Original array from which points will be removed.
        points_to_remove: Points to remove from the source_array.
    Returns:
        source_array (implicit): The source array without points_to_remove.
    Obtained from: https://stackoverflow.com/a/40056251
    """
    cumdims = (np.maximum(source_array.max(),points_to_remove.max())+1)**np.arange(points_to_remove.shape[1])
    return source_array[~np.in1d(source_array.dot(cumdims), points_to_remove.dot(cumdims))]


# TODO:
# This is suppoed to be faster than the removeItem we are currently using
# Try to get it working in the future
def removeItemBroken(A,B):
    dims = np.maximum(B.max(0),A.max(0))+1
    return A[~np.in1d(np.ravel_multi_index(A.T,dims),\
        np.ravel_multi_index(B.T,dims))]

def euclideanDistance(ref, coord):
    """
    Calculate the euclidean distance between two points.

    Args:
        ref: First point.
        coord: Second point.
    Returns:
        d: Euclidean distance between ref and coord.
    """
    x0,y0 = ref
    x,y = coord
    d = math.sqrt((x-x0)**2+(y-y0)**2)

    return d

minimum_line_segment_length_in_number_of_points = 5

def pointListToLines(edge_points):
    """
    Convert an array of edge points to a collection of lines.
    This involves joining up edge points in lines.

    Args:
        edge_points: Array of edge points.
    Returns:
        line_segments: List of lists. Each of the inner list items is a collection of points. 
    """

    # L is an ndarray that will contain all unique points
    L = edge_points
    L = np.array(edge_points)

    # total_num_points is the total number of points
    # To obtain this number, we just count the 0 dimension of the np array
    total_num_points = L.shape[0]
    print('There are ' + str(total_num_points) +' points in total.')

    # List of lists consisting of the line segments
    # This is what we are calculating
    list_of_line_segments = []

    # Create a numpy array that will contain all already visited points
    # The array is two-dimensional (since we have x and y values)
    # As a work-around we add the point (0,0) to obtain a non-empty array (required to concatenate with another array)
    visited_points = np.array([[0,0]])

    # List of neighbouring points
    neighbouring_points = []

    # List of neighbouring points where points have been removed that were already visited
    unvisited_neighbouring_points = []

    # Iteration counter
    # Careful, the counter is incremented at multiple locations in the code
    # Hence, the pointer jumps and does not go through all values
    counter = 0

    # Flag to indicate if we have tried reversing yet
    could_reverse_direction = True

    loop_counter = 0

    # Please note that the counter does not go through all values since it is incremented at multiple locations
    # total_num_pointsbda "total_num_points" stays unchanged
    start_time = time.time()
    while counter < total_num_points:

        # Just for counting the loops
        loop_counter += 1
        # TODO: This prints too many times, have it print in place
        print("Loop Counter: " + str(loop_counter))
        print("Total Number of Points: " + str(total_num_points))

        # Suppress divide by zero errors thrown by numpy (divide by zero or NaN)
        with np.errstate(divide='ignore', invalid='ignore'):

            try:

                #*****************************************
                # Time tracker
                elapsed_time = time.time() - start_time
                print("Loop Counter: " + str(loop_counter))
                print('Time 3 - start: ' + str(round(elapsed_time, 4))
                    + " (Processed " + str(round((counter/total_num_points) * 100, 2)) + "% of points)")
                #*****************************************

                # segmented_line_bar = {RandomChoice[DeleteCases[L, _?visited_points]]};

                # Remove from L all the points that we have seen already
                unvisited_points = removeItem(L, visited_points)

                # We select a random starting point out of all the points in the reduced array
                # TODO: Find out what the type of segmented_line_bar is
                segmented_line_bar = L[np.random.choice(unvisited_points.shape[0], 1, replace=False)]

                # The just selected starting point is now added to the list of seen points
                visited_points = np.concatenate((visited_points, segmented_line_bar))

                # In order to be able to concatenate above, the visited_points array needed to be non-empty
                # We achieved that by having added the (0,0) point prior to the concatenation
                # We need to remove that point again since we did not actually visit it
                # TODO: Is this point being removed multiple times?
                if counter == 0:
                    visited_points = removeItem(visited_points, np.array([[0,0]]))

                # Increment the counter
                counter = counter + 1

                # Set the couldReverse flag to true
                could_reverse_direction = True

                # Obtain a list of neighbouring points...
                # ...given the neighbourhood size defined before
                # ...given the starting point and the list of all points (minus the starting point)
                #TODO: If segmented_line_bar is a single point, should it be stored as such instead of in a list?
                #neighbouring_points = nearest(segmented_line_bar[0], L)
                neighbouring_points = nearest(segmented_line_bar[0], unvisited_points)
                import pdb; pdb.set_trace() 
                
                
                #*****************************************
                # Time tracker
                elapsed_time = time.time() - start_time
                print('Time 3 - Find starting point and its nearest neighbours: ' + str(round(elapsed_time, 4))
                    + " (Processed " + str(round((counter/total_num_points * 100), 2)) + "% of points)")
                #*****************************************

                # Remove from the list of neighbouring points all the seen points
                # TODO: Is this encoding the belief that a point can't be part of two line segments
                # Is this valid to believe?
                unvisited_neighbouring_points = removeItem(neighbouring_points, visited_points)

                # So far, we have only obtained a list of neighbours
                # But we do not know their distance from our focal point
                # We now sort the list by their distance from the focal point
                # We use the euclidian distance
                unvisited_neighbouring_points.tolist().sort(
                    key=lambda x: euclideanDistance(segmented_line_bar[0], x))

                # We convert the sorted list of neighbours into a Numpy array
                unvisited_neighbouring_points = np.asarray(unvisited_neighbouring_points)

                # We create a placeholder (numpy array) to store the next point
                next_point = np.array([])

                #*****************************************
                # Time tracker
                elapsed_time = time.time() - start_time
                print('Time 3 - Sort nearest neighbours: ' + str(round(elapsed_time, 4))
                    + " (Processed " + str(round((counter/total_num_points * 100), 2)) + "% of points)")
                #*****************************************

                # Now we loop through all the neighbours
                # As long as there are neighbours, loop
                while unvisited_neighbouring_points.shape[0] > 0:
                   

                    # TODO: Where does the 3 come from? Is that a parameter?
                    # If the length of the segmented line bar is smaller or equal to 3:
                    # If you have a very small cluster of points, i.e. segmented line bar is small, 
                    # don't try finding the mid point, just add the point that's closest
                    # to the starting point
                    if segmented_line_bar.shape[0] <= 3:
                        next_point = unvisited_neighbouring_points[0]

                    else:
                        d_numerator = (segmented_line_bar[-1]-segmented_line_bar[-2]) \
                            + 0.5*(segmented_line_bar[-2]-segmented_line_bar[-3])


                        # Normalise
                        d = d_numerator/np.linalg.norm(d_numerator)
                        

                        # flipud switches around x and y of the coord
                        # can't understand why this is being calculated
                        n = np.array([-1,1])*np.flipud(d)

                        # Ash
                        # It looks like this calculates the distance of each points in
                        # unvisited_neighbouring_points to the 'midpoint' calculated above which
                        # seems to be similar to calculating the distance to the points in
                        # segmented_line_bar
                        manhattan_distance = unvisited_neighbouring_points - segmented_line_bar[-1]

                        distance = np.sqrt(
                            np.dot(manhattan_distance, d)**2 \
                            + 2*np.dot(manhattan_distance, n)**2  
                            )
                        next_point = unvisited_neighbouring_points[np.argmin(distance)]
                        



                    segmented_line_bar = np.vstack((segmented_line_bar, next_point))
                    visited_points = np.vstack((visited_points, next_point))

                    counter = counter + 1

                    # ...
                    neighbouring_points = nearest(segmented_line_bar[-1], L)

                    # Remove all the seen points from the nearest list
                    unvisited_neighbouring_points = removeItem(neighbouring_points, visited_points)
                    

                list_of_line_segments.append(segmented_line_bar.tolist())

                #*****************************************
                # Time tracker
                elapsed_time = time.time() - start_time
                print('Time 3 - end: ' + str(round(elapsed_time, 4)) + " (Processed " + str(round((loop_counter/total_num_points * 100), 2)) + "% of points)")
                #*****************************************

            # TODO: Do not catch everything, be specific
            except Exception as e:

                print(e)

    # Sort the lines in ascending order based on length of segments
    # And: Remove any segment which is less than 12 points
    line_segments = [line for line in sorted(list_of_line_segments, key=len)[::-1] if len(line) > minimum_line_segment_length_in_number_of_points]

    return line_segments

def FourierSeriesApprox(x_fourier, y_fourier, t, order=50):
    """
    Approximate the Fourier series for a set of given input.

    Args:
        x_fourier:
        y_fourier:
        t:
        order:
    Returns: 
        fourier_approximation (implicit): 

    """
    # Function to approximate the fourier series

    cax = np.real(x_fourier)
    cay = np.imag(x_fourier)
    yax = np.real(y_fourier)
    yay = np.imag(y_fourier)
    n = order
    length = min(n, len(cax)) # Number of series to approximate (Maxorder is 180)
    x_appr = cax[:length]*np.cos(t*np.arange(0,length))+cay[:length]*np.sin(t*np.arange(0,length))
    y_appr = yax[:length]*np.cos(t*np.arange(0,length))+yay[:length]*np.sin(t*np.arange(0,length))
    x_appr[0] = x_appr[0]/2 # If[K==0,1,2]
    y_appr[0] = y_appr[0]/2
    return (np.sum(x_appr),np.sum(y_appr))

