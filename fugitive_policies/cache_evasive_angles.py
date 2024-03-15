import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
from simulator.load_environment import load_environment
# import skimage.measure
import torch
import cv2
from numpy.lib.stride_tricks import as_strided
# import heatmap seaborn
import seaborn as sns
import matplotlib.pyplot as plt


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

def calculate_desired_heading(start_location, end_location):
    return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])

def calculate_evasive_angle(current_location, terrain):
    """
    This function will look at the fugitives current location, hideouts, and terrain, and choose a direction to go
    to evade detection (into the forest)
    :param current_location:
    :param hideouts:
    :return:
    """
    # find locations where forest is pretty dense
    dense_forest = terrain.forest_density_array < .4

    # check around some fixed region of the fugitive
    ran = 250
    lb_x = max(int(current_location[0] - ran), 0) # lower bound x
    ub_x = min(int(current_location[0] + ran), terrain.dim_x) # upper bound x
    lb_y = max(int(current_location[1] - ran), 0) # lower bound y
    ub_y = min(int(current_location[1] + ran), terrain.dim_y) # upper bound y
    best_dist = np.inf
    candidate = None
    for i in range(lb_x, ub_x):
        for j in range(lb_y, ub_y):
            if i == current_location[0] and j== current_location[1]:
                continue
            s = (i, j)
            dist = np.linalg.norm(s - current_location)
            # if its a patch of forest and not a one off
            if np.sum(dense_forest[i - 3:i + 3, j - 3:j + 3]) < 18:
                continue
            if dist <= best_dist:
                best_dist = dist
                candidate = s

    if candidate is None:
        candidate = (1500, 1500)
    
    # angle = calculate_desired_heading(current_location, candidate)
    # return angle
    return candidate

def calculate_evasive_angle_fast(current_location, nonzero):
    target = np.array(current_location)
    distanced = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distanced)

    location = nonzero[nearest_index]
    manhattan_dist = np.abs(location - target)

    if manhattan_dist[0] > 250 or manhattan_dist[1] > 250:
        candidate = (1500, 1500)
    else:
        candidate = (nonzero[nearest_index] + 3)

    angle = calculate_desired_heading(current_location, candidate)
    return angle
    
def initialize_evasive_array(terrain):
    array = terrain
    array = (array < 0.4) * 1.0 # turn to float

    array_pool = pool2d(array, 6, 1, pool_mode='avg') * 36
    array_thresh = (array_pool > 2) * 255

    nonzero = np.where(array_thresh > 0)
    nonzero = np.stack(nonzero, axis=-1)

    return nonzero

if __name__ == "__main__":
    env = load_environment('simulator/configs/balance_game.yaml')
    array = env.terrain.forest_density_array
    array = (array < 0.4) * 1.0 # turn to float

    array_pool = pool2d(array, 6, 1, pool_mode='avg') * 36
    array_thresh = (array_pool > 2) * 255

    nonzero = np.where(array_thresh > 0)
    nonzero = np.stack(nonzero, axis=-1)

    start = time.time()
    for x in range(0, 2428, 100):
        for y in range(0, 2428, 100):
            # new_loc = calculate_evasive_angle(np.array((x, y)), env.terrain)
            # new_loc = calculate_evasive_angle_fast(np.array((x, y)), nonzero)
            print((x, y), new_loc)

    print(time.time() - start)