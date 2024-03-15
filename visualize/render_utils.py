import math
import numpy as np
from scipy.stats import multivariate_normal
import cv2
import matplotlib.pyplot as plt
import os

def save_video(ims, filename, fps=30.0):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


def combine_game_heatmap(game_img, heatmap_img):
    scale = heatmap_img.shape[0] / game_img.shape[0]
    game_img = cv2.resize(game_img, (int(game_img.shape[0] * scale), int(game_img.shape[1] * scale)))
    fig = plt.figure()
    
    # combine both images to one
    max_x = max(game_img.shape[0], heatmap_img.shape[0])
    max_y = max(game_img.shape[1], heatmap_img.shape[1])

    # # pad
    game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant',
                            constant_values=0)
    heatmap_img_reshaped = np.pad(heatmap_img, ((0, max_x - heatmap_img.shape[0]), (0, 0), (0, 0)), 'constant',
                                constant_values=0)

    img = np.concatenate((game_img_reshaped, heatmap_img_reshaped), axis=1)
    return img

def stack_game_heatmap(game_img, heatmap_img):
    scale = heatmap_img.shape[0] / game_img.shape[0]
    game_img = cv2.resize(game_img, (int(game_img.shape[0] * scale), int(game_img.shape[1] * scale)))
    fig = plt.figure()
    
    # combine both images to one
    max_x = max(game_img.shape[0], heatmap_img.shape[0])
    max_y = max(game_img.shape[1], heatmap_img.shape[1])

    # # pad
    game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant',
                            constant_values=0)
    heatmap_img_reshaped = np.pad(heatmap_img, ((0, max_x - heatmap_img.shape[0]), (0, 0), (0, 0)), 'constant',
                                constant_values=0)
    img = cv2.addWeighted(game_img_reshaped, 0.5, heatmap_img_reshaped, 0.5, 0)
    return img


def plot_mog_heatmap(mean, std, pi, res=10, visibility=0.4):
    """
    Plot the 2D gaussian for mixture of gaussians
    :param mean: (np array) Mean of the 2d gaussian, Num Gaussian Mixtures x 2
    :param std: (np array) log(std) of the 2d gaussian Num Gaussian Mixtures x 2
    :param pi: (np array) Probability of each gaussian, [Num Gaussian Mixtures]
    :return:
        grid: grid of values for the 2D gaussian heatmap
    """
    # create a grid of (x,y) coordinates at which to evaluate the kernels
    # Since fugitive locations are normalized, both x, y \in {0, 1}
    xlim = (0, 1)
    ylim = (0, 1)

    # Taking Resolution as 1/10th of the env grid size
    xres = math.ceil(2428/res)
    yres = math.ceil(2428/res)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = ylim[1] - np.linspace(ylim[0], ylim[1], yres)  # Y-axis reversed
    xx, yy = np.meshgrid(x, y)

    z_accum = np.zeros(xres*yres)
    # print("plotting map")
    # print(mean, std)
    for i in range(mean.shape[0]):

        mu = mean[i]
        s = std[i]

        # Calculate covariance matrix from logstd. Assuming covariance as a diagonal matrix
        var = s ** 2
        var = np.clip(var, 0.00001, 20) 
        # print(var)
        cov = np.eye(2) * var

        k1 = multivariate_normal(mean=mu, cov=cov)

        # evaluate kernels at grid points
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        zz = k1.pdf(xxyy)

        z_accum += (pi[i] + visibility) * zz

    # reshape and plot image
    grid = z_accum.reshape((xres, yres))
    return grid

def plot_gaussian_heatmap(mu, s, res=1):
    """
    Plot the 2D gaussian for one gaussian
    :param mean: (np array) Mean of the 2d gaussian, Num Gaussian Mixtures x 2
    :param std: (np array) log(std) of the 2d gaussian Num Gaussian Mixtures x 2
    :param pi: (np array) Probability of each gaussian, [Num Gaussian Mixtures]
    :return:
        grid: grid of values for the 2D gaussian heatmap
    """
    # create a grid of (x,y) coordinates at which to evaluate the kernels
    # Since fugitive locations are normalized, both x, y \in {0, 1}
    xlim = (0, 1)
    ylim = (0, 1)

    # Taking Resolution as 1/10th of the env grid size
    xres = math.ceil(2428/10)
    yres = math.ceil(2428/10)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = ylim[1] - np.linspace(ylim[0], ylim[1], yres)  # Y-axis reversed
    xx, yy = np.meshgrid(x, y)

    # z_accum = np.zeros(xres*yres)
    # print("plotting map")
    # print(mean, std)
    
    # Calculate covariance matrix from logstd. Assuming covariance as a diagonal matrix
    var = s ** 2
    var = np.clip(var, 0.00001, 20) 
    # print(var)
    cov = np.eye(2) * var

    k1 = multivariate_normal(mean=mu, cov=cov)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy)

    # z_accum += pi[i] * zz

    # reshape and plot image
    grid = zz.reshape((xres, yres))
    return grid
