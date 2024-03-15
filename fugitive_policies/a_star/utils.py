import math
import png
import numpy
import matplotlib.pyplot as plt


def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)


def png_to_ogm(filename, normalized=False, origin='lower'):
    """
    Convert a png image to occupancy data.
    :param filename: the image filename
    :param normalized: whether the data should be normalised, i.e. to be in value range [0, 1]
    :param origin:
    :return:
    """
    r = png.Reader(filename)
    img = r.read()
    img_data = list(img[2])

    out_img = []
    bitdepth = img[3]['bitdepth']

    for i in range(len(img_data)):

        out_img_row = []

        for j in range(len(img_data[0])):
            if j % img[3]['planes'] == 0:
                if normalized:
                    out_img_row.append(img_data[i][j]*1.0/(2**bitdepth))
                else:
                    out_img_row.append(img_data[i][j])

        out_img.append(out_img_row)

    if origin == 'lower':
        out_img.reverse()

    return out_img

def terrain_map_mountain_to_ogm(map_array, mountain_locations):
    pass


def plot_path(path, hideout_locs, point_period=4):
    """ This function is used to plot single selected path with shape (traj_num, traj_pt_num, coord_dim_num) """
    for hideout_id, hideout_loc in enumerate(hideout_locs):
        if hideout_id < 1:
            plt.plot(hideout_loc[0], hideout_loc[1], 'y*', markersize=20)
        elif hideout_id < 3:
            plt.plot(hideout_loc[0], hideout_loc[1], 'b*', markersize=20)
        else:
            raise NotImplementedError
    start_x, start_y = path[0]
    goal_x, goal_y = path[-1]

    # plot path
    path_arr = numpy.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], 'y')
    plt.scatter(path_arr[::point_period, 0], path_arr[::point_period, 1], c='r')

    # plot start point
    plt.plot(start_x, start_y, 'ro')

    # plot goal point
    plt.plot(goal_x, goal_y, 'go')

    plt.xlim([0, 2428])
    plt.ylim([0, 2428])
    plt.show()

def plot_multiple_paths(paths, hideout_locs):
    paths = paths[...,0:2]
    """ This function is used to plot multiple paths with shape (traj_num, traj_pt_num, coord_dim_num) """
    for hideout_id, hideout_loc in enumerate(hideout_locs):
        if hideout_id < 1:
            plt.plot(hideout_loc[0], hideout_loc[1], 'y*', markersize=20)
        elif hideout_id < 3:
            plt.plot(hideout_loc[0], hideout_loc[1], 'b*', markersize=20)
        else:
            raise NotImplementedError
    for path in paths:
        start_x, start_y = path[0]
        goal_x, goal_y = path[-1]

        # plot path
        path_arr = numpy.array(path)
        plt.plot(path_arr[:, 0], path_arr[:, 1])

        # plot start point
        plt.plot(start_x, start_y, 'ro')

        # plot goal point
        plt.plot(goal_x, goal_y, 'go')

    plt.xlim([0, 2428])
    plt.ylim([0, 2428])
    plt.show()    

def plot_both_paths(paths, hideout_locs, pt_num):
    """ This function is used to plot multiple paths with shape (traj_num, traj_pt_num, coord_dim_num) """
    for hideout_id, hideout_loc in enumerate(hideout_locs):
        if hideout_id < 1:
            plt.plot(hideout_loc[0], hideout_loc[1], 'y*', markersize=20)
        elif hideout_id < 3:
            plt.plot(hideout_loc[0], hideout_loc[1], 'b*', markersize=20)
        else:
            raise NotImplementedError
    for path in paths:
        start_x, start_y = path[0][:2]
        goal_x, goal_y = path[-1][:2]

        # plot path
        path_arr = numpy.array(path)
        # red
        plt.plot(path_arr[:, 0], path_arr[:, 1], 'r')
        # blue
        # blue start
        heli_start_x, heli_start_y = path[0][2:4]
        sp_start_x, sp_start_y = path[0][4:]

        plt.plot(path_arr[:, 2], path_arr[:, 3], 'b')
        plt.plot(path_arr[:, 4], path_arr[:, 5], 'c')

        # plot start point
        plt.plot(start_x, start_y, 'ro')
        plt.plot(heli_start_x, heli_start_y, 'bo')
        plt.plot(sp_start_x, sp_start_y, 'co')

        # plot goal point
        plt.plot(goal_x, goal_y, 'go')

        # plot first pt_num points
        if pt_num is not None:
            plt.scatter(path_arr[:pt_num, 0], path_arr[:pt_num, 1], c='r', alpha=1, s=10)
            plt.scatter(path_arr[:pt_num, 2], path_arr[:pt_num, 3], c='b', alpha=1, s=10)
            plt.scatter(path_arr[:pt_num, 4], path_arr[:pt_num, 5], c='c', alpha=1, s=10)

    plt.xlim([0, 2428])
    plt.ylim([0, 2428])
    # plt.show()    