import numpy as np

import matplotlib.pyplot as plt

data_path = f"/workspace/PrisonerEscape/datasets/test/small_subset/seed_{500}_known_44_unknown_33.npz"
data = np.load(data_path, allow_pickle=True)

red_locations = data['red_locations']

plt.figure()
for loc in red_locations:
    plt.scatter(loc[0], loc[1])

plt.savefig("visualize/test.png")
def plot_trajectory_occupancy_map(save_path, red_locs):

    known_hideout_locations = np.array([[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624],
                               [234, 171], [2398, 434], [633, 2136], [1590, 2]])
    unknown_hideout_locations = np.array([[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883],
                                          [1077, 1445], [602, 372], [80, 2274], [279, 477]])

    x = red_locs[:, 0] * 2428
    y = red_locs[:, 1] * 2428

    xy, z = np.unique(list(zip(x, y)), axis=0, return_counts=True)
    x = xy[:, 0]
    y = xy[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c='tab:blue', alpha=0.1, s=z * 2)

    k_hideout_x = known_hideout_locations[:, 0]
    k_hideout_y = known_hideout_locations[:, 1]
    u_hideout_x = unknown_hideout_locations[:, 0]
    u_hideout_y = unknown_hideout_locations[:, 1]
    ax.scatter(k_hideout_x, k_hideout_y, marker='^', c='gold', label='known hideout')
    ax.scatter(u_hideout_x, u_hideout_y, marker='^', c='slateblue', label='unknown hideout')

    ax.legend()
    ax.set_xlim(0, 2428)
    ax.set_ylim(0, 2428)
    plt.savefig(save_path)


def plot_paths(dataset_path, save_path):
    data = np.load(dataset_path, allow_pickle=True)
    # red_locations = data['red_locations'] * 2428
    known = [(2077, 2151), (2170, 603), (37, 1293), (1890, 30), (1151, 2369), (356, 78), (1751, 1433), (1638, 1028), (1482, 387), (457, 1221)]
    unknown = [(234, 2082), (1191, 950), (563, 750), (2314, 86), (1119, 1623), (1636, 2136), (602, 1781), (2276, 1007), (980, 118), (2258, 1598)]
    red_locs = data["red_locs"]
    x = red_locs[:, 0]*2428
    y = red_locs[:, 1]*2428
    fig, ax = plt.subplots(figsize =(10, 7))
    # Creating plot
    # # plt.hexbin(x, y, bins=30)
    # # plt.scatter([500], [500], c='r', s=100)
    plt.hist2d(x, y, bins=200)

    plt.scatter(np.array(known)[:, 0], np.array(known)[:, 1], marker='X', c='r', s=100, label="Known Hideouts")
    plt.scatter(np.array(unknown)[:, 0], np.array(unknown)[:, 1], marker='D', c='yellow', s=100, label="Unknown Hideouts")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)


if __name__ == '__main__':
    dataset_path = "/data/prisoner_datasets/october_datasets/2_detect/test_graph.npz"
    data = np.load(dataset_path, allow_pickle=True)

    known_hideout_locations = np.array([[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624],
                                        [234, 171], [2398, 434], [633, 2136], [1590, 2]])
    unknown_hideout_locations = np.array([[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883],
                                          [1077, 1445], [602, 372], [80, 2274], [279, 477]])

    # Print data statistics
    dones = data['dones']
    red_locs = data['red_locations']
    # detected_locations = data['detected_location']
    # detected_locations = detected_locations[detected_locations != -1]
    # detected_locations = detected_locations.reshape(-1, 2)
    # num_detections = np.where(detected_locations != -1)[0].shape
    # print("Num detections: ", num_detections)
    # print("Num steps: ", red_locs.shape[0])

    # Calculate histogram of the number of times the fugitive visited each hideout
    # end_locations = data['red_locs'][dones == 1, :] * 2428
    # counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for p in range(end_locations.shape[0]):
    #     u = (unknown_hideout_locations - end_locations[p]) * (unknown_hideout_locations - end_locations[p])
    #     u = u.dot(np.ones([2, 1]))
    #     counts[np.argmin(u)] += 1
    #
    # print(np.sum(counts))
    # print(counts)
    # plt.bar(np.arange(10)+1, counts)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # plt.savefig('visualize/3_dataset_train.png')
    # plot_trajectory_occupancy_map(red_locs=red_locs, save_path="visualize/hideouts.png")




# plot_paths("/star-data/sye40/datasets/rrt_fix_cams_corner/train_graphs.npz", "visualize/rsl_paths.png")
plot_trajectory_occupancy_map("visualize/2_dataset_test_traj_rollouts.png", red_locs)
