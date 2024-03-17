import os
import argparse
import time
from pathlib import Path

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
import yaml

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator import BlueSequenceEnv
from simulator.prisoner_env import PrisonerBothEnv
from simulator.gnn_wrapper import PrisonerGNNEnv
from simulator.prisoner_perspective_envs import PrisonerBlueEnv, PrisonerRedEnv
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid, AStarOnly
# from heuristic import HierRLBlue
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from blue_bc.policy import MLPNetwork, HighLevelPolicy
from blue_bc.maddpg_filtering import MADDPGFiltering, DDPGFiltering, MADDPGCommFiltering
from blue_bc.maddpg import BaseMADDPG, MADDPG, BaseHierCtrl, hierMADDPG
from SAC.sac import SAC
from red_bc.heuristic import BlueHeuristic
from visualize.render_utils import combine_game_heatmap
from heatmap import generate_heatmap_img
from visualize.render_utils import combine_game_heatmap, stack_game_heatmap
from blue_bc.utils import BaseTrainer, HierTrainer, blue_obs_type_from_estimator, get_modified_blue_obs, get_modified_blue_obs_high, \
                            get_modified_blue_obs_low, get_localized_trgt_gaussian_locations, load_filter, get_probability_grid, traj_data_collector


matplotlib.use('TkAgg')
import matplotlib.pylab
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video
from blue_bc.utils import BaseTrainer, HierTrainer
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from blue_bc.buffer import ReplayBuffer, Buffer
# from prioritized_memory import Memory
from diffuser.datasets.multipath import NAgentsIncrementalDataset
from fugitive_policies.diffusion_policy import DiffusionGlobalPlannerSelHideouts, DiffusionStateOnlyGlobalPlanner
from trajectory_grader.traj_grader_dataset import TrajGraderDataset
from torch.utils.data import DataLoader
from trajectory_grader.trajectory_graders import Grader_FC

from enum import Enum, auto

def diffusion_global_rl_local_collect(config, env_config):
    # INFO: set up dirs structure 
    base_dir = Path(config["environment"]["dir_path"])
    video_dir = base_dir / "video"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    # INFO: Save the config into the para dir
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    # INFO: Load the environment
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    # FIXME: what's the use of the next line?
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]
    # env.seed(config["environment"]["seed"])

    blue_policy = BlueHeuristic(env, debug=False)
    if config["train"]["behavior_policy"] == 'diffusion':
        diffusion_path = "./saved_models/diffusion_models/one_hideout/random/hs/diff_995000.pt"
        ema_path = "./saved_models/diffusion_models/one_hideout/random/hs/ema_995000.pt"
        estimator_path = "./saved_models/traj_graders/H240_T100/est_p4/grader_log/best.pth"
        red_policy = DiffusionGlobalPlannerSelHideouts(env, diffusion_path, ema_path, estimator_path, max_speed=env.fugitive_speed_limit)
    elif config["train"]["behavior_policy"] == 'AStar':
        red_policy = AStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
    elif config["train"]["behavior_policy"] == 'AStar_only':
        red_policy = AStarOnly(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
    elif config["train"]["behavior_policy"] == 'diffusion_state_only':
        diffusion_path = "./saved_models/diffusions/diffusion.pth"
        # traj_grader_path = "./logs/marl/20240107-163028/regular_a_star_filtering/grader_log/best.pth"
        red_policy = DiffusionStateOnlyGlobalPlanner(env, diffusion_path, plot=False, traj_grader_path=None)    
    else:
        raise NotImplementedError

    env = PrisonerRedEnv(env, blue_policy)
    # MDN_filter = load_filter(filtering_model_config=config["train"]["filtering_model_config"], filtering_model_path=config["train"]["filtering_model_path"], device=device)
    # MDN_filter = torch.load("/home/wu/GatechResearch/Zixuan/PrisonerEscape/IROS_2023_logs/filter/Heu/combine/20230302-1616/best.pth").to(device)
    # env.set_filter(filter_model=MDN_filter) 
    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset(seed=None, reset_type=None, red_policy=red_policy)
    # blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    # filtering_input = copy.deepcopy(env.get_t_d1_d0_vel())
    # filtering_input = copy.deepcopy(env.get_t_init_le_timeInterval())
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    # INFO: Load the sac model
    agent_num = 1 # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"] 
    filtering_input_dims = [[0] for i in range(agent_num)] # no filter input for from the fugitive perspective
    obs_dims=[red_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    sac = SAC(  num_in_pol = red_observation[0].shape[0], 
                num_out_pol = action_dim_per_agent, 
                num_in_critic = (red_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                discrete_action = False, 
                gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"],  entropy_lr=config["train"]["entropy_lr"],
                hidden_dim=config["train"]["hidden_dim"], policy_type=config["train"]["policy_type"], device=device, constrained=False)

    sac.init_from_save("./saved_models/prisoner_sacs/Pri_s25r13c005/model.pth", evaluate=True)
    recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, buffer_dims=obs_ac_dims, is_cuda=config["environment"]["cuda"])
    # INFO: Specify the env update period and waypoint series update period

    env_episode_num, waypt_episode_num = 1500, 1

    for env_ep in range(recent_episode, env_episode_num):

        # INFO: Initialize some buffers
        

        # INFO: Set the dist penalty coefficient
        env.set_dist_coeff(-1, -1, 0)

        for waypt_dp in range(0, waypt_episode_num):
            # INFO: Start a new episode
            red_observation, red_partial_observation = env.reset(seed=env_ep, reset_type=None, red_policy=red_policy, waypt_seed=env_ep*waypt_episode_num+waypt_dp)
            incremental_dataset = NAgentsIncrementalDataset(env)

            # last_two_detections_vel = env.get_t_init_le_timeInterval()
            prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            t = 0
            imgs = []
            done = False

            # INFO: Initialize the episode reward as zero
            agents_episode_reward = np.zeros(agent_num)
            episode_detection = np.array([0])
            redLocations_when_detectBlue = []
            prisoner_locs = []

            while not done:
                # INFO: run episode
                t = t + 1
                # INFO: Use the same policy to explore
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
                # to_waypt_vec_normalized = torch_red_observation[0][-3:-1] / (torch.linalg.norm(torch_red_observation[0][-3:-1]) + 1e-3)
                # torch_agent_actions = [to_waypt_vec_normalized]
                torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_by_hs_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
                if not done:
                    agents_episode_reward = agents_episode_reward + rewards
                    episode_detection = episode_detection + red_detected_by_hs_flag
                # next_torch_red_observation = [Variable(torch.Tensor(next_red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
                next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
                
                red_observation = next_red_observation
                prisoner_loc = next_prisoner_loc

                prisoner_locs.append(prisoner_loc)
                if np.any(np.linalg.norm(env.get_relative_hs_loc(), axis=-1) < 20):
                    redLocations_when_detectBlue.append(prisoner_loc)

                # print("blue rewards: ", rewards)
                # if env_ep % config["train"]["video_step"] == 0:
                #     # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                #     # search_party_locations, helicopter_locations = env.get_blue_locations()
                #     # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                #     game_img = env.render('Policy', show=True, fast=True)
                #     # img = stack_game_heatmap(game_img, heatmap_img)
                #     imgs.append(game_img)
            if t == 400:
                agents_episode_reward = agents_episode_reward - 50
            # INFO: save the diffusion path, blue init loc and associated path reward in the dataset
            # replay_buffer.push(red_observation, agent_actions, rewards, next_red_observation, done) 
            np.savez(dataset_dir / ("diffusionPath_blueInitLoc_Reward_%d.npz" % (env_ep*waypt_episode_num+waypt_dp)), 
                diffusion_path=env.waypoints,
                blue_init_loc=np.array((env.init_heli_loc+env.init_sp_loc)), 
                reward=agents_episode_reward,
                redLocations_when_detectBlue=redLocations_when_detectBlue, 
                prisoner_locs = prisoner_locs,
                )
            # print("reward is: ", agents_episode_reward)
            # print("episode detection is", episode_detection)
            # print("episode length is", t)
            print("complete %f of the training" % ((env_ep*waypt_episode_num+waypt_dp)/(env_episode_num*waypt_episode_num)))
            # if env_ep % config["train"]["video_step"] == 0:
            #     video_path = video_dir / (str(env_ep*waypt_episode_num+waypt_dp) + ".mp4")
            #     save_video(imgs, str(video_path), fps=10)

    return

def visualize_traj_with_reward_detection():

    def patch_shapes(mapsize, shape, centers, radii, color):
        patches = []
        if shape == "rect":
            for center, radius in zip(centers, radii):
                downleft_x = center[0] - radius
                downleft_y = mapsize[1] - center[1] - radius - 1
                patch = plt.Rectangle((downleft_x, downleft_y), radius*2, radius*2, linewidth=2, edgecolor=color, facecolor='none')
                patches.append(patch)
        if shape == "circle":
            for center, radius in zip(centers, radii):
                center_x = center[0]
                # center_y = mapsize[1] - center[1] - 1
                center_y = center[1]
                patch = plt.Circle((center_x, center_y), radius, linewidth=1.2, edgecolor=color, facecolor='none', alpha=0.7, linestyle='-')
                patches.append(patch)
        return patches

    def create_circle_array(radius, map, center):
        # Create a grid of coordinates
        x, y = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))
        
        # Calculate distance from the specified center
        distance_from_center = np.sqrt((x - center[0])**2 + (y - (map.shape[1] - center[1]))**2)
        
        # Create an array with 1 where distance is less than or equal to the radius, 0 otherwise
        circle_array = np.where(distance_from_center <= radius, 1, 0)
        
        return circle_array

    def create_squares_array(x_y_half_ranges, map, centers):
        map_copy = copy.deepcopy(map)
        margin = 1
        for center, x_y_half_range in zip(centers, x_y_half_ranges):
            x_half_range, y_half_range = x_y_half_range
            image_x = map.shape[0] - center[1]
            image_y = center[0]
            upper_left_corner_x = image_x - y_half_range - margin
            upper_left_corner_y = image_y - x_half_range - margin
            lower_right_corner_x = upper_left_corner_x + 2 * y_half_range + 2 * margin
            lower_right_corner_y = upper_left_corner_y + 2 * x_half_range + 2 * margin
            map_copy[upper_left_corner_x:lower_right_corner_x, upper_left_corner_y:lower_right_corner_y] = 1
        return map_copy

    def read_camera_file(camera_file_path):
        """Generate a lists of camera objects from file

        Args:
            camera_file_path (str): path to camera file

        Raises:
            ValueError: If Camera file does not have a u or k at beginning of each line

        Returns:
            (list, list): Returns known camera locations and unknown camera locations
        """
        unknown_camera_locations = []
        known_camera_locations = []
        camera_file = open(camera_file_path, "r").readlines()
        for line in camera_file:
            line = line.strip().split(",")
            if line[0] == 'u':
                unknown_camera_locations.append([int(line[1]), int(line[2])])
            elif line[0] == 'k':
                known_camera_locations.append([int(line[1]), int(line[2])])
            else:
                raise ValueError(
                    "Camera file format is incorrect, each line must start with 'u' or 'k' to denote unknown or known")
        return known_camera_locations, unknown_camera_locations

    # INFO: Initialize the environment
    env = load_environment(env_config)
    # INFO: Initialize datasets
    # visualize_dataset = TrajGraderDataset(file_root="./logs/marl/20240110-001808/data")
    visualize_dataset = TrajGraderDataset(file_root="./logs/marl/prisoner_1500_dataset/data")
    # INFO: Initialize dataloader
    visualize_data_loader = DataLoader(visualize_dataset, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)
    # INFO: Begin visualization
    epoch_num = 1

    res = 10

    # INFO: Initialize the detection map and visit map as zeros
    detection_map = np.zeros((int(np.ceil(2428/res)), int(np.ceil(2428/res))))
    visit_map = np.zeros((int(np.ceil(2428/res)), int(np.ceil(2428/res))))


    # fig, (axis1, axis2) = plt.subplots(2)
    for ep in tqdm(range(epoch_num)):
        for batch_idx, batch in enumerate(visualize_data_loader):
            add_map = np.zeros((2428, 2428))
            # INFO: batch is composed of red trajectories, {hideout, start loc}, start loc, rewards
            traj, detection, loc, rew = batch
            red_path = traj.reshape(-1, 2) * 2428
            # detection = detection * 2428
            color = plt.cm.viridis(rew.item())
            # plt.plot(red_path[:,0], red_path[:,1], color=color, alpha=0.5)
            # INFO: stack the detection map
            if min(detection.shape) != 0:
                # plt.scatter(detection[:,:,0], detection[:,:,1], color="b")
                # add_map[detection[:,:,0].detach().cpu().numpy().astype(int), detection[:,:,1].detach().cpu().numpy().astype(int)] = 1
                nn_output = (np.ones((detection.shape[0], detection.shape[1])), detection, 0.01*np.ones_like(detection))
                add_map = get_probability_grid(nn_output, res=res)
                detection_map = detection_map + add_map
            # INFO: stack the visit map
            # if min(loc.shape) != 0:
            #     # plt.scatter(detection[:,:,0], detection[:,:,1], color="b")
            #     # add_map[detection[:,:,0].detach().cpu().numpy().astype(int), detection[:,:,1].detach().cpu().numpy().astype(int)] = 1
            #     nn_output = (np.ones((loc.shape[0], loc.shape[1])), loc, 0.01*np.ones_like(loc))
            #     add_map = get_probability_grid(nn_output, res=res)
            #     visit_map = visit_map + add_map

            print("%f has been analyzed" % (batch_idx/1500))
        detection_map = detection_map / detection_map.max()
        # visit_map = visit_map / visit_map.max() + 1e-3

        # detection_map = detection_map / visit_map
        # detection_map = detection_map / detection_map.max()

        # INFO: set mountain range as one
        circle_array = create_circle_array(radius=300//res, map=detection_map, center=(1000//res, 1600//res))
        rect_array = create_squares_array(x_y_half_ranges=[(65//res, 135//res),(135//res, 65//res)], map=detection_map, centers=[(1600//res, 1800//res),(1600//res, 1800//res)])
        # rect_array = create_squares_array(x_y_half_ranges=[(65//res, 135//res),(135//res, 65//res),(200//res, 100//res)], map=detection_map, centers=[(1600//res, 1800//res),(1600//res, 1800//res),(1000//res, 1000//res)])
        detection_map = np.maximum(detection_map, circle_array) 
        detection_map_gt = np.maximum(detection_map, rect_array) 

    np.savez("./saved_models/prisoner_sacs/Pri_s25r13c005/costmap_addzone.npz", costmap=detection_map_gt, res=res)

    # plt.xlim(0, 2428)
    # plt.ylim(0, 2428)
    # plt.show()

    # Define custom color boundaries
    boundaries = [0, 0.2, 0.5, 1.0]  # Adjust the boundaries as needed
    colors = ['black', 'brown', 'orange', 'oldlace']  # Adjust the colors as needed

    # Plot heatmap with custom color scale
    cmap = plt.cm.colors.ListedColormap(colors)
    cmap2 = mcolors.LinearSegmentedColormap.from_list('cmap2', list(zip(boundaries, colors)))
    norm = mcolors.BoundaryNorm(boundaries, len(colors) - 1)

    # INFO: Plotting the detection map and visit map
    # plt.imshow(detection_map_gt, cmap='magma', interpolation='nearest', extent=(0, 2428, 0, 2428))
    plt.imshow(detection_map_gt, cmap=cmap2, interpolation='nearest', extent=(0, 2428, 0, 2428))
    plt.axis('off')
    # plt.imshow(visit_map, cmap='magma', interpolation='nearest')

    # INFO: Decide if we need to draw cameras on the map
    # rect = patch_shapes(mapsize=detection_map.shape, shape="rect", centers=[(1600//res, 1800//res)], radii=[110//res], color="r")
    # known_camera_locations, unknown_camera_locations = read_camera_file("simulator/camera_locations/explored_camera.txt")
    # cam_ranges = []
    # for cam_loc in unknown_camera_locations:
    #     cam_range = 3 * (4.0 * env.terrain.detection_coefficient_given_location(cam_loc) * 0.5 * 15 + 1)
    #     cam_ranges.append(cam_range)
    # circles = patch_shapes(mapsize=(2428,2428), shape="circle", centers=np.array(unknown_camera_locations), radii=np.array(cam_ranges), color="cyan")
    # for patch in circles:
    #     plt.gca().add_patch(patch)
    # # Add text on the heatmap
    # plt.text(1800, 1400, 'Cameras', fontsize=11, fontweight=600, color='cyan')
    plt.text(800, 1500, 'Danger \n  Zone', fontsize=11, fontweight=600, color='r')

    # INFO: Decide if we need to add colorbar
    # plt.colorbar()  

    # plt.title('Costmap')
    plt.savefig('costmap_addzone.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_all_traj(vis_option="speed_vs_traj"):

    # INFO: Initialize the environment
    env = load_environment(env_config)
    # INFO: Initialize datasets
    traj_roots = ["./logs/IROS2024/benchmark_results/Diffusion_RL/Diffusion_RL_prisoner/data",
                  "./logs/IROS2024/benchmark_results/SAC/SAC_prisoner/data"]
    downsample_rate = 1
    for diffusion_file, sac_file in zip(sorted(os.listdir(traj_roots[0])), sorted(os.listdir(traj_roots[1]))):
        # INFO: load the datasets of diffusion and sac
        diffusion_np_file = np.load(os.path.join(traj_roots[0], diffusion_file), allow_pickle=True)
        sac_np_file = np.load(os.path.join(traj_roots[1], sac_file), allow_pickle=True)

        diffusion_path = diffusion_np_file["diffusion_path"]
        blue_locations_diffusion = diffusion_np_file["blue_locs"][::downsample_rate,:,:]
        blue_locations_sac = sac_np_file["blue_locs"][::downsample_rate,:,:]
        prisoner_locations_diffusion = diffusion_np_file["prisoner_locs"][::downsample_rate,:]
        prisoner_locations_sac = sac_np_file["prisoner_locs"][::downsample_rate,:]
        hideout_locations_diffusion = diffusion_np_file["hideout_locs"]
        hideout_locations_sac = sac_np_file["hideout_locs"]
        closest_dist_diffusion = diffusion_np_file["closest_dist"][::downsample_rate]
        normalized_closest_dist_diffusion = closest_dist_diffusion / (2428*1.414)
        closest_dist_sac = sac_np_file["closest_dist"][::downsample_rate]
        normalized_closest_dist_sac = closest_dist_sac / (2428*1.414)
        prisoner_speed_diffusion = diffusion_np_file["prisoner_speed"][::downsample_rate] / 15
        prisoner_speed_sac = sac_np_file["prisoner_speed"][::downsample_rate] / 15
        # closest_dist = closest_dist / closest_dist.max()
        print("diffusion_path = ", diffusion_path)

        # subplot_size = [(7, 7), (7, 7), (5, 5), (5, 5)]
        figure, axes = plt.subplots(2, 2)
        # INFO: Create a colormap
        boundaries = [0, 1.0]  # Adjust the boundaries as needed
        colors_heli = ['white', 'blue']  # Adjust the colors as needed
        colors_sp = ['white', 'cyan']  # Adjust the colors as needed
        colors_prisoner = ['red', 'white']  # Adjust the colors as needed

        # INFO: Plot heatmap with custom color scale
        cmap_heli_linear = mcolors.LinearSegmentedColormap.from_list('cmap_heli_linear', list(zip(boundaries, colors_heli)))
        cmap_sp_linear = mcolors.LinearSegmentedColormap.from_list('cmap_sp_linear', list(zip(boundaries, colors_sp)))
        cmap_prisoner_linear = mcolors.LinearSegmentedColormap.from_list('cmap_prisoner_linear', list(zip(boundaries, colors_prisoner)))

        # axes.scatter(blue_locations[:,0,0], blue_locations[:,0,1], c=np.arange(prisoner_locations.shape[0]), cmap=cmap_heli_linear, alpha=1)
        # axes.plot(blue_locations[:,0,0], blue_locations[:,0,1], c='b')
        # axes.scatter(blue_locations[:,1,0], blue_locations[:,1,1], c=np.arange(prisoner_locations.shape[0]), cmap=cmap_sp_linear, alpha=0.5)
        sc_diffusion_dist = axes[0,0].scatter(prisoner_locations_diffusion[:,0], prisoner_locations_diffusion[:,1], s=5, c=closest_dist_diffusion, cmap=cmap_prisoner_linear, vmin=0, vmax=500)
        sc_sac_dist = axes[0,1].scatter(prisoner_locations_sac[:,0], prisoner_locations_sac[:,1], s=5, c=closest_dist_sac, cmap=cmap_prisoner_linear, vmin=0, vmax=500)
        if vis_option == "speed_vs_traj":
            # INFO: if we plot speed vs traj
            sc_diffusion_speed = axes[1,0].scatter(prisoner_locations_diffusion[:,0], prisoner_locations_diffusion[:,1], s=5, c=prisoner_speed_diffusion, cmap=cmap_prisoner_linear, vmin=0, vmax=15)
            sc_sac_speed = axes[1,1].scatter(prisoner_locations_sac[:,0], prisoner_locations_sac[:,1], s=5, c=prisoner_speed_sac, cmap=cmap_prisoner_linear, vmin=0, vmax=15)
        elif vis_option == "speed_vs_time":
            # INFO: if we plot speed vs time
            sc_diffusion_speed = axes[1,0].plot(prisoner_speed_diffusion)
            sc_sac_speed = axes[1,1].plot(prisoner_speed_sac)
            # sc_diffusion_dist_vs_time = axes[1,0].plot(normalized_closest_dist_diffusion)
            # sc_sac_dist_vs_time = axes[1,1].plot(normalized_closest_dist_sac)
        else:
            raise NotImplementedError
        # INFO: If there is diffusion path
        axes[0,0].scatter(diffusion_path[:,0], diffusion_path[:,1], s=10, c='purple', marker='D')
        axes[0,0].plot(diffusion_path[:,0], diffusion_path[:,1], c='purple', linewidth=1.0)
        if vis_option == "speed_vs_traj":
            axes[1,0].scatter(diffusion_path[:,0], diffusion_path[:,1], s=50, c='purple', marker='D')
            axes[1,0].plot(diffusion_path[:,0], diffusion_path[:,1], c='purple', linewidth=1.0)

        axes[0,0].scatter(prisoner_locations_diffusion[:,0][0], prisoner_locations_diffusion[:,1][0], s=30, c='gold', marker='o') # prisoner initial pose
        axes[0,0].scatter(prisoner_locations_diffusion[:,0][-1], prisoner_locations_diffusion[:,1][-1], s=30, c='blue', marker='o') # prisoner end pose
        axes[0,0].scatter(hideout_locations_diffusion[:,0], hideout_locations_diffusion[:,1], s=100, c='gold', marker='*') # hideout locations

        axes[0,1].scatter(prisoner_locations_sac[:,0][0], prisoner_locations_sac[:,1][0], s=30, c='gold', marker='o') # prisoner initial pose
        axes[0,1].scatter(prisoner_locations_sac[:,0][-1], prisoner_locations_sac[:,1][-1], s=30, c='blue', marker='o') # prisoner end pose
        axes[0,1].scatter(hideout_locations_sac[:,0], hideout_locations_sac[:,1], s=100, c='gold', marker='*') # hideout locations

        if vis_option == "speed_vs_traj":
            axes[1,0].scatter(prisoner_locations_diffusion[:,0][0], prisoner_locations_diffusion[:,1][0], s=120, c='gold', marker='o') # prisoner initial pose
            axes[1,0].scatter(prisoner_locations_diffusion[:,0][-1], prisoner_locations_diffusion[:,1][-1], s=120, c='blue', marker='o') # prisoner end pose
            axes[1,0].scatter(hideout_locations_diffusion[:,0], hideout_locations_diffusion[:,1], s=200, c='gold', marker='*') # hideout locations

            axes[1,1].scatter(prisoner_locations_sac[:,0][0], prisoner_locations_sac[:,1][0], s=120, c='gold', marker='o') # prisoner initial pose
            axes[1,1].scatter(prisoner_locations_sac[:,0][-1], prisoner_locations_sac[:,1][-1], s=120, c='blue', marker='o') # prisoner end pose
            axes[1,1].scatter(hideout_locations_sac[:,0], hideout_locations_sac[:,1], s=200, c='gold', marker='*') # hideout locations

        cax = figure.add_axes([0.25, 0.96, 0.5, 0.03])  # [left, bottom, width, height]
        cbar_dist_diffusion = plt.colorbar(sc_sac_dist, cax=cax, ax=axes[0,0], ticks=[0, 250, 500], format=mticker.FixedFormatter(['0', '250', '> 500']), orientation='horizontal')
        cbar_dist_diffusion.ax.xaxis.set_ticks_position('top')
        # cbar_dist_sac = plt.colorbar(sc_sac_dist, ax=axes[0,1], ticks=[0, 250, 500], format=mticker.FixedFormatter(['0', '250', '> 500']), orientation='horizontal')
        # cbar_dist_sac.ax.xaxis.set_ticks_position('top')

        if vis_option == "speed_vs_traj":
            cbar_speed = plt.colorbar(sc_sac_speed, ax=axes[1,1], ticks=[0, 7.5, 15], format=mticker.FixedFormatter(['0', '7.5', '15']), orientation='vertical')

        axes[0,0].imshow(env.custom_render_canvas(show=False, option=["terrain", "cameras"], large_icons=False), extent=(0, 2428, 0, 2428))
        axes[0,1].imshow(env.custom_render_canvas(show=False, option=["terrain", "cameras"], large_icons=False), extent=(0, 2428, 0, 2428))
        axes[0,0].axis("off")
        axes[0,1].axis("off")

        if vis_option == "speed_vs_traj":
            axes[1,0].imshow(env.custom_render_canvas(show=False, option=["terrain", "cameras"], large_icons=False), extent=(0, 2428, 0, 2428))
            axes[1,1].imshow(env.custom_render_canvas(show=False, option=["terrain", "cameras"], large_icons=False), extent=(0, 2428, 0, 2428))

        axes[0,0].axis('square')
        axes[0,0].set_title('Diffusion-RL Trajectory', font = {'weight': 'bold', 'size': 10})
        axes[0,0].set_xlim(0, 2428)
        axes[0,0].set_ylim(0, 2428)

        axes[0,1].axis('square')
        axes[0,1].set_title('SAC Trajectory', font = {'weight': 'bold', 'size': 10})
        axes[0,1].set_xlim(0, 2428)
        axes[0,1].set_ylim(0, 2428)

        if vis_option == "speed_vs_traj":
            # axes[1,0].axis('square')
            axes[1,0].set_title('Diffusion-RL Speed', font = {'weight': 'bold', 'size': 10})
            axes[1,0].set_xlim(0, 2428)
            axes[1,0].set_ylim(0, 2428)

            # axes[1,1].axis('square')
            axes[1,1].set_title('SAC Speed', font = {'weight': 'bold', 'size': 10})
            axes[1,1].set_xlim(0, 2428)
            axes[1,1].set_ylim(0, 2428)
        else:
            axes[1,0].set_title('Diffusion-RL Speed', font = {'weight': 'bold', 'size': 10})
            axes[1,0].set_xlabel("Timestep", font = {'weight': 'bold', 'size': 8})
            axes[1,0].set_ylabel("Speed", font = {'weight': 'bold', 'size': 8})
            axes[1,0].set_box_aspect(1) 

            axes[1,1].set_title('SAC Speed', font = {'weight': 'bold', 'size': 10})    
            axes[1,1].set_xlabel("Timestep", font = {'weight': 'bold', 'size': 8})
            axes[1,1].set_ylabel("Speed", font = {'weight': 'bold', 'size': 8})
            axes[1,1].set_box_aspect(1)       

        # plt.savefig('evasive_traj.png', dpi=200, bbox_inches='tight')
        plt.show()

    return 


def visualize_custom_environment():

    # INFO: Initialize the environment
    env = load_environment(env_config)
    env.reset_env(seed=600)
    # INFO: Initialize datasets
    traj_roots = ["./logs/IROS2024/benchmark_results/Diffusion_RL/Diffusion_RL_prisoner/data",
                  "./logs/IROS2024/benchmark_results/SAC/SAC_prisoner/data"]
    for diffusion_file, sac_file in zip(sorted(os.listdir(traj_roots[0])), sorted(os.listdir(traj_roots[1]))):
        # INFO: load the datasets of diffusion and sac
        diffusion_np_file = np.load(os.path.join(traj_roots[0], diffusion_file), allow_pickle=True)
        sac_np_file = np.load(os.path.join(traj_roots[1], sac_file), allow_pickle=True)

        diffusion_path = diffusion_np_file["diffusion_path"]
        # closest_dist = closest_dist / closest_dist.max()
        print("diffusion_path = ", diffusion_path)
        figure, axis = plt.subplots()
        # axis.scatter(hideout_locations_sac[:,0], hideout_locations_sac[:,1], s=400, c='gold', marker='*') # hideout locations
        axis.imshow(env.custom_render_canvas(show=False, option=["prisoner", "terrain", "hideouts", "search", "cameras"], large_icons=False), extent=(0, 2428, 0, 2428))
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        # plt.axis('off')
        plt.savefig('custom_env.png', dpi=200, bbox_inches='tight')
        plt.show()


def split_red_directions_to_direction_speed(directions):
    red_actions_norm_angle_vel = []
    red_actions_directions = np.split(directions, 1)
    fugitive_v_limit = 15
    for idx in range(len(red_actions_directions)):
        fugitive_direction = red_actions_directions[idx]
        if np.linalg.norm(fugitive_direction) > 1:
            fugitive_direction = fugitive_direction / np.linalg.norm(fugitive_direction)
        fugitive_speed = np.minimum(np.linalg.norm(fugitive_direction), 1.0) * fugitive_v_limit
        red_actions_norm_angle_vel.append(np.array([fugitive_speed, np.arctan2(fugitive_direction[1], fugitive_direction[0])]))
    return red_actions_norm_angle_vel[0] 

if __name__ == '__main__':
    config = config_loader(path="./red_rl_diffusion/configs/parameters_training_combine.yaml") # load model configuration
    env_config = config_loader(path=config["environment"]["env_config_file"])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("./logs/marl") / timestr
    config["environment"]["dir_path"] = str(base_dir)
    env_config["reward_setting"] = "rl"

    # visualize_all_traj(vis_option="speed_vs_time")
    visualize_custom_environment()

