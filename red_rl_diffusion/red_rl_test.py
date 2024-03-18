import os
import time
from pathlib import Path

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
import yaml

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator.prisoner_perspective_envs import PrisonerRedEnv
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid, AStarOnly
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from fugitive_policies.a_star_VO import AStarVO
# from heuristic import HierRLBlue
import matplotlib
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from blue_bc.maddpg import BaseMADDPG
from SAC.sac import SAC
from red_bc.heuristic import BlueHeuristic

matplotlib.use('TKagg')
import matplotlib.pylab
from utils import save_video
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
# from prioritized_memory import Memory
from diffuser.datasets.multipath import NAgentsIncrementalDataset
from fugitive_policies.diffusion_policy import DiffusionGlobalPlannerSelHideouts
from fugitive_policies.diffusion_policy import DiffusionGlobalPlannerSelHideouts, DiffusionStateOnlyGlobalPlanner

from enum import Enum, auto

class Estimator(Enum):
    DETECTIONS = auto()
    LEARNED_ESTIMATOR = auto()
    LINEAR_ESTIMATOR = auto()
    MDN_ESTIMATOR = auto()
    NO_DETECTIONS = auto()
    FLAT_SEQUENCE = auto()

def heuristic_evaluate(bench, bench_env_name, base_dir, episode_num = 100):
    evaluate_config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_network.yaml")
    env_config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_env.yaml")
    evaluate_base_dir = str(base_dir/(bench+"_"+bench_env_name))

    # INFO: set data container
    models_episodes_detection_rate = []
    models_episodes_detection_rate_std = []
    models_episodes_closest_dist = []
    models_episodes_closest_dist_std = []
    models_episodes_timestep = []
    models_episodes_timestep_std = []
    models_episodes_successRate = []
    models_episodes_successRate_std = []
    agents_models_episodes_reward = []   
    agents_models_episodes_reward_std = []

    # INFO: construct folder structures
    evaluate_log_dir = evaluate_base_dir + "/log" 
    detection_rate_path = evaluate_log_dir + "/detections.txt"
    detection_rate_std_path = evaluate_log_dir + "/detections_std.txt"
    closest_dist_path = evaluate_log_dir + "/closest_dist.txt"
    closest_dist_std_path = evaluate_log_dir + "/closest_dist_std.txt"
    timestep_path = evaluate_log_dir + "/time.txt"
    timestep_std_path = evaluate_log_dir + "/time_std.txt"
    successRate_path = evaluate_log_dir + "/success.txt"
    successRate_std_path = evaluate_log_dir + "/success_std.txt"
    reward_path = evaluate_log_dir + "/scores.txt"
    reward_std_path = evaluate_log_dir + "/scores_std.txt"
    evaluate_video_dir = evaluate_base_dir + "/video/"
    os.makedirs(evaluate_log_dir, exist_ok=True)
    os.makedirs(evaluate_video_dir, exist_ok=True)

    # INFO: Load the environment
    device = 'cuda' if evaluate_config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, evaluate_config["environment"]["seed"]))

    # INFO: set seeds
    np.random.seed(evaluate_config["environment"]["seed"])
    random.seed(evaluate_config["environment"]["seed"])

    env = load_environment(env_config)
    
    blue_policy = BlueHeuristic(env, debug=False)

    if evaluate_config["train"]["behavior_policy"] == 'AStar':
        red_policy = AStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
    elif evaluate_config["train"]["behavior_policy"] == 'RRTStar':
        red_policy = RRTStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, n_iter=2000, goal_sample_rate=0.1, step_len=75, search_radius=75)
    elif evaluate_config["train"]["behavior_policy"] == 'VO':
        red_policy = AStarVO(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
    else:
        raise NotImplementedError
    
    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()


    episodes_detection = []
    episodes_closest_dist = []
    episodes_timestep = []
    episodes_success = []
    agents_episodes_reward = []
    for ep in range(episode_num):

        eval_seed = int(ep + 1e6 + 1)

        # INFO: set seeds for numpy and environment as the episode num
        np.random.seed(ep)
        random.seed(ep)
        red_observation, red_partial_observation = env.reset(seed=eval_seed)
        red_policy.reset()

        t = 0
        imgs = []
        done = False
        agent_num = 1
        episode_detection = np.array([0])
        episode_closest_dist = np.array([0])
        agents_episode_reward = np.zeros(agent_num)
        while not done:
            # INFO: run episode
            t = t + 1
            red_actions = red_policy.predict(env.get_fugitive_observation())
            next_red_observation, rewards, done, i, _, red_detected_by_hs_flag = env.step(red_actions[0])
            if not done:
                episode_closest_dist = episode_closest_dist + np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1))
                agents_episode_reward = agents_episode_reward + env.get_eval_score()
                episode_detection = episode_detection + red_detected_by_hs_flag
            if t == env_config["max_timesteps"]:
                agents_episode_reward = agents_episode_reward - 50
            if ep % 50 == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
        print("This episode contains %d steps" % t)
        episodes_detection.append(episode_detection)
        episodes_closest_dist.append(episode_closest_dist / t)
        episodes_timestep.append(t)
        episodes_success.append(t<env_config["max_timesteps"])
        agents_episodes_reward.append(agents_episode_reward)
        print("complete %f of the testing" % (ep/episode_num))
        if ep % 50 == 0:
            video_path = evaluate_video_dir + (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
    models_episodes_detection_rate.append(np.array(episodes_detection).mean(axis=0))
    models_episodes_detection_rate_std.append(np.array(episodes_detection).std(axis=0))
    models_episodes_closest_dist.append(np.array(episodes_closest_dist).mean(axis=0))
    models_episodes_closest_dist_std.append(np.array(episodes_closest_dist).std(axis=0))
    models_episodes_timestep.append(np.array(episodes_timestep).mean(axis=0))
    models_episodes_timestep_std.append(np.array(episodes_timestep).std(axis=0))
    models_episodes_successRate.append(np.array(episodes_success).mean(axis=0))
    models_episodes_successRate_std.append(np.array(episodes_success).std(axis=0))
    agents_models_episodes_reward.append(np.array(agents_episodes_reward).mean(axis=0))
    agents_models_episodes_reward_std.append(np.array(agents_episodes_reward).std(axis=0))
    # INFO: save the mean and std into files
    np.savetxt(detection_rate_path, (episodes_detection))
    np.savetxt(detection_rate_std_path, (models_episodes_detection_rate_std))
    np.savetxt(closest_dist_path, (episodes_closest_dist))
    np.savetxt(closest_dist_std_path, (models_episodes_closest_dist_std))
    np.savetxt(timestep_path, (episodes_timestep))
    np.savetxt(timestep_std_path, (models_episodes_timestep_std))
    np.savetxt(successRate_path, (episodes_success))
    np.savetxt(successRate_std_path, (models_episodes_successRate_std))
    np.savetxt(reward_path, (agents_episodes_reward))
    np.savetxt(reward_std_path, (agents_models_episodes_reward_std))
    return

def red_rl_baseline(bench, bench_env_name, base_dir, episode_num = 100):
    # INFO: load the configurations from the base dir
    config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_network.yaml")
    env_config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_env.yaml")
    models_episodes_detection_rate = []
    models_episodes_detection_rate_std = []
    models_episodes_closest_dist = []
    models_episodes_closest_dist_std = []
    models_episodes_timestep = []
    models_episodes_timestep_std = []
    models_episodes_successRate = []
    models_episodes_successRate_std = []
    agents_models_episodes_reward = []   
    agents_models_episodes_reward_std = []
    # INFO: set up file and folder structure
    evaluate_log_dir = str(base_dir / (bench+"_"+bench_env_name) / "log")
    video_dir = base_dir / (bench+"_"+bench_env_name) / "video"
    model_dir = base_dir / ("model_"+bench_env_name)
    detection_rate_path = evaluate_log_dir + "/detections.txt"
    detection_rate_std_path = evaluate_log_dir + "/detections_std.txt"
    closest_dist_path = evaluate_log_dir + "/closest_dist.txt"
    closest_dist_std_path = evaluate_log_dir + "/closest_dist_std.txt"
    timestep_path = evaluate_log_dir + "/time.txt"
    timestep_std_path = evaluate_log_dir + "/time_std.txt"
    successRate_path = evaluate_log_dir + "/success.txt"
    successRate_std_path = evaluate_log_dir + "/success_std.txt"
    reward_path = evaluate_log_dir + "/scores.txt"
    reward_std_path = evaluate_log_dir + "/scores_std.txt"
    os.makedirs(evaluate_log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    # INFO: Load the environment
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # INFO: set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

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
    else:
        raise NotImplementedError

    env = PrisonerRedEnv(env, blue_policy)
    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())

    # INFO: Load the maddpg model
    agent_num = 1 # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"] 
    filtering_input_dims = [[0] for i in range(agent_num)] # no filter input for from the fugitive perspective
    obs_dims=[red_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    maddpg = BaseMADDPG(
                        agent_num = agent_num, 
                        num_in_pol = red_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (red_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    maddpg.init_from_save(model_dir / "model.pth")
    recent_episode = 0

    episodes_detection = []
    episodes_closest_dist = []
    episodes_timestep = []
    episodes_success = []
    agents_episodes_reward = []
    for ep in range(recent_episode, episode_num):

        eval_seed = int(ep + 1e6 + 1)

        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset(seed=eval_seed)
        incremental_dataset = NAgentsIncrementalDataset(env)

        # last_two_detections_vel = env.get_t_init_le_timeInterval()
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        episode_detection = np.array([0])
        episode_closest_dist = np.array([0])
        agents_episode_reward = np.zeros(agent_num)
        while not done:
            # INFO: run episode
            t = t + 1

            # INFO: Use maddpg
            torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_red_observation, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] 
            next_red_observation, rewards, done, i, _, red_detected_by_hs_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            if not done:
                episode_closest_dist = episode_closest_dist + np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1))
                agents_episode_reward = agents_episode_reward + env.get_eval_score()
                episode_detection = episode_detection + red_detected_by_hs_flag   
            if t == env_config["max_timesteps"]:
                agents_episode_reward = agents_episode_reward - 50                

            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % 50 == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)
        episodes_detection.append(episode_detection)
        episodes_closest_dist.append(episode_closest_dist / t)
        episodes_timestep.append(t)
        episodes_success.append(t<env_config["max_timesteps"])
        agents_episodes_reward.append(agents_episode_reward)

        print("complete %f of the testing" % (ep/episode_num))
        if ep % 50 == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
    models_episodes_detection_rate.append(np.array(episodes_detection).mean(axis=0))
    models_episodes_detection_rate_std.append(np.array(episodes_detection).std(axis=0))
    models_episodes_closest_dist.append(np.array(episodes_closest_dist).mean(axis=0))
    models_episodes_closest_dist_std.append(np.array(episodes_closest_dist).std(axis=0))
    models_episodes_timestep.append(np.array(episodes_timestep).mean(axis=0))
    models_episodes_timestep_std.append(np.array(episodes_timestep).std(axis=0))
    models_episodes_successRate.append(np.array(episodes_success).mean(axis=0))
    models_episodes_successRate_std.append(np.array(episodes_success).std(axis=0))
    agents_models_episodes_reward.append(np.array(agents_episodes_reward).mean(axis=0))
    agents_models_episodes_reward_std.append(np.array(agents_episodes_reward).std(axis=0))
    # INFO: save the mean and std into files
    np.savetxt(detection_rate_path, (episodes_detection))
    np.savetxt(detection_rate_std_path, (models_episodes_detection_rate_std))
    np.savetxt(closest_dist_path, (episodes_closest_dist))
    np.savetxt(closest_dist_std_path, (models_episodes_closest_dist_std))
    np.savetxt(timestep_path, (episodes_timestep))
    np.savetxt(timestep_std_path, (models_episodes_timestep_std))
    np.savetxt(successRate_path, (episodes_success))
    np.savetxt(successRate_std_path, (models_episodes_successRate_std))
    np.savetxt(reward_path, (agents_episodes_reward))
    np.savetxt(reward_std_path, (agents_models_episodes_reward_std))
    return

def red_rl_baseline_sac_evaluate(bench, bench_env_name, base_dir, episode_num = 100):
    # INFO: load the configurations from the base dir
    config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_network.yaml")
    env_config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_env.yaml")
    models_episodes_detection_rate = []
    models_episodes_detection_rate_std = []
    models_episodes_closest_dist = []
    models_episodes_closest_dist_std = []
    models_episodes_timestep = []
    models_episodes_timestep_std = []
    models_episodes_successRate = []
    models_episodes_successRate_std = []
    agents_models_episodes_reward = []   
    agents_models_episodes_reward_std = []

    # INFO: set up file and folder structure  
    evaluate_log_dir = str(base_dir / (bench+"_"+bench_env_name) / "log")
    video_dir = base_dir / (bench+"_"+bench_env_name) / "video"
    dataset_dir = base_dir / (bench+"_"+bench_env_name) / "data"
    model_dir = base_dir / ("model_"+bench_env_name)
    detection_rate_path = evaluate_log_dir + "/detections.txt"
    detection_rate_std_path = evaluate_log_dir + "/detections_std.txt"
    closest_dist_path = evaluate_log_dir + "/closest_dist.txt"
    closest_dist_std_path = evaluate_log_dir + "/closest_dist_std.txt"
    timestep_path = evaluate_log_dir + "/time.txt"
    timestep_std_path = evaluate_log_dir + "/time_std.txt"
    successRate_path = evaluate_log_dir + "/success.txt"
    successRate_std_path = evaluate_log_dir + "/success_std.txt"
    reward_path = evaluate_log_dir + "/scores.txt"
    reward_std_path = evaluate_log_dir + "/scores_std.txt"
    os.makedirs(evaluate_log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # INFO: Load the environment
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # INFO: set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())

    # INFO: Load the maddpg model
    agent_num = 1 # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"] 
    filtering_input_dims = [[0] for i in range(agent_num)]
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
                hidden_dim=config["train"]["hidden_dim"], device=device, constrained=False)
    sac.init_from_save(model_dir / "model.pth", evaluate=True)
    recent_episode = 0

    episodes_detection = []
    episodes_closest_dist = []
    episodes_timestep = []
    episodes_success = []
    agents_episodes_reward = []
    for ep in range(recent_episode, episode_num):

        eval_seed = int(ep + 1e6 + 1)

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset(seed=eval_seed)
        incremental_dataset = NAgentsIncrementalDataset(env)

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        episode_detection = np.array([0])
        episode_closest_dist = np.array([0])
        agents_episode_reward = np.zeros(agent_num)

        prisoner_locs = []
        blue_locs = []
        closest_dist = []
        prisoner_speeds = []
        while not done:
            # INFO: run episode
            t = t + 1

            # INFO: Use sac
            torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
            torch_agent_actions = sac.select_action(torch_red_observation)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            next_red_observation, rewards, done, i, _, red_detected_by_hs_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            if not done:
                episode_closest_dist = episode_closest_dist + np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1))
                agents_episode_reward = agents_episode_reward + env.get_eval_score()
                episode_detection = episode_detection + red_detected_by_hs_flag   
            if t == env_config["max_timesteps"]:
                agents_episode_reward = agents_episode_reward - 50            

            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc
            prisoner_locs.append(prisoner_loc)
            search_party_locations, helicopter_locations = env.get_blue_locations()
            blue_loc = np.concatenate((helicopter_locations, search_party_locations))
            blue_locs.append(blue_loc)
            closest_dist.append(np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1)))
            prisoner_speeds.append(np.linalg.norm(env.prisoner.step_dist_xy))

            if ep % 50 == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)
        episodes_detection.append(episode_detection)
        episodes_closest_dist.append(episode_closest_dist / t)
        episodes_timestep.append(t)
        episodes_success.append(t<env_config["max_timesteps"])
        agents_episodes_reward.append(agents_episode_reward)

        print("complete %f of the testing" % (ep / episode_num))
        if ep % 50 == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
            np.savez(dataset_dir / ("diffusionPath_blueInitLoc_Reward_%d.npz" % (ep)), prisoner_locs=prisoner_locs, prisoner_speed=prisoner_speeds, blue_locs=blue_locs, diffusion_path=env.waypoints, closest_dist=closest_dist, hideout_locs=env.hideout_locations)

    # INFO: these are the final mean and std (across all episodes)
    models_episodes_detection_rate.append(np.array(episodes_detection).mean(axis=0))
    models_episodes_detection_rate_std.append(np.array(episodes_detection).std(axis=0))
    models_episodes_closest_dist.append(np.array(episodes_closest_dist).mean(axis=0))
    models_episodes_closest_dist_std.append(np.array(episodes_closest_dist).std(axis=0))
    models_episodes_timestep.append(np.array(episodes_timestep).mean(axis=0))
    models_episodes_timestep_std.append(np.array(episodes_timestep).std(axis=0))
    models_episodes_successRate.append(np.array(episodes_success).mean(axis=0))
    models_episodes_successRate_std.append(np.array(episodes_success).std(axis=0))
    agents_models_episodes_reward.append(np.array(agents_episodes_reward).mean(axis=0))
    agents_models_episodes_reward_std.append(np.array(agents_episodes_reward).std(axis=0))
    # INFO: save the mean and std into files
    np.savetxt(detection_rate_path, (episodes_detection))
    np.savetxt(detection_rate_std_path, (models_episodes_detection_rate_std))
    np.savetxt(closest_dist_path, (episodes_closest_dist))
    np.savetxt(closest_dist_std_path, (models_episodes_closest_dist_std))
    np.savetxt(timestep_path, (episodes_timestep))
    np.savetxt(timestep_std_path, (models_episodes_timestep_std))
    np.savetxt(successRate_path, (episodes_success))
    np.savetxt(successRate_std_path, (models_episodes_successRate_std))
    np.savetxt(reward_path, (agents_episodes_reward))
    np.savetxt(reward_std_path, (agents_models_episodes_reward_std))
    return

def red_rl_piece_sac_evaluate(bench, bench_env_name, base_dir, episode_num = 100):
    # INFO: load the configurations from the base dir
    config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_network.yaml")
    env_config = config_loader(path=base_dir/("parameter_"+bench_env_name)/"parameters_env.yaml")
    models_episodes_detection_rate = []
    models_episodes_detection_rate_std = []
    models_episodes_closest_dist = []
    models_episodes_closest_dist_std = []
    models_episodes_timestep = []
    models_episodes_timestep_std = []
    models_episodes_successRate = []
    models_episodes_successRate_std = []
    agents_models_episodes_reward = []   
    agents_models_episodes_reward_std = []

    # INFO: set up file and folder structure
    evaluate_log_dir = str(base_dir / (bench+"_"+bench_env_name) / "log")
    model_dir = base_dir / ("model_"+bench_env_name)
    video_dir = base_dir / (bench+"_"+bench_env_name) / "video"
    dataset_dir = base_dir / (bench+"_"+bench_env_name) / "data"
    detection_rate_path = evaluate_log_dir + "/detections.txt"
    detection_rate_std_path = evaluate_log_dir + "/detections_std.txt"
    closest_dist_path = evaluate_log_dir + "/closest_dist.txt"
    closest_dist_std_path = evaluate_log_dir + "/closest_dist_std.txt"
    timestep_path = evaluate_log_dir + "/time.txt"
    timestep_std_path = evaluate_log_dir + "/time_std.txt"
    successRate_path = evaluate_log_dir + "/success.txt"
    successRate_std_path = evaluate_log_dir + "/success_std.txt"
    reward_path = evaluate_log_dir + "/scores.txt"
    reward_std_path = evaluate_log_dir + "/scores_std.txt"
    os.makedirs(evaluate_log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # INFO: Load the environment
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    # INFO: diffusion only
    if "Sel" not in bench:
        diffusion_path = model_dir / "diffusion.pth"
        red_policy = DiffusionStateOnlyGlobalPlanner(env, diffusion_path, plot=False, traj_grader_path=None, sel=False)  
    # INFO: diffusion + sel 
    else:
        diffusion_path = model_dir / "diffusion.pth"
        costmap_path = model_dir / "costmap.npz"
        red_policy = DiffusionStateOnlyGlobalPlanner(env, diffusion_path, plot=False, traj_grader_path=None, costmap=np.load(costmap_path)["costmap"], res=np.load(costmap_path)["res"], sel=True)    


    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset(seed=None, reset_type=None, red_policy=red_policy)
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
    sac.init_from_save(model_dir / "model.pth")
    recent_episode = 0

    episodes_detection = []
    episodes_closest_dist = []
    episodes_timestep = []
    episodes_success = []
    agents_episodes_reward = []
    for ep in range(recent_episode, episode_num):
        eval_seed = int(ep + 1e6 + 1)

        env.set_dist_coeff(-1, -1, 0)

        # INFO: set seeds for numpy and environment as the episode num
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        red_observation, red_partial_observation = env.reset(seed=eval_seed, reset_type=None, red_policy=red_policy, waypt_seed=eval_seed)
        incremental_dataset = NAgentsIncrementalDataset(env)

        # last_two_detections_vel = env.get_t_init_le_timeInterval()
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        episode_detection = np.array([0])
        episode_closest_dist = np.array([0])
        agents_episode_reward = np.zeros(agent_num)
        prisoner_locs = []
        blue_locs = []
        closest_dist = []
        prisoner_speeds = []
        

        while not done:
            # INFO: run episode
            t = t + 1

            # INFO: Use diffusion
            torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
            if bench == "Diffusion":
                to_waypt_vec_normalized = torch_red_observation[0][-3:-1] / (torch.linalg.norm(torch_red_observation[0][-3:-1]) + 1e-3)
                torch_agent_actions = [to_waypt_vec_normalized]
            else:
                torch_agent_actions = sac.select_action(torch_red_observation)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            next_red_observation, rewards, done, i, _, red_detected_by_hs_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            if not done:
                episode_closest_dist = episode_closest_dist + np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1))
                agents_episode_reward = agents_episode_reward + env.get_eval_score()
                episode_detection = episode_detection + red_detected_by_hs_flag  
            if t == env_config["max_timesteps"]:
                agents_episode_reward = agents_episode_reward - 50          

            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc
            prisoner_locs.append(prisoner_loc)
            search_party_locations, helicopter_locations = env.get_blue_locations()
            blue_loc = np.concatenate((helicopter_locations, search_party_locations))
            blue_locs.append(blue_loc)
            prisoner_speeds.append(np.linalg.norm(env.prisoner.step_dist_xy))
            closest_dist.append(np.min(np.linalg.norm(np.vstack((*env.get_blue_locations()[0], *env.get_blue_locations()[1])) - np.hstack((env.get_prisoner_location())), axis=-1)))

            if ep % 50 == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)
        episodes_detection.append(episode_detection)
        episodes_closest_dist.append(episode_closest_dist / t)
        episodes_timestep.append(t)
        episodes_success.append(t<env_config["max_timesteps"])
        agents_episodes_reward.append(agents_episode_reward)

        print("complete %f of the training" % (ep/episode_num))
        if ep % 50 == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
            # np.savez(dataset_dir / ("diffusionPath_blueInitLoc_Reward_%d.npz" % (ep)), prisoner_locs=prisoner_locs, blue_locs=blue_locs, diffusion_path=env.waypoints, closest_dist=closest_dist)
    models_episodes_detection_rate.append(np.array(episodes_detection).mean(axis=0))
    models_episodes_detection_rate_std.append(np.array(episodes_detection).std(axis=0))
    models_episodes_closest_dist.append(np.array(episodes_closest_dist).mean(axis=0))
    models_episodes_closest_dist_std.append(np.array(episodes_closest_dist).std(axis=0))
    models_episodes_timestep.append(np.array(episodes_timestep).mean(axis=0))
    models_episodes_timestep_std.append(np.array(episodes_timestep).std(axis=0))
    models_episodes_successRate.append(np.array(episodes_success).mean(axis=0))
    models_episodes_successRate_std.append(np.array(episodes_success).std(axis=0))
    agents_models_episodes_reward.append(np.array(agents_episodes_reward).mean(axis=0))
    agents_models_episodes_reward_std.append(np.array(agents_episodes_reward).std(axis=0))
    # INFO: save the mean and std into files
    np.savetxt(detection_rate_path, (episodes_detection))
    np.savetxt(detection_rate_std_path, (models_episodes_detection_rate_std))
    np.savetxt(closest_dist_path, (episodes_closest_dist))
    np.savetxt(closest_dist_std_path, (models_episodes_closest_dist_std))
    np.savetxt(timestep_path, (episodes_timestep))
    np.savetxt(timestep_std_path, (models_episodes_timestep_std))
    np.savetxt(successRate_path, (episodes_success))
    np.savetxt(successRate_std_path, (models_episodes_successRate_std))
    np.savetxt(reward_path, (agents_episodes_reward))
    np.savetxt(reward_std_path, (agents_models_episodes_reward_std))
    return

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
    # INFO: these configurations are used for benchmark
    bench_env_name = "prisoner"
    benchmark_type = ["A_Star(escape)","RRT_Star(escape)", "VO", "DDPG", "SAC", "Diffusion", "Diffusion_RL", "Sel_Diffusion_RL"]
    benchmark_type = ["Diffusion_RL"]

    benchmark_folder = {"A_Star(escape)": "./logs/RAL2024/benchmark_results/A_Star(escape)",
                        "RRT_Star(escape)": "./logs/RAL2024/benchmark_results/RRT_Star(escape)",
                        "VO": "./logs/RAL2024/benchmark_results/VO",
                        "DDPG": "./logs/RAL2024/benchmark_results/DDPG",
                        "SAC": "./logs/RAL2024/benchmark_results/SAC",
                        "Diffusion": "./logs/RAL2024/benchmark_results/Diffusion",
                        "Diffusion_RL": "./logs/RAL2024/benchmark_results/Diffusion_RL",
                        "Sel_Diffusion_RL": "./logs/RAL2024/benchmark_results/Sel_Diffusion_RL",
                        }

    for bench in benchmark_type:
        if "A_Star" in bench or "RRT" in bench or "VO" in bench:
            # INFO: benchmark heuristics
            heuristic_evaluate(bench, bench_env_name, base_dir=Path(benchmark_folder[bench]), episode_num = 100)
        elif "DDPG" in bench:
            # INFO: benchmark DDPG 
            red_rl_baseline(bench, bench_env_name, base_dir=Path(benchmark_folder[bench]), episode_num = 100)
        elif "SAC" in bench:
            # INFO: benchmark SAC
            red_rl_baseline_sac_evaluate(bench, bench_env_name, base_dir=Path(benchmark_folder[bench]), episode_num = 100)
        elif "Diffusion" in bench:
            # INFO: benchmark diffusion + RL and diffusion only
            red_rl_piece_sac_evaluate(bench, bench_env_name, base_dir=Path(benchmark_folder[bench]), episode_num = 100)                                            
        else:
            raise NotImplementedError

