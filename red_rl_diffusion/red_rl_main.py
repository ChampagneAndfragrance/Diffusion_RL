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
import matplotlib
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from blue_bc.maddpg import BaseMADDPG
from SAC.sac import SAC
from red_bc.heuristic import BlueHeuristic


matplotlib.use('TkAgg')
import matplotlib.pylab
from utils import save_video
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from blue_bc.buffer import ReplayBuffer, Buffer
# from prioritized_memory import Memory
from diffuser.datasets.multipath import NAgentsIncrementalDataset
from fugitive_policies.diffusion_policy import DiffusionStateOnlyGlobalPlanner

from enum import Enum, auto


def red_rl_baseline(config, env_config):
    # INFO: set up file and folder structure   
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)
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
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

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
    
    if config["train"]["continue"]:
        maddpg.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split('.')
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, buffer_dims=obs_ac_dims, is_cuda=config["environment"]["cuda"])
    for ep in range(recent_episode, config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset(seed=ep)
        incremental_dataset = NAgentsIncrementalDataset(env)

        # last_two_detections_vel = env.get_t_init_le_timeInterval()
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            if ep % config["train"]["video_step"] == 0:
                # INFO: Check the video at first to see if there is anything wrong
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
                torch_agent_actions = maddpg.step(torch_red_observation, explore=True)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            else:
                # INFO: Use the same policy to explore
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
                torch_agent_actions = maddpg.step(torch_red_observation, explore=True)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
                replay_buffer.push(red_observation, agent_actions, rewards, next_red_observation, done)                
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % config["train"]["video_step"] == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)

        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= 2 * config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=False)
                maddpg.update(sample, a_i, train_option="regular", logger=logger)

            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)

    return

def red_rl_baseline_sac(config, env_config):
    # INFO: set up file and folder structure   
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)
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
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()
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
    if config["train"]["continue"]:
        sac.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split('.')
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, buffer_dims=obs_ac_dims, is_cuda=config["environment"]["cuda"])
    for ep in range(recent_episode, config["train"]["episode_num"]):

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset()
        incremental_dataset = NAgentsIncrementalDataset(env)

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            if ep % config["train"]["video_step"] == 0:
                # INFO: Use the same policy to explore
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
                torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            else:
                # INFO: Use the same policy to explore
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
                torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
                replay_buffer.push(red_observation, agent_actions, rewards, next_red_observation, done)                
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % config["train"]["video_step"] == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)

        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            sac.save(model_dir / (str(ep) + ".pth"))
            sac.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= 2 * config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            for a_i in range(agent_num):
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=False)
                sac.update_bl(sample, a_i, train_option="regular", logger=logger)

        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)

    return

def red_rl_piece_sac(config, env_config):
    # INFO: set up file and folder structure   
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)

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
    # INFO: set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    diffusion_path = "./saved_models/diffusions/diffusion.pth"
    red_policy = DiffusionStateOnlyGlobalPlanner(env, diffusion_path, plot=False, traj_grader_path=None)    


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
    if config["train"]["continue"]:
        sac.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split('.')
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, buffer_dims=obs_ac_dims, is_cuda=config["environment"]["cuda"])
    for ep in range(recent_episode, config["train"]["episode_num"]):

        # INFO: Set the dist penalty coefficient
        env.set_dist_coeff(ep, config["train"]["dist_coeff_episode_num"], 0.05)

        # INFO: Start a new episode
        
        red_observation, red_partial_observation = env.reset(seed=ep, reset_type=None, red_policy=red_policy, waypt_seed=ep)
        incremental_dataset = NAgentsIncrementalDataset(env)

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False

        

        while not done:
            # INFO: run episode
            t = t + 1

            if ep % config["train"]["video_step"] == 0:
                # INFO: Use diffusion guidance
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
                # to_waypt_vec_normalized = torch_red_observation[0][-3:-1] / (torch.linalg.norm(torch_red_observation[0][-3:-1]) + 1e-3)
                # torch_agent_actions = [to_waypt_vec_normalized]
                torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
            else:
                # INFO: Use diffusion guidance
                torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
                # to_waypt_vec_normalized = torch_red_observation[0][-3:-1] / (torch.linalg.norm(torch_red_observation[0][-3:-1]) + 1e-3)
                # torch_agent_actions = [to_waypt_vec_normalized]
                torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
                replay_buffer.push(red_observation, agent_actions, rewards, next_red_observation, done)                
            # next_torch_red_observation = [Variable(torch.Tensor(next_red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % config["train"]["video_step"] == 0:
                # grid = get_probability_grid(env.nonlocalized_trgt_gaussians, np.array(env.prisoner.location))
                # search_party_locations, helicopter_locations = env.get_blue_locations()
                # heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=env.nonlocalized_trgt_gaussians[1][0]*2428)
                game_img = env.render('Policy', show=False, fast=True)
                # img = stack_game_heatmap(game_img, heatmap_img)
                imgs.append(game_img)

        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            sac.save(model_dir / (str(ep) + ".pth"))
            sac.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= 2 * config["train"]["batch_size"]:
            for a_i in range(agent_num):
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=False)
                sac.update_bl(sample, a_i, train_option="regular", logger=logger)

        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)

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
    config = config_loader(path="./red_rl_diffusion/configs/parameters_training_combine.yaml") # load model configuration
    env_config = config_loader(path=config["environment"]["env_config_file"])
    """create base dir"""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("./logs/marl") / timestr
    os.makedirs(base_dir, exist_ok=True)
    """Benchmark Starts Here"""
    # INFO: Specify the benchmarking parameters: random seeds, learning rates
    seeds = [0]
    critic_lrs = [0.003]
    policy_lrs = [0.003]
    entropy_lrs = [0.003]
    threat_lrs = [0.003]
    load_checkpoint = False
    start_episode = 0
    if load_checkpoint:
        print("\033[33mYou are loading checkpoint\033[33m")
    else:
        print("\033[33mYou are NOT loading checkpoint\033[33m")
    
    for seed in seeds:
        for c_lr in critic_lrs:
            for p_lr in policy_lrs:
                for e_lr in entropy_lrs:
                    """Modify the config"""
                    config["environment"]["seed"] = seed
                    config["train"]["critic_lr"] = c_lr
                    config["train"]["policy_lr"] = p_lr
                    config["train"]["entropy_lr"] = e_lr
                    config["train"]["continue"] = load_checkpoint
                    config["train"]["start_episode"] = start_episode

                    """create base dir name for each setting"""    
                    base_dir = Path("./logs/marl") / timestr / (config["train"]["policy_type"]+"_"+config["train"]["path_type"])
                    config["environment"]["dir_path"] = str(base_dir)
                    if config["train"]["policy_level"] == "subpolicy" and config["train"]["policy_type"] == "ddpg":
                        # INFO: DDPG baseline training
                        red_rl_baseline(config, env_config)
                    elif config["train"]["policy_level"] == "subpolicy" and config["train"]["policy_type"] == "sac" and config["train"]["model_type"] == "free" and config["train"]["path_type"] == "whole":
                        # INFO: SAC baseline training
                        red_rl_baseline_sac(config, env_config)
                    elif config["train"]["policy_level"] == "subpolicy" and config["train"]["policy_type"] == "sac" and config["train"]["model_type"] == "free" and config["train"]["path_type"] == "piece":
                        # INFO: Diffusion + RL training
                        red_rl_piece_sac(config, env_config)
                    else:
                        raise NotImplementedError
