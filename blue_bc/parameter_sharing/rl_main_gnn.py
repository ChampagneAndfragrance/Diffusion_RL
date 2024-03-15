import os
import argparse
import time
from pathlib import Path

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
import yaml
import copy

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator import BlueSequenceEnv
from simulator.prisoner_env import PrisonerBothEnv
from simulator.prisoner_perspective_envs import PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid
from heuristic import HierRLBlue
import matplotlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from blue_bc.policy import MLPNetwork, HighLevelPolicy
from blue_bc.maddpg_filtering import MADDPGFiltering, DDPGFiltering, AttentionMADDPGFiltering, ActorAttentionCritcFiltering, MADDPGCommFiltering
from blue_bc.CTCE_DDPG import CTCE_DDPG_Filtering

matplotlib.use('agg')
import matplotlib.pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video
from blue_bc.utils import BaseTrainer, HierTrainer, blue_obs_type_from_estimator, get_modified_blue_obs
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from buffer import ReplayBuffer, Buffer
from blue_bc.sequence_buffer import SequenceReplayBuffer
from prioritized_memory import Memory
from maddpg import BaseMADDPG, MADDPG, BaseDDPG
from enum import Enum, auto

class Estimator(Enum):
    DETECTIONS = auto()
    LINEAR_ESTIMATOR = auto()
    NO_DETECTIONS = auto()
    FLAT_SEQUENCE = auto()

def main_reg_NeLeGt_seq(config, env_config):
    # set up file and folder structure
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator) 
    blue_raw_obs_type = blue_obs_type_from_estimator("no_estimator", Estimator)   
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    # log_dir = os.path.join("logs", "bc", str(time))
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon, max_speed=15)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env, max_speed=15)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()
    blue_observation_raw = get_modified_blue_obs(env, blue_raw_obs_type, Estimator)
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    obs_dims=[blue_observation[i].shape[0]//config["train"]["seq_len"] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    maddpg = BaseMADDPG(agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)

    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = SequenceReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"], sequence_length=config["train"]["seq_len"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    total_iter = 0
    total_linear_estimator_error = 0
    for ep in range(config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation_raw = get_modified_blue_obs(env, blue_raw_obs_type, Estimator)
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            total_iter = total_iter + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation_raw = get_modified_blue_obs(env, blue_raw_obs_type, Estimator)
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # linear_estimator_error = env.get_linear_estimation_error()
            # total_linear_estimator_error = total_linear_estimator_error + linear_estimator_error
            # print("The average estimation error is:", total_linear_estimator_error/total_iter)

            rewards = dist_reward + detect_reward
            replay_buffer.push_oar(blue_observation_raw, agent_actions, rewards, next_blue_observation_raw, done)

            blue_observation = next_blue_observation
            blue_observation_raw = next_blue_observation_raw
            blue_partial_observation = next_blue_partial_observation
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample_oar_sequence_from_array(config["train"]["batch_size"], config["train"]["seq_len"], to_gpu=config["environment"]["cuda"], norm_rews=True)
                maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
            logger.add_scalar('agent%i/average_linear_estimator_error' % a_i, total_linear_estimator_error/total_iter, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def main_reg_NeLeGt(config, env_config):
    # set up file and folder structure
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    # log_dir = os.path.join("logs", "bc", str(time))
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env, max_speed=15, cost_coeff=1000)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()
    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num

    maddpg = BaseMADDPG(agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
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

    """Initialize the buffer"""
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_filter_loc_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    total_iter = 0
    total_linear_estimator_error = 0
    for ep in range(recent_episode, config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            total_iter = total_iter + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            # rearrange actions to be per environment (each [] element corresponds to an environment id)
            # actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])]     
    #         # print("current t = ", t)
    #         # red_action = red_policy.predict(red_observation)
    #         # blue_actions = blue_heuristic.step_observation(blue_observation)
    #         """Partial Blue Obs"""
    #         # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
    #         """Full Blue Obs"""
            # action, new_detection, hier_action = blue_hier_policy.predict_full_observation(blue_observation)
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # linear_estimator_error = env.get_linear_estimation_error()
            # total_linear_estimator_error = total_linear_estimator_error + linear_estimator_error
            # print("The average estimation error is:", total_linear_estimator_error/total_iter)

            rewards = dist_reward + detect_reward
            replay_buffer.push(blue_observation, agent_actions, rewards, next_blue_observation, done)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
                imgs.append(game_img)
            
        print("complete %f of the training" % (ep/float(config["train"]["episode_num"])))
        if ep % config["train"]["video_step"] == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % config["train"]["save_interval"] == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            if config["environment"]["cuda"]:
                maddpg.prep_training(device='gpu')
            else:
                maddpg.prep_training(device='cpu')

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=True)
                maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
            logger.add_scalar('agent%i/average_linear_estimator_error' % a_i, total_linear_estimator_error/total_iter, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def main_reg_filtering(config, env_config):
    # set up file and folder structure
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()

    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)

    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num
    # maddpg = DDPGFiltering(
    #                 filtering_model_config = config["train"]["filtering_model_config"],
    #                 filtering_model_path = config["train"]["filtering_model_path"],
    #                 agent_num = agent_num, 
    #                 num_in_pol = blue_observation[0].shape[0], 
    #                 num_out_pol = action_dim_per_agent, 
    #                 num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent), 
    #                 discrete_action = False, 
    #                 gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    maddpg = MADDPGFiltering(
                        filtering_model_config = config["train"]["filtering_model_config"],
                        filtering_model_path = config["train"]["filtering_model_path"],
                        agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
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





    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = SequenceReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_dims=obs_ac_dims, is_cuda=config["environment"]["cuda"], filter_dim = (109, 9))
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    for ep in range(recent_episode, config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        gnn_obs = env.get_gnn_obs() # gnn_obs[0]: [agent_num (camera, blue agent, etc.), feature_num]
        gnn_sequence_obs = env.get_gnn_sequence() # gnn_sequence_obs[0]: [seq_len, agent_num (camera, blue agent, etc.), feature_num]
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            gnn_sequence_obs = env.get_gnn_sequence()
            gnn_input_tensors = [torch.tensor(i).unsqueeze(0).to(device) for i in gnn_sequence_obs]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, gnn_input_tensors, explore=True)
            # err = maddpg.test_loc_est_error(normalized_prisoner_gt_loc=np.array(prisoner_loc)/2428, filtering_input=gnn_input_tensors)
            # print("prisoner_loc est. error = ", err)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5

            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            next_gnn_obs = env.get_gnn_obs()
            # next_k_detections = env.get_last_k_fugitive_detections()
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            rewards = dist_reward + detect_reward
            # print("The dist_reward is: ", dist_reward)
            # print("The detect_reward is: ", detect_reward)

            replay_buffer.push(blue_observation, agent_actions, rewards, next_blue_observation, done, gnn_obs, next_gnn_obs, np.array(prisoner_loc)/2428)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            gnn_obs = next_gnn_obs
            prisoner_loc = next_prisoner_loc

            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
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
                sample = replay_buffer.sample(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=True)
                maddpg.update(sample[0:5], sample[5:], a_i, train_option="regular", logger=logger)
                # maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def main_reg_last_two_filtering(config, env_config):
    # set up file and folder structure
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
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
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, max_speed=env.fugitive_speed_limit, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()

    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    # filtering_input = copy.deepcopy(env.get_t_d1_d0_vel())
    filtering_input = copy.deepcopy(env.get_t_init_le_timeInterval())

    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2 + env_config["comm_dim"]
    # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
    filtering_input_dims = [[len(filtering_input)] for i in range(agent_num)]
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num
    # maddpg = DDPGFiltering(
    #                 filtering_model_config = config["train"]["filtering_model_config"],
    #                 filtering_model_path = config["train"]["filtering_model_path"],
    #                 agent_num = agent_num, 
    #                 num_in_pol = blue_observation[0].shape[0], 
    #                 num_out_pol = action_dim_per_agent, 
    #                 num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent), 
    #                 discrete_action = False, 
    #                 gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    maddpg = MADDPGCommFiltering(
                        filtering_model_config = config["train"]["filtering_model_config"],
                        filtering_model_path = config["train"]["filtering_model_path"],
                        agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0], 
                        num_out_pol = action_dim_per_agent, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        comm_dim = env_config["comm_dim"],
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], filter_lr=config["train"]["filter_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    

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





    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_filter_loc_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    for ep in range(recent_episode, config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        # gnn_obs = env.get_gnn_obs() # gnn_obs[0]: [agent_num (camera, blue agent, etc.), feature_num]
        # gnn_sequence_obs = env.get_gnn_sequence() # gnn_sequence_obs[0]: [seq_len, agent_num (camera, blue agent, etc.), feature_num]
        # last_two_detections_vel = env.get_t_d1_d0_vel()
        last_two_detections_vel = env.get_t_init_le_timeInterval()
        # prior_input = filtering_input[..., 0:3]
        # dynamic_input = filtering_input[..., 3:]
        # sel_input = filtering_input
        # filtering_output = model(torch.Tensor(prior_input).unsqueeze(0).to(device), torch.Tensor(dynamic_input).unsqueeze(0).to(device), torch.Tensor(sel_input).unsqueeze(0).to(device))
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # gnn_sequence_obs = env.get_gnn_sequence()
            mlp_input_tensors = torch.Tensor(last_two_detections_vel).unsqueeze(0).to(device)
            # get actions as torch Variables
            torch_agent_actions, torch_agents_comms = maddpg.step(torch_obs, mlp_input_tensors, explore=True)
            # err = maddpg.test_loc_est_error(normalized_prisoner_gt_loc=np.array(prisoner_loc)/2428, filtering_input=gnn_input_tensors)
            # print("prisoner_loc est. error = ", err)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
            agents_comms = [ac_comm.data.cpu().numpy() for ac_comm in torch_agents_comms]
            agent_actionsComms = [np.concatenate((ac.data.cpu().numpy(), ac_comm.data.cpu().numpy()), axis=-1) for (ac, ac_comm) in zip(torch_agent_actions, torch_agents_comms)]
            # env.update_filtering_reward(maddpg.get_filter_loss(mlp_input_tensors, np.array(prisoner_loc)/2428))
            env.update_comm(agents_comms)
            # print("maddpg.filtering_loss = ", maddpg.get_filter_loss(mlp_input_tensors, np.array(prisoner_loc)/2428))
            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # next_last_two_detections_vel = env.get_t_d1_d0_vel()
            next_last_two_detections_vel = env.get_t_init_le_timeInterval()
            # next_k_detections = env.get_last_k_fugitive_detections()
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            rewards = dist_reward + detect_reward
            # print("The dist_reward is: ", dist_reward)
            # print("The detect_reward is: ", detect_reward)

            replay_buffer.push_filter(blue_observation, agent_actionsComms, rewards, next_blue_observation, done, last_two_detections_vel, next_last_two_detections_vel, np.array(prisoner_loc)/2428)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            last_two_detections_vel = next_last_two_detections_vel
            prisoner_loc = next_prisoner_loc

            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
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
                sample = replay_buffer.sample_filter(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=True)
                maddpg.update(sample, a_i, train_option="regular", logger=logger)
                # maddpg.update(sample, a_i, logger=logger)
            # if ep % 1 == 0 and ep != 0 and ep > 100:
            #     maddpg.update_filter(replay_buffer, config, data_dir=dataset_dir, curr_episode=ep, epoch_num=1)

            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def main_CTCE_reg_last_two_filtering(config, env_config):
    # set up file and folder structure
    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    """Specify the writer"""
    logger = SummaryWriter(log_dir=log_dir)
    """Save the config into the para dir"""
    with open(parameter_dir / "parameters_network.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", 'w') as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    """Load the environment"""
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0
    print("Loaded environment variation %d with seed %d" % (variation, config["environment"]["seed"]))
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]
    env.seed(config["environment"]["seed"])
    if config["environment"]["fugitive_policy"] == "heuristic":
        fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    elif config["environment"]["fugitive_policy"] == "a_star":
        fugitive_policy = AStarAdversarialAvoid(env, max_speed=15, cost_coeff=1000)
    else:
        raise ValueError("fugitive_policy should be heuristic or a_star")
    env = PrisonerBlueEnv(env, fugitive_policy)
    """Reset the environment"""
    _, blue_partial_observation = env.reset()

    blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
    filtering_input = copy.deepcopy(env.get_t_d1_d0_vel())

    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    """Load the model"""
    agent_num = env.num_helicopters + env.num_search_parties
    action_dim_per_agent = 2
    total_action_dim = action_dim_per_agent * agent_num
    # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
    filtering_input_dims = [[len(filtering_input)] for i in range(agent_num)]
    obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
    ac_dims=[action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    # hier_high_act_dim = agent_num * env.subpolicy_num
    # hier_low_act_dim = agent_num * env.max_para_num
    # maddpg = DDPGFiltering(
    #                 filtering_model_config = config["train"]["filtering_model_config"],
    #                 filtering_model_path = config["train"]["filtering_model_path"],
    #                 agent_num = agent_num, 
    #                 num_in_pol = blue_observation[0].shape[0], 
    #                 num_out_pol = action_dim_per_agent, 
    #                 num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent), 
    #                 discrete_action = False, 
    #                 gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    maddpg = CTCE_DDPG_Filtering(
                        filtering_model_config = config["train"]["filtering_model_config"],
                        filtering_model_path = config["train"]["filtering_model_path"],
                        agent_num = agent_num, 
                        num_in_pol = blue_observation[0].shape[0] * agent_num, 
                        num_out_pol = total_action_dim, 
                        num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                        discrete_action = False, 
                        gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], filter_lr=config["train"]["filter_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
    
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





    # blue_hier_policy = HierRLBlue(env, maddpg, device)
    """Initialize the buffer"""
    replay_buffer = ReplayBuffer(config["train"]["buffer_size"], agent_num, obs_ac_filter_loc_dims=obs_ac_filter_loc_dims, is_cuda=config["environment"]["cuda"])
    # imgs = []
    # last_t = 0
    # t = 0
    # done = False
    for ep in range(recent_episode, config["train"]["episode_num"]):
        maddpg.prep_rollouts(device=device)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        # print("go into done branch")

        """Start a new episode"""
        _, blue_partial_observation = env.reset()
        blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
        # gnn_obs = env.get_gnn_obs() # gnn_obs[0]: [agent_num (camera, blue agent, etc.), feature_num]
        # gnn_sequence_obs = env.get_gnn_sequence() # gnn_sequence_obs[0]: [seq_len, agent_num (camera, blue agent, etc.), feature_num]
        last_two_detections_vel = env.get_t_d1_d0_vel()
        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
            # gnn_sequence_obs = env.get_gnn_sequence()
            # last_two_detections_vel = env.get_t_d1_d0_vel()
            mlp_input_tensors = torch.Tensor(last_two_detections_vel).unsqueeze(0).to(device)
            # get actions as torch Variables
            torch_agent_actions, filter_outputs = maddpg.step(torch_obs, mlp_input_tensors, explore=True)
            env.set_localized_mus(filter_outputs)
            # err = maddpg.test_loc_est_error(normalized_prisoner_gt_loc=np.array(prisoner_loc)/2428, filtering_input=gnn_input_tensors)
            # print("prisoner_loc est. error = ", err)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5

            _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions)))
            next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            next_last_two_detections_vel = env.get_t_d1_d0_vel()
            # next_k_detections = env.get_last_k_fugitive_detections()
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            rewards = dist_reward + detect_reward
            # print("The dist_reward is: ", dist_reward)
            # print("The detect_reward is: ", detect_reward)

            replay_buffer.push_filter(blue_observation, agent_actions, rewards, next_blue_observation, done, last_two_detections_vel, next_last_two_detections_vel, np.array(prisoner_loc)/2428)

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            last_two_detections_vel = next_last_two_detections_vel
            prisoner_loc = next_prisoner_loc

            # print("blue rewards: ", rewards)
            if ep % config["train"]["video_step"] == 0:
                game_img = env.render('Policy', show=False, fast=True)
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

            for a_i in range(1):
                sample = replay_buffer.sample_filter(config["train"]["batch_size"], to_gpu=config["environment"]["cuda"], norm_rews=True)
                maddpg.update(sample, a_i, train_option="regular", logger=logger)
                # maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep)
  
    #     """update Q func"""
    #     if hier_buffer._n > config["train"]["batch_size"]:
    #         for _ in range(config["train"]["steps_per_update"]):
    #             hier_trainer.update(hier_buffer.sample(config["train"]["batch_size"]), logger=logger)
    return

def split_directions_to_direction_speed(directions):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    search_party_v_limit = 20
    helicopter_v_limit = 127
    for idx in range(len(blue_actions_directions)):
        if idx < 5:
            search_party_direction = blue_actions_directions[idx]
            if np.linalg.norm(search_party_direction) > 1:
                search_party_direction = search_party_direction / np.linalg.norm(search_party_direction)
            search_party_speed = np.maximum(np.linalg.norm(search_party_direction), 1.0) * search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = np.maximum(np.linalg.norm(helicopter_direction), 1.0) * helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel 

def split_to_batch(per_sample, device):
    """per_sample[0]: [[obs(6), agent_actions(6), rewards(6), next_obs(6), dones(6), td(6)], [...], [...], ..., [...]]
       desired_sample: [[[obs0(batch_size)], [obs1(batch_size)], ..., [obs5(batch_size)]], [], [], [], []]"""
    per_sample_np = np.transpose(np.array(per_sample[0]), (1, 2, 0))
    per_sample_np = per_sample_np.tolist()
    item_num = len(per_sample_np)
    agent_num = len(per_sample_np[0])
    for item_idx in range(item_num):
        for agent_idx in range(agent_num):
            per_sample_np[item_idx][agent_idx] = torch.Tensor(np.vstack(per_sample_np[item_idx][agent_idx])).to(device)
    per_sample[0] = per_sample_np
    # obs = []
    # acs = []
    # rews = []
    # next_obs = []
    # dones = []
    # desired_sample = [[], [], [], [], []]
    # batch_size = len(per_sample)
    # agent_num = len(per_sample[0])
    # for item_idx in range(len(desired_sample)):
    #     for agent_idx in range(agent_num):
    #         for batch_idx in range(batch_size):
    #             it_ag_ba = per_sample[batch_idx][item_idx]
    return per_sample

def split_to_batch_filtering(per_sample, device):
    """ 
       per_sample[0]: [[obs(6), agent_actions(6), rewards(6), next_obs(6), dones(6), td(6), filtering_input(1), prisoner_loc(1)], [...], [...], ..., [...]]
       desired_sample: [[[obs0(batch_size)], [obs1(batch_size)], ..., [obs5(batch_size)]], [], [], [], []]
    """
    filtering_inputs = []
    prisoner_locs = []
    batched_agents = []

    for sample in per_sample[0]:
        # obs, agent_actions, rewards, next_obs, dones, td, filtering_input, prisoner_loc = sample
        original_batch = sample[:-2]
        filtering_input = sample[-2]
        prisoner_loc = sample[-1]
        filtering_inputs.append(filtering_input)
        prisoner_locs.append(prisoner_loc)
        batched_agents.append(original_batch)

    per_sample_np = np.array(batched_agents)
    per_sample_np = np.transpose(per_sample_np, (1, 2, 0))
    per_sample_np = per_sample_np.tolist()
    item_num = len(per_sample_np)
    agent_num = len(per_sample_np[0])
    for item_idx in range(item_num):
        for agent_idx in range(agent_num):
            per_sample_np[item_idx][agent_idx] = torch.Tensor(np.vstack(per_sample_np[item_idx][agent_idx])).to(device)
    per_sample[0] = per_sample_np
    filtering_inputs = torch.Tensor(np.vstack(filtering_inputs)).to(device)
    prisoner_locs = torch.Tensor(np.vstack(prisoner_locs)).to(device)
    return per_sample, filtering_inputs, prisoner_locs


if __name__ == '__main__':
    config = config_loader(path="./blue_bc/parameters_training_gnn.yaml")  # load model configuration
    env_config = config_loader(path=config["environment"]["env_config_file"])
    """create base dir"""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("./logs/marl") / timestr
    os.makedirs(base_dir, exist_ok=True)
    """Benchmark Starts Here"""
    # Specify the benchmarking parameters: random seeds, sharing strategies, env type, learning rates, PER parameters
    seeds = [0] # 0, 1, 2, 3, 4, 5
    structures = ["regular"] # "regular", "per", hier
    fugitive_policys = ["a_star"] # "a_star", "heuristic"
    estimators = ["filtering"] # "no_estimator", "linear_estimator", "filtering", "predicting", "ground_truth", "detections", "det_filt", "seq_fc"
    critic_lrs = [0.00025] # 0.0005
    policy_lrs = [0.000125] # 0.00025
    filter_lrs = [0.0001]
    load_checkpoint = False
    if load_checkpoint:
        print("\033[33mYou are loading checkpoint\033[33m")
    else:
        print("\033[33mYou are NOT loading checkpoint\033[33m")

    alpha = [0.3] # 0.1, 0.2, 0.3, 0.4, 0.5
    beta_increment_per_sampling = [0.00001] # 0, 0.00001, 0.00002, 0.00003, 0.00004
    
    for seed in seeds:
        for c_lr in critic_lrs:
            for p_lr in policy_lrs:
                for f_lr in filter_lrs:
                    for structure in structures:
                        for policy in fugitive_policys:
                            for est in estimators:
                                for a in alpha:
                                    for beta_increment in beta_increment_per_sampling:
                                        """Modify the config"""
                                        config["environment"]["seed"] = seed
                                        config["train"]["critic_lr"] = c_lr
                                        config["train"]["policy_lr"] = p_lr
                                        config["train"]["filter_lr"] = f_lr
                                        config["train"]["continue"] = load_checkpoint
                                        config["environment"]["structure"] = structure
                                        config["environment"]["fugitive_policy"] = policy
                                        config["per"]["a"] = a
                                        config["per"]["beta_increment_per_sampling"] = beta_increment
                                        config["environment"]["estimator"] = est
                                        if est == "ground_truth":
                                            env_config["include_fugitive_location_in_blue_obs"] = True
                                        else:
                                            env_config["include_fugitive_location_in_blue_obs"] = False

                                        """create base dir name for each setting"""    
                                        base_dir = Path("./logs/marl") / timestr / (structure+"_"+policy+"_"+est)
                                        config["environment"]["dir_path"] = str(base_dir)

                                        if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] in ["no_estimator", "linear_estimator", "ground_truth", "detections"]:
                                            main_reg_NeLeGt(config, env_config)
                                        elif config["environment"]["structure"] == "regular" and config["environment"]["estimator"] in ["seq_fc"]:
                                            main_reg_NeLeGt_seq(config, env_config)
                                        # if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "linear_estimator":
                                        #     main_reg_linearEstimator(config)
                                        elif config["environment"]["structure"] == "regular" and config["environment"]["estimator"] in ["filtering", "det_filt"]:
                                            main_reg_last_two_filtering(config, env_config)
                                        elif config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "predicting":
                                            main_reg_predicting(config)
                                        # if config["environment"]["structure"] == "regular" and config["environment"]["estimator"] == "ground_truth":
                                        #     main_reg_groundTruth(config)

                                        elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] in ["no_estimator", "linear_estimator", "ground_truth"]:
                                            main_per_NeLeGt(config, env_config)
                                        # if config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "linear_estimator":
                                        #     main_per_linearEstimator(config)
                                        elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "filtering":
                                            main_per_filtering(config, env_config)
                                        elif config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "predicting":
                                            main_per_predicting(config)
                                        # if config["environment"]["structure"] == "per" and config["environment"]["estimator"] == "ground_truth":
                                        #     main_per_groundTruth(config)



                                        # elif config["environment"]["structure"] == "per":
                                        #     main_per(config)
                                        # elif config["environment"]["structure"] == "filter_per":
                                        #     main_filter_per(config, env_config)
                                        # elif config["environment"]["structure"] == "hier":
                                        #     main_hier(config)

                                        else:
                                            raise NotImplementedError