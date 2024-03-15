import os
import sys

from matplotlib.streamplot import Grid

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator.prisoner_perspective_envs import PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid
import matplotlib
import torch
import copy
from torch.autograd import Variable
import numpy as np
from blue_bc.maddpg_filtering import MADDPGFiltering, AttentionMADDPGFiltering, MADDPGCommFiltering
from blue_bc.maddpg_shared import MADDPGFilteringShared

matplotlib.use('agg')
import matplotlib.pylab
from utils import save_video
from blue_bc.utils import  blue_obs_type_from_estimator, get_modified_blue_obs
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from enum import Enum, auto
from visualize.render_utils import combine_game_heatmap, plot_mog_heatmap
from heatmap import generate_heatmap_img

class Estimator(Enum):
    DETECTIONS = auto()
    LINEAR_ESTIMATOR = auto()
    NO_DETECTIONS = auto()

def split_directions_to_direction_speed(directions, search_party_v_limit):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    # search_party_v_limit = 20
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

def get_probability_grid(nn_output, true_location=None):
    pi, mu, sigma = nn_output
    pi = pi.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    grid = plot_mog_heatmap(mu[0], sigma[0], pi[0])
    return grid

def run_episode(path):
    config_path = os.path.join(path, 'parameter/parameters_network.yaml')
    
    config = config_loader(path=config_path)  # load model configuration
    config["train"]["filtering_model_path"] = None

    env_config_path = os.path.join(path, 'parameter/parameters_env.yaml')
    env_config = config_loader(env_config_path)

    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0


    start_episode = 0
    end_episode = 1 # 25001
    episode_inc = 1000
    for checkpoint in np.arange(start_episode, end_episode, episode_inc):
        episode_num = 10
        start_seed = 0 
        for ep in range(episode_num):
            seed_now = start_seed + ep
            print("Loaded environment variation %d with seed %d" % (variation, seed_now))
            # set seeds
            np.random.seed(seed_now)
            random.seed(seed_now)
            env = load_environment(env_config)
            env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]
            env.seed(seed_now)
            if config["environment"]["fugitive_policy"] == "heuristic":
                fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
            elif config["environment"]["fugitive_policy"] == "a_star":
                fugitive_policy = AStarAdversarialAvoid(env)
            else:
                raise ValueError("fugitive_policy should be heuristic or a_star")
            env = PrisonerBlueEnv(env, fugitive_policy)
            # """Reset the environment"""
            _, blue_partial_observation = env.reset()
            blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # """Load the model"""
            agent_num = env.num_helicopters + env.num_search_parties
            action_dim_per_agent = 2
            obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
            ac_dims=[action_dim_per_agent for i in range(agent_num)]
            obs_ac_filter_loc_dims = [obs_dims, ac_dims]
            maddpg = MADDPGFiltering(
                                filtering_model_config = config["train"]["filtering_model_config"],
                                filtering_model_path = config["train"]["filtering_model_path"],
                                agent_num = agent_num, 
                                num_in_pol = blue_observation[0].shape[0], 
                                num_out_pol = action_dim_per_agent, 
                                num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                                discrete_action = False, 
                                gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
                
            maddpg.init_from_save(os.path.join(path, 'model','%d.pth'%checkpoint))

            # Run the episode
            maddpg.prep_rollouts(device=device)
            explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - 0) / config["train"]["n_exploration_eps"]
            maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
            maddpg.reset_noise()
            # print("go into done branch")

            # """Start a new episode"""
            _, blue_partial_observation = env.reset()
            blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # gnn_obs = env.get_gnn_obs() # gnn_obs[0]: [agent_num (camera, blue agent, etc.), feature_num]
            gnn_sequence_obs = env.get_gnn_sequence() # gnn_sequence_obs[0]: [seq_len, agent_num (camera, blue agent, etc.), feature_num]
            # prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            t = 0
            imgs = []
            true_locations = []
            grids = []
            done = False
            while not done:
                # # INFO: run episode
                t = t + 1
                torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
                gnn_sequence_obs = env.get_gnn_sequence()
                gnn_input_tensors = [torch.tensor(i).unsqueeze(0).to(device) for i in gnn_sequence_obs]
                # get actions as torch Variables
                torch_agent_actions = maddpg.step(torch_obs, gnn_input_tensors, explore=False)

                # render filtering output
                filtering_output = maddpg.filtering_model(gnn_input_tensors)
                true_location = np.array(env.prisoner.location)
                true_locations.append(true_location)

                grid = get_probability_grid(filtering_output, true_location)
                # grids.append(grid)

                heatmap_img = generate_heatmap_img(grid, true_location=env.prisoner.location)
                game_img = env.render('Policy', show=False, fast=True)
                img = combine_game_heatmap(game_img, heatmap_img)
                imgs.append(img)

                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5

                _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions), env.search_party_speed))
                next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
                blue_observation = next_blue_observation

                # game_img = env.render('Policy', show=False, fast=True)
                # imgs.append(game_img)
                print(t)
            
                save_video(imgs, "visuaulize/episode%d/rl_test_seed%d.mp4"%(checkpoint, seed_now), fps=10)

def run_mlp_filter_episode(path):
    config_path = os.path.join(path, 'parameter/parameters_network.yaml')
    
    config = config_loader(path=config_path)  # load model configuration
    config["train"]["filtering_model_path"] = None

    env_config_path = os.path.join(path, 'parameter/parameters_env.yaml')
    env_config = config_loader(env_config_path)

    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0


    start_episode = 0
    end_episode = 1 # 25001
    episode_inc = 1000
    for checkpoint in np.arange(start_episode, end_episode, episode_inc):
        episode_num = 30
        start_seed = 0 
        for ep in range(episode_num):
            seed_now = start_seed + ep
            print("Loaded environment variation %d with seed %d" % (variation, seed_now))
            # set seeds
            np.random.seed(seed_now)
            random.seed(seed_now)
            env = load_environment(env_config)
            env.seed(seed_now)
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
            action_dim_per_agent = 2 + 0 # env_config["comm_dim"]
            # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
            filtering_input_dims = [[len(filtering_input)] for i in range(agent_num)]
            obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
            ac_dims=[action_dim_per_agent for i in range(agent_num)]
            loc_dims = [len(prisoner_loc) for i in range(agent_num)]
            obs_ac_dims = [obs_dims, ac_dims]
            obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
            # maddpg = AttentionMADDPGFiltering(
            #                     filtering_model_config = config["train"]["filtering_model_config"],
            #                     filtering_model_path = config["train"]["filtering_model_path"],
            #                     agent_num = agent_num, 
            #                     num_in_pol = blue_observation[0].shape[0], 
            #                     num_out_pol = action_dim_per_agent, 
            #                     num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
            #                     discrete_action = False, 
            #                     gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], filter_lr=config["train"]["filter_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
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
            # maddpg = MADDPGFilteringShared(
            #                     env.type_dict,
            #                     filtering_model_config = config["train"]["filtering_model_config"],
            #                     filtering_model_path = config["train"]["filtering_model_path"],
            #                     agent_num = agent_num, 
            #                     num_in_pol = blue_observation[0].shape[0], 
            #                     num_out_pol = action_dim_per_agent, 
            #                     num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
            #                     discrete_action = False, 
            #                     gamma=config["train"]["gamma"], 
            #                     tau=config["train"]["tau"], 
            #                     critic_lr=config["train"]["critic_lr"], 
            #                     policy_lr=config["train"]["policy_lr"], 
            #                     filter_lr=config["train"]["filter_lr"], 
            #                     hidden_dim=config["train"]["hidden_dim"], 
            #                     device=device)          
            maddpg.init_from_save(os.path.join(path, 'model','%d.pth'%checkpoint))
            # Run the episode
            maddpg.prep_rollouts(device=device)
            explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - 0) / config["train"]["n_exploration_eps"]
            maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
            maddpg.reset_noise()
            # print("go into done branch")

            # """Start a new episode"""
            _, blue_partial_observation = env.reset()
            blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # last_two_detections_vel = env.get_t_d1_d0_vel()
            last_two_detections_vel = env.get_t_init_le_timeInterval()
            prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            t = 0
            imgs = []
            true_locations = []
            grids = []
            done = False
            while not done:
                # # INFO: run episode
                t = t + 1
                torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
                # gnn_sequence_obs = env.get_gnn_sequence()
                # last_two_detections_vel = env.get_t_d1_d0_vel()
                last_two_detections_vel = env.get_t_init_le_timeInterval()
                mlp_input_tensors = torch.Tensor(last_two_detections_vel).unsqueeze(0).to(device)
                # get actions as torch Variables
                # torch_agent_actions, torch_agents_comms = maddpg.step(torch_obs, mlp_input_tensors, explore=False)
                torch_agent_actions = maddpg.step(torch_obs, mlp_input_tensors, explore=False)

                # render filtering output
                filtering_output = maddpg.filtering_model(*maddpg.split_filtering_input(mlp_input_tensors))
                true_location = np.array(env.prisoner.location)
                true_locations.append(true_location)

                search_party_locations, helicopter_locations = env.get_blue_locations()

                grid = get_probability_grid(filtering_output, true_location)
                # grids.append(grid)

                heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=filtering_output[1][0].detach().cpu().numpy()*2428)
                game_img = env.render('Policy', show=False, fast=True)
                img = combine_game_heatmap(game_img, heatmap_img)
                imgs.append(img)

                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5

                _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions), env.search_party_speed))
                next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
                blue_observation = next_blue_observation

                # game_img = env.render('Policy', show=False, fast=True)
                # imgs.append(game_img)
                print(t)
            
                save_video(imgs, "visuaulize/episode%d/rl_fine_tune_seed_fromVel_withMu%d.mp4"%(checkpoint, seed_now), fps=10)

def run_hier_filter_episode(path):
    config_path = os.path.join(path, 'parameter/parameters_network.yaml')
    
    config = config_loader(path=config_path)  # load model configuration
    config["train"]["filtering_model_path"] = None

    env_config_path = os.path.join(path, 'parameter/parameters_env.yaml')
    env_config = config_loader(env_config_path)

    blue_obs_type = blue_obs_type_from_estimator(config["environment"]["estimator"], Estimator)    
    device = 'cuda' if config["environment"]["cuda"] else 'cpu'
    epsilon = 0.1
    variation = 0


    start_episode = 0
    end_episode = 1 # 25001
    episode_inc = 1000
    for checkpoint in np.arange(start_episode, end_episode, episode_inc):
        episode_num = 30
        start_seed = 0 
        for ep in range(episode_num):
            seed_now = start_seed + ep
            print("Loaded environment variation %d with seed %d" % (variation, seed_now))
            # set seeds
            np.random.seed(seed_now)
            random.seed(seed_now)
            env = load_environment(env_config)
            env.seed(seed_now)
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
            high_action_dim_per_agent = 8 + env_config["comm_dim"]
            # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
            filtering_input_dims = [[len(filtering_input)] for i in range(agent_num)]
            obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
            ac_dims=[action_dim_per_agent for i in range(agent_num)]
            loc_dims = [len(prisoner_loc) for i in range(agent_num)]
            obs_ac_dims = [obs_dims, ac_dims]
            obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
            # maddpg = AttentionMADDPGFiltering(
            #                     filtering_model_config = config["train"]["filtering_model_config"],
            #                     filtering_model_path = config["train"]["filtering_model_path"],
            #                     agent_num = agent_num, 
            #                     num_in_pol = blue_observation[0].shape[0], 
            #                     num_out_pol = action_dim_per_agent, 
            #                     num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
            #                     discrete_action = False, 
            #                     gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], filter_lr=config["train"]["filter_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
            maddpg = MADDPGCommFiltering(
                                filtering_model_config = config["train"]["filtering_model_config"],
                                filtering_model_path = config["train"]["filtering_model_path"],
                                agent_num = agent_num, 
                                num_in_pol = blue_observation[0].shape[0], 
                                num_out_pol = high_action_dim_per_agent, 
                                num_in_critic = (blue_observation[0].shape[0] + high_action_dim_per_agent) * agent_num, 
                                comm_dim = env_config["comm_dim"],
                                discrete_action = True, 
                                gamma=config["train"]["gamma"], tau=config["train"]["tau"], critic_lr=config["train"]["critic_lr"], policy_lr=config["train"]["policy_lr"], filter_lr=config["train"]["filter_lr"], hidden_dim=config["train"]["hidden_dim"], device=device)
            # maddpg = MADDPGFilteringShared(
            #                     env.type_dict,
            #                     filtering_model_config = config["train"]["filtering_model_config"],
            #                     filtering_model_path = config["train"]["filtering_model_path"],
            #                     agent_num = agent_num, 
            #                     num_in_pol = blue_observation[0].shape[0], 
            #                     num_out_pol = action_dim_per_agent, 
            #                     num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
            #                     discrete_action = False, 
            #                     gamma=config["train"]["gamma"], 
            #                     tau=config["train"]["tau"], 
            #                     critic_lr=config["train"]["critic_lr"], 
            #                     policy_lr=config["train"]["policy_lr"], 
            #                     filter_lr=config["train"]["filter_lr"], 
            #                     hidden_dim=config["train"]["hidden_dim"], 
            #                     device=device)          
            maddpg.init_from_save(os.path.join(path, 'model','%d.pth'%checkpoint))
            # Run the episode
            maddpg.prep_rollouts(device=device)
            explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - 0) / config["train"]["n_exploration_eps"]
            maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
            maddpg.reset_noise()
            # print("go into done branch")

            # """Start a new episode"""
            _, blue_partial_observation = env.reset()
            blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
            # last_two_detections_vel = env.get_t_d1_d0_vel()
            last_two_detections_vel = env.get_t_init_le_timeInterval()
            prisoner_loc = copy.deepcopy(env.get_prisoner_location())
            t = 0
            imgs = []
            true_locations = []
            grids = []
            done = False
            while not done:
                # # INFO: run episode
                t = t + 1
                torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
                # gnn_sequence_obs = env.get_gnn_sequence()
                # last_two_detections_vel = env.get_t_d1_d0_vel()
                last_two_detections_vel = env.get_t_init_le_timeInterval()
                mlp_input_tensors = torch.Tensor(last_two_detections_vel).unsqueeze(0).to(device)
                # get actions as torch Variables
                # torch_agent_actions, torch_agents_comms = maddpg.step(torch_obs, mlp_input_tensors, explore=False)
                torch_agent_actions = maddpg.step(torch_obs, mlp_input_tensors, explore=False)

                # render filtering output
                filtering_output = maddpg.filtering_model(*maddpg.split_filtering_input(mlp_input_tensors))
                true_location = np.array(env.prisoner.location)
                true_locations.append(true_location)

                search_party_locations, helicopter_locations = env.get_blue_locations()

                grid = get_probability_grid(filtering_output, true_location)
                # grids.append(grid)

                heatmap_img = generate_heatmap_img(grid, sigma=5, true_location=env.prisoner.location, sp_locations=search_party_locations, hc_locations=helicopter_locations, mu_locations=filtering_output[1][0].detach().cpu().numpy()*2428)
                game_img = env.render('Policy', show=False, fast=True)
                img = combine_game_heatmap(game_img, heatmap_img)
                imgs.append(img)

                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5

                _, next_blue_partial_observation, detect_reward, dist_reward, done, _ = env.step(split_directions_to_direction_speed(np.concatenate(agent_actions), env.search_party_speed))
                next_blue_observation = get_modified_blue_obs(env, blue_obs_type, Estimator)
                blue_observation = next_blue_observation

                # game_img = env.render('Policy', show=False, fast=True)
                # imgs.append(game_img)
                print(t)
            
                save_video(imgs, "visuaulize/episode%d/rl_fine_tune_seed_fromVel_withMu%d.mp4"%(checkpoint, seed_now), fps=10)

if __name__ == "__main__":
    path = "/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/marl/20230107-163820/hier_a_star_filtering"
    run_hier_filter_episode(path)