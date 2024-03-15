import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
from simulator.load_environment import load_environment
from simulator.prisoner_perspective_envs import PrisonerEnv
from simulator.prisoner_env_variations import initialize_prisoner_environment
from heatmap import generate_heatmap_img
import os
from visualize.render_utils import combine_game_heatmap, save_video, plot_mog_heatmap
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid
import numpy as np
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import yaml
from simulator import PrisonerBothEnv, PrisonerBlueEnv, PrisonerEnv
from simulator.gnn_wrapper import PrisonerGNNEnv
from blue_policies.heuristic import BlueHeuristic

def get_probability_grid(nn_output, true_location=None):
    pi, mu, sigma = nn_output
    pi = pi.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    grid = plot_mog_heatmap(mu, sigma, pi)
    return grid

def _create_one_hot_agents(agent_dict, timesteps=1):
    one_hot_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    agents = [agent_dict["num_known_cameras"] + agent_dict["num_unknown_cameras"],
              agent_dict["num_helicopters"], agent_dict["num_search_parties"]]
    a = np.repeat(one_hot_base, agents, axis=0)  # produce [num_agents, 3] (3 for each in one-hot)
    # one_hot = np.repeat(np.expand_dims(a, 0), self.sequence_length, axis=0) # produce [seq_len, num_agents, 3]
    one_hot = np.repeat(np.expand_dims(a, 0), timesteps, axis=0)  # produce [timesteps, num_agents, 3]
    return one_hot

def render(env, config, policy, model, target_timestep, device, seed):
    """
    """
    show = False
    imgs = []

    episode_return = 0.0
    done = False

    # Initialize empty observations
    gnn_obs, blue_obs = env.reset()
    policy.reset()
    policy.init_behavior()

    i = 0
    timestep = 0
    indices = target_timestep // 5
    true_locations = []
    grids = []
    seq_len = 16
    num_agents = env.total_agents_num
    num_feats = gnn_obs[0].shape[-1]
    if config["datasets"]["one_hot_agents"]:
        num_feats += 3
    if config["datasets"]["detected_location"]:
        num_feats += 2
    if config["datasets"]["timestep"]:
        num_feats += 1
    sequence_obs = torch.zeros((1, seq_len, num_agents, num_feats)).to(device)

    agent_dict = {"num_known_cameras": env.num_known_cameras,
                  "num_unknown_cameras": env.num_unknown_cameras,
                  "num_helicopters": env.num_helicopters,
                  "num_search_parties": env.num_search_parties}

    while not done:
        i += 1
        action = policy.predict(blue_obs)
        gnn_obs, blue_obs, reward, done, _ = env.step(action)
        # blue_observation = env.get_blue_observation()
        episode_return += reward

        true_location = np.array(env.prisoner.location)
        true_locations.append(true_location)

        # num_agents_dup = np.array([gnn_obs[3]] * env.total_agents_num)

        # demonstration_tensor = torch.from_numpy(gnn_obs).to(device).float().unsqueeze(0)
        demonstration_tensor = [torch.from_numpy(i).to(device).float().unsqueeze(0) for i in gnn_obs]
        agent_obs = demonstration_tensor[0]

        # Augment blue observation based on the config file:
        if config["datasets"]["detected_location"]:
            detected_bools = agent_obs[:, :, 0]
            detected_location = torch.Tensor(true_location)/2428
            detected_location = detected_location.unsqueeze(0).to(device)
            detected_agent_locs = torch.einsum("ij,ik->ijk", detected_bools, detected_location) # timesteps, agents, 2
            agent_obs = torch.cat((agent_obs, detected_agent_locs), dim=2)

        if config["datasets"]["timestep"]:
            t = torch.tensor([(i-1)/env.max_timesteps]).to(device)
            t = t.unsqueeze(0)
            t = t.unsqueeze(0)
            t = t.repeat(1, num_agents, 1).to(device)
            agent_obs = torch.cat((agent_obs, t), dim=2)

        if config["datasets"]["one_hot_agents"]:
            one_hots = _create_one_hot_agents(agent_dict)
            one_hots = torch.Tensor(one_hots).to(device)

        # need to reshape obervations to be a history of blue observations
        # Remove the oldest obs in the sequence and append the latest blue obs
        sequence_obs = sequence_obs[:, 1:, :]
        agent_obs = agent_obs.unsqueeze(dim=1)
        sequence_obs = torch.cat((sequence_obs, agent_obs), dim=1)
        demonstration_tensor[0] = sequence_obs
        # demonstration_tensor[3] = torch.tensor(num_agents_dup).to(device)

        output = model.predict(demonstration_tensor)
        # Output is a mixture of Gaussians.
        query_location = (output[0][0, indices], output[1][0, indices, :], output[2][0, indices, :])
        max_idx = torch.argmax(query_location[0])
        print("Predicted loc mean: {}, True loc: {}".format(query_location[1][max_idx], true_location/2428))
        grid = get_probability_grid(query_location, true_location)
        grids.append(grid)

        if timestep >= target_timestep:
            heatmap_img = generate_heatmap_img(grids[timestep - target_timestep], true_location=env.prisoner.location)
            game_img = env.render('Policy', show=False, fast=True)
            img = combine_game_heatmap(game_img, heatmap_img)
            imgs.append(img)
        timestep += 1
        if done:
            break

    save_video(imgs, f"visualize/heatmap_{seed}.mp4", fps=5)


def run_single_seed(seed, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_num = 0
    target_timestep = 0

    epsilon = 0.1
    env_path = "simulator/configs/balance_game.yaml"

    env = load_environment(env_path)
    red_policy = AStarAdversarialAvoid(env, cost_coeff=1000)
    env = PrisonerBlueEnv(env, red_policy)

    env = PrisonerGNNEnv(env)
    policy = BlueHeuristic(env, debug=False)

    render(env, config, policy, model, target_timestep, device, seed)


if __name__ == "__main__":
    model_folder_path = '/nethome/mnatarajan30/codes/PrisonerEscape/logs/contrastive_gnn/simclr/multistep/20220925-1322'
    config_path = os.path.join(model_folder_path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    batch_size = config["batch_size"]

    device = config["device"]
    # Load model
    model = configure_model(config).to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
    model.eval()
    seed = 300
    run_single_seed(seed, model, config)

