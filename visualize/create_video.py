from simulator.prisoner_env_variations import initialize_prisoner_environment
import numpy as np
import torch
from simulator.prisoner_batch_wrapper import PrisonerBatchEnv
from fugitive_policies.heuristic import HeuristicPolicy
from simulator.prisoner_env import PrisonerEnv
from simulator.prisoner_env_variations import initialize_prisoner_environment
from heatmap import generate_heatmap_img
import os
from visualize.render_utils import combine_game_heatmap, save_video, plot_mog_heatmap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import yaml

def get_probability_grid(nn_output, true_location=None):
    pi, mu, sigma = nn_output
    pi = pi.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    grid = plot_mog_heatmap(mu[0], sigma[0], pi[0])
    return grid

def render(env, policy, model, target_timestep, device, seed):
    """
    """

    show = False
    imgs = []

    episode_return = 0.0
    done = False

    # Initialize empty observations
    observation_space = env.blue_observation_space
    observation = env.reset()
    i = 0
    timestep = 0
    indices = target_timestep // 5
    true_locations = []
    grids = []

    while not done:
        i += 1
        fugitive_observation = env.get_fugitive_observation()
        action = policy.predict(fugitive_observation)
        blue_observation, reward, done, _ = env.step(action[0])
        # blue_observation = env.get_blue_observation()
        episode_return += reward

        demonstration_tensor = torch.from_numpy(blue_observation).to(device).float().unsqueeze(0)
        # print(demonstration_tensor.shape)
        output = model(demonstration_tensor)
        # print(output.shape)
        # print(output[1].shape)
        query_location = (output[0], output[1][:, :, indices:indices+2], output[2][:, :, indices:indices+2])
        true_location = np.array(env.prisoner.location)
        true_locations.append(true_location)

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

def run_single_seed(seed, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_num = 0
    seq_len=16
    target_timestep = 30

    epsilon = 0.1
    env = initialize_prisoner_environment(map_num, observation_step_type="Blue", seed=seed)

    heuristic = HeuristicPolicy(env, epsilon=epsilon)

    env = PrisonerBatchEnv(env, batch_size=seq_len)
    policy = HeuristicPolicy(env)
    # policy = HeuristicPolicy(env, random_mountain=True, epsilon=epsilon)
    # policy = HeuristicPolicy(env, random_mountain=False, mountain_travel="optimal")
    
    render(env, policy, model, target_timestep, device, seed)
    # for key in stats:
    #     print(key, np.mean(stats[key]), np.std(stats[key]))

    # return stats


if __name__ == "__main__":
    # model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220602-0157'
    # model_folder_path = '/nethome/sye40/PrisonerEscape/logs/connected/20220602-2339'
    model_folder_path = '/nethome/sye40/PrisonerEscape/logs/filtering/20220603-0053'
    config_path = os.path.join(model_folder_path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    batch_size = config["batch_size"]

    # Configure dataloaders
    _, test_dataloader = load_datasets(config["datasets"], batch_size)

    device = config["device"]
    # Load model
    model = configure_model(config["model"], config["datasets"]["num_heads"]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder_path, "epoch_100.pth")))

    seed = 5
    run_single_seed(seed, model)

    # total_stats = []
    # for seed in range(1, 10):
    #     stats = run_single_seed(seed, path, render=True)
    #     total_stats.append(stats)

    # # initialize a dictionary of {statistic: list}
    # accum_stats = {}
    # stat_keys = total_stats[0].keys()
    # for key in stat_keys:
    #     accum_stats[key] = []

    # # accumulate each timestep into the dictionary
    # for stat in total_stats:
    #     for key in stat_keys:
    #         accum_stats[key].extend(stat[key])

    # print("TOTAL STATS:")
    # for key in accum_stats:
    #     print(key, np.mean(accum_stats[key]), np.std(accum_stats[key]))

    # # compute binary counts
    # prob = np.array(accum_stats["probability"])
    # print((prob >= 0.5).sum(), len(prob), (prob >= 0.5).sum()/len(prob))


    
    