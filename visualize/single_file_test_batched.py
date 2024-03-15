import sys, os
sys.path.append(os.getcwd())

import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch

from torch.utils.data import DataLoader
from datasets.load_datasets import load_datasets, load_dataset, load_dataset_with_config_and_file_path

from utils import get_configs, set_seeds
from visualize.render_utils import plot_mog_heatmap

import matplotlib.pyplot as plt
from heatmap import generate_heatmap_img
import cv2

from utils import save_video
from datasets.convert_to_csv import get_dataframe

def get_probability_grid(nn_output, index):
    pi, mu, sigma = nn_output
    pi = pi.detach().squeeze().cpu().numpy()
    sigma = sigma.detach().squeeze().cpu().numpy()
    mu = mu.detach().squeeze().cpu().numpy()
    # print(mu.shape)
    grid = plot_mog_heatmap(mu[index], sigma[index], pi[index])
    return grid

set_seeds(0)

# model_folder_path = 'logs/hybrid_gnn/20220727-2331'

# model_folder_path = '/nethome/sye40/PrisonerEscape/logs_dgl/gnn/total/20220730-2052'
# model_folder_path = '/nethome/sye40/PrisonerEscape/logs/rrt_fixed/homo_gnn/20220810-1126'
model_folder_path = '/nethome/mnatarajan30/codes/PrisonerEscape/logs/contrastive_gnn/simclr/multistep/20220925-1322'

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = 32

seq_len = config["datasets"]["seq_len"]
num_heads = config["datasets"]["num_heads"]
step_length = config["datasets"]["step_length"]
include_current = config["datasets"]["include_current"]
view = config["datasets"]["view"]
multi_head = config["datasets"]["multi_head"]

# test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# _, test_dataloader = load_datasets(config["datasets"], batch_size)
# test_dataset = load_dataset(config["datasets"], "test_path")

# dataset_path = "/workspace/PrisonerEscape/datasets/test/small_subset"
# dataset_path = "datasets/small_subset"
# dataset_path = "datasets/ilrt_test/gnn_map_0_run_3_RRT"

# dataset_path = "/nethome/sye40/PrisonerEscape/datasets/brandon_test_correct/gnn_map_0_run_4_RRT"

dataset_path = "/data/contrastive_learning_datasets/balance_game_astar/fixed_cams_astar_include_camera_at_start/gnn_map_0_run_100_AStar/"
start = 0
episode_indices, episode_lengths, dataframes, names = [], [], [], []
print(os.listdir(dataset_path))
for file_name in os.listdir(dataset_path):
    names.append(file_name.split("_")[1])
    print(os.path.join(dataset_path, file_name))
    np_file = np.load(os.path.join(dataset_path, file_name), allow_pickle=True)
    episode_lengths.append(len(np_file["red_locations"]))
    episode_indices.append(start + len(np_file["red_locations"]))
    start = start + len(np_file["red_locations"])

    dataframes.append(get_dataframe(np_file))

print(os.listdir(dataset_path))

print(episode_lengths)
print(episode_indices)
test_dataset = load_dataset_with_config_and_file_path(config["datasets"], dataset_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
logprob_stack = []

index = 0  # For filtering

bin_count = 0

heatmap_images = []

prob_vals = []
ll_vals = []

pi_stacked, mu_stacked, sigma_stacked, y_stacked = [], [], [], []

for x_test, y_test in test_dataloader:
    pi, mu, sigma = model.forward(x_test)
    pi_stacked.append(pi)
    mu_stacked.append(mu)
    sigma_stacked.append(sigma)
    y_stacked.append(y_test)

    logprob = model.get_stats(x_test, y_test)
    logprob_stack.append(logprob)
    del logprob

pi_stack = torch.cat(pi_stacked, dim=0)
mu_stack = torch.cat(mu_stacked, dim=0)
sigma_stack = torch.cat(sigma_stacked, dim=0)
y_stacked = torch.cat(y_stacked, dim=0)
logprob_stack = torch.cat(logprob_stack, dim=0)

runs_logprob = logprob_stack.tensor_split(episode_indices, dim=0)[:-1]
runs_pi = pi_stack.tensor_split(episode_indices, dim=0)[:-1]
runs_mu = mu_stack.tensor_split(episode_indices, dim=0)[:-1]
runs_sigma = sigma_stack.tensor_split(episode_indices, dim=0)[:-1]
y_stacked = y_stacked.tensor_split(episode_indices, dim=0)[:-1]

def visualize_single_run(pi, mu, sigma, ys, name):
    heatmap_images = []
    for i in range(len(pi)):
        nn_output = (pi[i], mu[i], sigma[i])
        grid = get_probability_grid(nn_output, index)
        heatmap_img = generate_heatmap_img(grid, true_location=ys[i][index]*2428)
        heatmap_images.append(heatmap_img)
    save_video(heatmap_images, f"visualize/brandon_3/{name}.mp4", fps=15.0)

index = 0 # For filtering
for pi, mu, sigma, ys, lp, df, name in zip(runs_pi, runs_mu, runs_sigma, y_stacked, runs_logprob, dataframes, names):
    # print(lp.shape)
    lp = lp.detach().cpu().numpy()
    df['log_likelihood'] = lp[:, index]
    print(df)
    df.to_csv(f"visualize/brandon_3/{name}.csv")
    visualize_single_run(pi, mu, sigma, ys, name)
# the order of these names is incorrect because of the data file paths are not sorted

# for i in range(len(test_dataset)):
#     x_test, y_test = test_dataset[i]
#     # print(x_test)
#     x_test = [torch.from_numpy(x).to(device).unsqueeze(0) for x in x_test]
#     # x_test = torch.from_numpy(x_test).to(device).unsqueeze(0)
#     # 
    
#     ############## Binary Count
#     nn_output = model.forward(x_test)
#     # print(nn_output[0].shape)

#     val = mix_multinorm_cdf_nn(nn_output, 0, y_test)
#     prob_vals.append(val)
#     if val >= 0.5:
#         bin_count += 1
    

#     ################## Heatmaps
#     # grid = get_probability_grid(nn_output, index)
#     # heatmap_img = generate_heatmap_img(grid, true_location=y_test[index]*2428)
#     # # cv2.imwrite("visualize/ilrt_test/heatmap_{}.png".format(i), heatmap_img)
#     # heatmap_images.append(heatmap_img)

#     ###### Log likelihood
#     y_test = torch.from_numpy(y_test).to(device)
#     logprob = model.get_stats(x_test, y_test)
#     ll_val = logprob[0][index]
#     ll_vals.append(ll_val.cpu().item())

#     print(logprob)
#     # print(val, ll_val.cpu().item())

# save_video(heatmap_images, "visualize/94.mp4", fps=15.0)
# print(bin_count / len(test_dataset))

# import pandas as pd 
# d = {'probabilities': prob_vals, "log likelihoods": ll_vals}
# df = pd.DataFrame(d) 
    
# # saving the dataframe 
# df.to_csv('tmp/new_tests/91_results_new_model.csv') 