import sys, os

sys.path.append(os.getcwd())

import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch

from datasets.dataset import VectorPrisonerDataset, GNNPrisonerDataset
# from datasets.old_gnn_dataset import LSTMGNNSequence
from torch.utils.data import DataLoader
from datasets.load_datasets import load_datasets, load_dataset, load_dataset_with_config_and_file_path

from utils import get_configs, set_seeds
from visualize.render_utils import plot_mog_heatmap

import matplotlib.pyplot as plt
from heatmap import generate_heatmap_img
import cv2

from evaluate.mixture_evaluation import mix_multinorm_cdf_nn
from utils import save_video


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

model_folder_path ='/nethome/mnatarajan30/codes/PrisonerEscape/logs/contrastive_gnn/simclr/multistep/20220925-1322'

config_path = os.path.join(model_folder_path, "config.yaml")
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
batch_size = config["batch_size"]

seq_len = config["datasets"]["seq_len"]
num_heads = config["datasets"]["num_heads"]
step_length = config["datasets"]["step_length"]
include_current = config["datasets"]["include_current"]
view = config["datasets"]["view"]
multi_head = config["datasets"]["multi_head"]

dataset_path = config["datasets"]["test_path"]

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
test_dataset = load_dataset_with_config_and_file_path(config["datasets"], dataset_path)

num_episodes = 2
# Load each episode at once from the test dataset ...
done_locations = test_dataset.done_locations
red_locs = test_dataset.red_locs   # To plot entire rollout of red for the current episode
prev_episode_end_step = 0
curr_episode = 0
detections = test_dataset.detected_location.reshape(-1, 2)
last_detection = [-1, -1]
for i in range(done_locations[num_episodes]):
    episode_end_step = done_locations[curr_episode]
    red_path = red_locs[prev_episode_end_step+1:episode_end_step]
    x_test, y_test = test_dataset[i]
    x_test = [torch.from_numpy(x).to(device).unsqueeze(0) for x in x_test]
    # temp = [x_test[0], x_test[2], x_test[3], x_test[4]]
    ############## Binary Count
    nn_output = model.forward(x_test)
    # nn_output = model.predict(temp)
    # print(nn_output[0].shape)

    val = mix_multinorm_cdf_nn(nn_output, 0, y_test)
    prob_vals.append(val)
    if val >= 0.5:
        bin_count += 1

    ################## Heatmaps
    grid = get_probability_grid(nn_output, index)
    if detections[i][0] != -1:
        last_detection = detections[i] * 2428
    heatmap_img = generate_heatmap_img(grid, true_location=y_test[index] * 2428,
                                       red_path=red_path * 2428,
                                       curr_step=i-prev_episode_end_step+1,
                                       last_detection=last_detection)
    heatmap_images.append(heatmap_img)

    ###### Log likelihood
    y_test = torch.from_numpy(y_test).to(device)
    logprob = model.get_stats(x_test, y_test)[0]
    ll_val = logprob[0][index]
    ll_vals.append(ll_val.cpu().item())

    if i == done_locations[curr_episode]:
        prev_episode_end_step = episode_end_step
        curr_episode += 1
        last_detection = [-1, -1]


save_video(heatmap_images, "visualize/CL_camera_blue.mp4", fps=15.0)
print(bin_count / len(test_dataset))

import pandas as pd

d = {'probabilities': prob_vals, "log likelihoods": ll_vals}
df = pd.DataFrame(d)

# saving the dataframe
df.to_csv('tmp/new_tests/CL_camera_blue_results_new_model.csv')