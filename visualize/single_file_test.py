import sys, os
sys.path.append(os.getcwd())
import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch
import pandas as pd
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
from collections import defaultdict

def get_probability_grid(nn_output, index):
    pi, mu, sigma = nn_output
    pi = pi.detach().squeeze().cpu().numpy()
    sigma = sigma.detach().squeeze().cpu().numpy()
    mu = mu.detach().squeeze().cpu().numpy()
    # print(mu.shape)
    grid = plot_mog_heatmap(mu[index], sigma[index], pi[index])
    return grid

set_seeds(0)


model_folder_path = '/data/manisha/prisoner_logs/seed_5/7_detects/gnn/20221012-2351'

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

index = 0 # For filtering

bin_count = 0

heatmap_images = []

prob_vals = []
ll_vals = []
ade_vals = []
dist_thresh_vals = []
one_sigma_vals = []
two_sigma_vals = []
three_sigma_vals = []

test_dataset = load_dataset_with_config_and_file_path(config["datasets"], dataset_path)

num_episodes = 2
done_locations = test_dataset.done_locations
red_locs = test_dataset.red_locs
prev_episode_end_step = 0
curr_episode = 0
detections = test_dataset.detected_location.reshape(-1, 2)
last_detection = [-1, -1]

total_one_sigma, total_two_sigma, total_three_sigma = 0, 0, 0

for i in range(done_locations[num_episodes]):
    episode_end_step = done_locations[curr_episode]
    red_path = red_locs[prev_episode_end_step+1: episode_end_step]
    x_test, y_test = test_dataset[i]
    x_test = [torch.from_numpy(x).to(device).unsqueeze(0) for x in x_test]
    # x_test = torch.from_numpy(x_test).to(device).unsqueeze(0)

    nn_output = model.forward(x_test)

    ################## Heatmaps
    grid = get_probability_grid(nn_output, index)
    if detections[i][0] != -1:
        last_detection = detections[i] * 2428
    heatmap_img = generate_heatmap_img(grid, true_location=y_test[index]*2428,
                                       red_path=red_path*2428,
                                       curr_step=i-prev_episode_end_step+1,
                                       last_detection=last_detection)
    heatmap_images.append(heatmap_img)

    # ###### Log likelihood
    # y_test = torch.from_numpy(y_test).to(device)
    # logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = model.decoder.get_stats(nn_output=nn_output, red_locs=y_test)
    # total_one_sigma += one_sigma
    # total_two_sigma += two_sigma
    # total_three_sigma += three_sigma
    # print(f"i:{i}, logprob:{logprob[:,0].detach().cpu().numpy()}, ade:{ade[:,0].detach().cpu().numpy()}, dist_thresh:{dist_thresh[:,0].detach().cpu().numpy()}")
    # print("---------------------------------")

    # ll_vals.append(logprob.detach().cpu().numpy())
    # ade_vals.append(ade.detach().cpu().numpy())
    # dist_thresh_vals.append(dist_thresh.detach().cpu().numpy())
    # one_sigma_vals.append(one_sigma.detach().cpu().numpy())
    # two_sigma_vals.append(two_sigma.detach().cpu().numpy())
    # three_sigma_vals.append(three_sigma.detach().cpu().numpy())

    if i == done_locations[curr_episode]:
        prev_episode_end_step = episode_end_step
        curr_episode += 1
        last_detection = [-1, -1]

save_video(heatmap_images, "visualize/gnn_policy_7_time_cam_blue.mp4", fps=15.0)

# print(total_one_sigma/(i+1), total_two_sigma/(i+1), total_three_sigma/(i+1))
# # ESV values (difference from ideal Gaussian)
# print(total_one_sigma/(i+1) - 0.39, total_two_sigma/(i+1) - 0.86, total_three_sigma/(i+1) - 0.99)


# ll_vals = np.concatenate(ll_vals, axis=0)
# ade_vals = np.concatenate(ade_vals, axis=0)
# dist_thresh_vals = np.concatenate(dist_thresh_vals, axis=0)
# one_sigma_vals = np.concatenate(one_sigma_vals, axis=0)
# two_sigma_vals = np.concatenate(two_sigma_vals, axis=0)
# three_sigma_vals = np.concatenate(three_sigma_vals, axis=0)
#
# d = defaultdict()
# d['steps'] = np.arange(i+1)
# for i in range(13):
#     d['ll_timestep_{}'.format(i*5)] = ll_vals[:, i]
#     d['ade_timestep_{}'.format(i*5)] = ade_vals[:, i]
#     d['dist_thresh_timestep_{}'.format(i*5)] = dist_thresh_vals[:, i]
#     d['one_sigma_timestep_{}'.format(i*5)] = one_sigma_vals[:, i]
#     d['two_sigma_timestep_{}'.format(i*5)] = two_sigma_vals[:, i]
#     d['three_sigma_timestep_{}'.format(i*5)] = three_sigma_vals[:, i]
# df = pd.DataFrame(d)
# #
# # # saving the dataframe
# # Store eval metrics as a csv file in the same logdir as the model
# # df.to_csv(model_folder_path + 'eval_metrics.csv')
# df.to_csv('visualize/eval_metrics.csv')