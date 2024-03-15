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
from datasets.load_datasets import load_datasets, load_dataset

from utils import get_configs, set_seeds

import matplotlib.pyplot as plt
import cv2

from visualize.render_utils import plot_gaussian_heatmap
from heatmap import generate_heatmap_img

# model_folder_path = "/star-data/logs/cvae/20220720-1647"
# model_folder_path = "/star-data/sye40/logs/cvae/20220721-1302"
# model_folder_path = '/star-data/sye40/logs/gmm_cvae_mixture/20220722-0335'
# model_folder_path = '/star-data/sye40/logs/gmm_cvae_mixture/20220722-0400'
model_folder_path = '/star-data/sye40/logs/gmm_cvae_mixture/20220722-1313'

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

# test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# _, test_dataloader = load_datasets(config["datasets"], batch_size)
test_dataset = load_dataset(config["datasets"], "test_path")

device = config["device"]
# Load model
model = configure_model(config).to(device)

model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
logprob_stack = []

index = 1

for i in range(1, 1000, 50):
    # x_test, y_test = test_dataset[i]
    # x_test = torch.from_numpy(x_test).to(device).unsqueeze(0)
    # # y_test = torch.from_numpy(y_test).to(device) * 2428
    # # print(y_test)
    # locs = model.predict(x_test, num_samples=500) * 2428
    # a = locs.squeeze().cpu().detach().numpy()
    # print(a)
    # plt.figure()
    # plt.plot(a[:, index, 0], a[:, index, 1], "r.")
    # plt.plot(y_test[index][0]*2428, y_test[index][1]*2428, "b.")
    # # set xlim and ylim
    # plt.xlim(0, 2428)
    # plt.ylim(0, 2428)
    # plt.savefig("visualize/mixture_gaussian/test_" + str(i) + ".png")
    # plt.close()

    x_test, y_test = test_dataset[i]
    x_test = torch.from_numpy(x_test).to(device).unsqueeze(0)
    # y_test = torch.from_numpy(y_test).to(device) * 2428
    nn_output = model.predict(x_test, num_samples=10)
    
    nn_output = nn_output[1]
    mu = nn_output.mean.cpu().detach().view(13, 2)
    std = nn_output.stddev.cpu().detach().view(13, 2)
    grid = plot_gaussian_heatmap(mu[index, :].numpy(), mu[index,:].numpy())
    heatmap_img = generate_heatmap_img(grid, true_location=y_test[index]*2428)
    cv2.imwrite("visualize/heatmap_2/heatmap_{}.png".format(i), heatmap_img)