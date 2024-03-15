import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from functools import wraps
import torch.optim as optim
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torch
import random
import shutil

import pickle
from tqdm import tqdm
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from multi_step_prediction.dataset import PrisonerLocationDataset, create_dataset
# from multi_step_prediction.utils_msp import proba_distribution, calculate_loss, plot_gaussian_heatmap
from torch.utils.data import Dataset, DataLoader
# from multi_step_prediction.dataset import create_dataset, create_dataset_sequence, create_dataset_weighted, create_dataset_sequence_weighted


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):

        super(EncoderRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, hidden_state=None):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            batch_size = x.sorted_indices.size(0)
        else:
            batch_size = x.size(0)
        x = x.to(self.device).float()
        # x = self.fc1(x)
        if hidden_state is None:
            # Initializing the hidden state for the first input with zeros
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

            # Initializing the cell state for the first input with zeros
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        else:
            h0, c0 = hidden_state
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshaping the output
        # out = out[:, -1, :] # Hidden output from all timesteps out[:, -1, :] is the hidden output from the last timestep == hn
        hn = hn.view(-1, hn.shape[-1])
        return hn


class VariationalEncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, z_dim=8, bias=False):

        super(VariationalEncoderRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        # Feature extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU()
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.enc_mean = nn.Linear(hidden_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()
        )

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.prior_mean = nn.Linear(hidden_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()
        )

        # recurrence
        self.rnn = nn.GRU(hidden_dim + hidden_dim, hidden_dim, num_layers, bias, batch_first=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, hidden_state=None):
        all_enc_mean, all_enc_std, all_prior_mean, all_prior_std = [], [], [], []
        batch_size = x.size(0)
        seq_len = x.shape[1]
        x = x.to(self.device).float()
        if hidden_state is None:
            # Initializing the hidden state for the first input with zeros
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        for t in range(seq_len):
            phi_x_t = self.phi_x(x[:, t])

            # Encoder
            enc_t = self.enc(torch.cat((phi_x_t, hidden_state[-1]), dim=-1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # Prior
            prior_t = self.prior(hidden_state[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Recurrence
            _, hidden_state = self.rnn(torch.cat((phi_x_t, phi_z_t), dim=-1).unsqueeze(0), hidden_state)

            # Append mean and std from encoder and prior dist to calculate loss
            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)

            # Return hidden state from last timestep to decoder
        hn = hidden_state.view(-1, hidden_state.shape[-1])
        res = torch.cat((phi_z_t, hn), dim=-1)
        return res, all_enc_mean, all_enc_std, all_prior_mean, all_prior_std

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


class ContrastiveEncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):

        super(ContrastiveEncoderRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim // 2
        self.num_layers = num_layers

        self.lstm_1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, hidden_state=None):
        positive_obs, negative_obs, last_positive_timestep, last_negative_timestep = x

        batch_size = positive_obs.size(0)
        positive_obs = positive_obs.to(self.device).float()
        negative_obs = negative_obs.to(self.device).float()
        last_positive_timestep = last_positive_timestep.to(self.device)
        last_negative_timestep = last_negative_timestep.to(self.device)

        # Assume hidden state is never initialized
        # Initializing the hidden state for the first input with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        # Initializing the cell state for the first input with zeros
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        # Using the same LSTM for both positive and negative observations
        positive_out, (positive_hn, positive_cn) = self.lstm_1(positive_obs, (h0, c0))
        negative_out, (negative_hn, negative_cn) = self.lstm_2(negative_obs, (h0, c0))

        # Concatenate the two hidden embeddings?
        positive_hn = positive_hn.view(-1, positive_hn.shape[-1])
        negative_hn = negative_hn.view(-1, negative_hn.shape[-1])

        return (positive_hn, negative_hn, last_positive_timestep, last_negative_timestep)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class ModifiedContrastiveEncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 target_network=None, use_momentum=True):
        """
        Use an online and a target network for the positive and negative
        observations. The target network is updated as an exponential moving
        average of the online network

        Both the positive and the negative observations are passed into the online
        and the target networks, and two similarity measures are considered

        The output of the online network (for positive and negative) is concat
        and fed into the decoder. At test time, we will not need the target network.

        :param input_dim:
        :param hidden_dim:
        :param num_layers:
        """

        super(ModifiedContrastiveEncoderRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim // 2
        self.num_layers = num_layers

        self.online_encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True)
        self.target_encoder = target_network

        # self.target_encoder.requires_grad_(False)
        self.use_momentum = use_momentum  # Set use_momentum to false during eval/test
        self.beta = 0.99

        self.target_ema_updater = EMA(beta=self.beta)
        # send a mock input tensor to instantiate singleton parameters
        self.forward((torch.randn(2, 8, input_dim, device=self.device), torch.randn(2, 8, input_dim, device=self.device), torch.randn(2, 1, device=self.device), torch.randn(2, 1, device=self.device)))

    @property
    def device(self):
        return next(self.parameters()).device

    @singleton('target_encoder')
    def _get_target_encoder(self):
        if not self.use_momentum:
            target_encoder = copy.deepcopy(self.online_encoder)
        else:
            target_encoder = copy.deepcopy(self.online_encoder)
            # target_encoder = self.target_encoder if self.target_encoder is not None else copy.deepcopy(self.online_encoder)
        target_encoder.requires_grad_(False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, hidden_state=None):
        positive_obs, negative_obs, last_positive_timestep, last_negative_timestep = x

        batch_size = positive_obs.size(0)
        positive_obs = positive_obs.to(self.device).float()
        negative_obs = negative_obs.to(self.device).float()
        last_positive_timestep = last_positive_timestep.to(self.device)
        last_negative_timestep = last_negative_timestep.to(self.device)

        # Assume hidden state is never initialized
        # Initializing the hidden state for the first input with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        # Initializing the cell state for the first input with zeros
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        # Project both positive and negative observations using online and target networks
        online_proj_pos_out, (online_proj_pos_hn, _) = self.online_encoder(positive_obs, (h0, c0))
        online_proj_neg_out, (online_proj_neg_hn, _) = self.online_encoder(negative_obs, (h0, c0))

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_pos_out, (target_proj_pos_hn, _) = target_encoder(positive_obs, (h0, c0))
            target_proj_neg_out, (target_proj_neg_hn, _) = target_encoder(negative_obs, (h0, c0))

        # Concatenate the two hidden embeddings?
        positive_hn = online_proj_pos_hn.view(-1, online_proj_pos_hn.shape[-1])
        negative_hn = online_proj_neg_hn.view(-1, online_proj_neg_hn.shape[-1])
        concat_embed = torch.cat((positive_hn, negative_hn), dim=-1)

        # Contrastive loss
        loss_one = loss_fn(online_proj_pos_hn, target_proj_neg_hn.detach())
        loss_two = loss_fn(online_proj_neg_hn, target_proj_pos_hn.detach())

        loss = loss_one + loss_two

        epsilon = 1e-5
        time_diff = torch.abs(last_negative_timestep - last_positive_timestep) + epsilon
        loss = loss/time_diff * 100

        return (concat_embed, loss.mean())
