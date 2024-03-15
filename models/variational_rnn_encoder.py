import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models.

Source Code adapted from: https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py
"""


class VariationalRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, num_layers=1, bias=False):

        super(VariationalRNNEncoder, self).__init__()

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
        # print(x)
        batch_size = x.size(0)
        seq_len = x.shape[1]
        x = x.to(self.device).float()

        if hidden_state is None:
            # initializing hidden state to zeros
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
        return res #, all_enc_mean, all_enc_std, all_prior_mean, all_prior_std

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)