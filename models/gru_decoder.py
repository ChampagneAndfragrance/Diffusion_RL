import sys, os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.distributions import Normal
import math
import random

from torch import Tensor
from typing import Callable, Dict, List, Tuple, TypeVar, Union
import torch.distributions as D
# from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch, insert_dim, NoNorm


import math
import torch
from torch.distributions import Categorical

from models.utils import log_prob

from models.decoder import MixtureDensityDecoder

class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gaussians, num_layers):
        super(GRUDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.mixture_head =MixtureDensityDecoder(
            input_dim=hidden_dim,
            output_dim=2, # output dimension is always 2
            num_gaussians=num_gaussians,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_encoding, hidden_state):
        """ input_encoding: size of (B x Seq) """
        
        batch_size = x.size(0)
        x = x.to(self.device).float()
        if hidden_state is None:
            # Initializing the hidden state for the first input with zeros
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

            # Initializing the cell state for the first input with zeros
            # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        else:
            h0, c0 = hidden_state
        out, hn = self.gru(x, h0)

        # Reshaping the output
        # out = out[:, -1, :] # Hidden output from all timesteps out[:, -1, :] is the hidden output from the last timestep == hn
        hn = hn.view(-1, hn.shape[-1])
        return hn

        # zx = torch.cat([x_encoded, z], dim=1)
        # initial_h = self.relu(self.initial_h_layer(zx))

        # # B x seq_len x 2
        # init_state = -1 * torch.ones((x_encoded.shape[0], 2), device=self.device)
        # input_ = torch.cat([zx, init_state], dim=1)
        # h = initial_h

        # res = []
        # for _ in range(horizon):
        #     h = self.rnn_cell(input_, h)
        #     next_state = self.rnn_to_state(h)
        #     res.append(next_state)
        #     input_ = torch.cat([zx, next_state], dim=1)
        
        # res = torch.stack(res, dim=1)
        # return res