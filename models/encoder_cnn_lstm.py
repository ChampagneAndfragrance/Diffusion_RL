import torch.nn as nn
import torch

import sys, os
sys.path.append(os.getcwd())
import torch
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, lstm_encoder, cnn_encoder, lstm_out, cnn_out, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm_encoder = lstm_encoder
        self.cnn_encoder = cnn_encoder

        self.linear = nn.Linear(lstm_out + cnn_out, hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        blue_obs, image = x
        blue_obs = blue_obs.to(self.device).float()
        image = image.to(self.device).float().unsqueeze(1)

        lstm_out = self.lstm_encoder(blue_obs)
        cnn_out = self.cnn_encoder(image)

        res = self.linear(torch.cat((lstm_out, cnn_out), dim=1))
        return res 