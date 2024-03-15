import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(EncoderFC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlin = nn.ReLU()
        self.fc1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = x.to(self.device).float()
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x