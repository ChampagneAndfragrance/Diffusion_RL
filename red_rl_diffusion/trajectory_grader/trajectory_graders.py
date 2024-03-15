from torch.utils.tensorboard import SummaryWriter
# import wandb
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffuser.models.encoder import EncoderRNN 

class TrajGraderNet(nn.Module):
    def __init__(self, dinput_dim, tinput_dim, hidden_dim, out_dim, constrain_out=True, nonlin=F.relu):
        super(TrajGraderNet, self).__init__()

        self.detection_encoder = EncoderRNN(dinput_dim, hidden_dim)
        self.traj_encoder = TrajEncoder(tinput_dim, hidden_dim)
        self.grader_decoder = GraderDecoder(input_dim=2*hidden_dim, out_dim=out_dim, hidden_dim=hidden_dim)

    def forward(self, detections, traj):
        detection_embeddings = self.detection_encoder(detections)
        traj_embeddings = self.traj_encoder(traj)
        embeddings = torch.cat([detection_embeddings, traj_embeddings], dim=-1)
        est_rew = self.grader_decoder(embeddings)
        return est_rew

    def compute_loss(self, det, traj, y, loss_func=nn.MSELoss()):
        y_pred = self.forward(det, traj)
        loss = loss_func(y_pred, y)
        return loss
    
class GraderDecoder(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(GraderDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, nonlin=F.relu):
        super(TrajEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        return h3
    

class Grader_FC(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Grader_FC, self).__init__()

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.fc4.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            # self.out_fn = lambda x: x
            raise NotImplementedError

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        h1 = self.nonlin(self.fc1((X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out

    def compute_loss(self, x, y, loss_func=nn.MSELoss()):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss
    
class joint_traj_grader_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlin=torch.nn.ReLU()) -> None:
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.nonlin = nonlin
        self.grader_in_fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.grader_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.grader_out_fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, traj):
        traj = (self.grader_in_fc(traj))
        output, (h_n, c_n) = self.grader_lstm(traj)
        grade = self.grader_out_fc(h_n[0])
        return grade
    
    def compute_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
        return loss
    
class joint_traj_grader_sum(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlin=torch.nn.ReLU()) -> None:
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.nonlin = nonlin

        self.grader_in_fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.grader_fc = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.grader_out_fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, traj):
        traj = self.nonlin(self.grader_in_fc(traj))
        traj = self.nonlin(self.grader_fc(traj))
        grades = self.grader_out_fc(traj)
        grade = torch.sum(grades, dim=1)
        return grade
    
    def compute_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
        return loss

class joint_traj_grader_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlin=torch.nn.ReLU()) -> None:
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.nonlin = nonlin

        self.grader_in_fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.grader_fc = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.grader_out_fc = nn.Linear(in_features=self.hidden_size, out_features=2)

    def forward(self, traj):
        sel_vec = torch.Tensor([[0], [1]])
        traj = self.nonlin(self.grader_in_fc(traj))
        traj = self.nonlin(self.grader_fc(traj))
        logits = (self.grader_out_fc(traj))
        grades = nn.functional.gumbel_softmax(logits, hard=True) @ sel_vec
        grade = -torch.sum(grades, dim=1)
        return grade
    
    def compute_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
        return loss
    
class joint_traj_grader_cnn(nn.Module):
    def __init__(self, input_channel_dim, input_dim, hidden_dim, nonlin=torch.nn.ReLU()) -> None:
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.input_channel_dim = input_channel_dim

        self.nonlin = nonlin

        self.conv1 = nn.Conv1d(in_channels=self.input_channel_dim, out_channels=4*self.input_channel_dim, kernel_size=10)
        self.downsample1 = nn.AvgPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=4*self.input_channel_dim, out_channels=self.input_channel_dim, kernel_size=3)
        self.downsample2 = nn.AvgPool1d(kernel_size=2)

        self.grader_out_fc = nn.Linear(in_features=22, out_features=1)

    def forward(self, traj):
        traj = self.nonlin(self.downsample1(self.conv1(traj)))
        traj = self.nonlin(self.downsample2(self.conv2(traj)))
        grade = self.grader_out_fc(traj.view(traj.shape[0],-1))
        return grade
    
    def compute_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
        return loss