import numpy as np
import torch
from torch import nn
import pdb
import copy



class joint_traj_grader(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.grader_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.grader_out_fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, traj):
        output, (h_n, c_n) = self.grader_lstm(traj)
        grade = self.grader_out_fc(h_n[0])
        return grade
    
    def cal_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
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
        logits = self.grader_out_fc(traj)
        grades = nn.functional.gumbel_softmax(logits, hard=True) @ sel_vec
        grade = -torch.sum(grades, dim=1)
        return grade
    
    def compute_loss(self, traj, gt_cummulative_rew, loss_func):
        est_cummulative_rew = self.forward(traj)
        loss = loss_func(est_cummulative_rew, gt_cummulative_rew)
        return loss