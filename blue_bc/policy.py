from turtle import forward
import numpy as np
import torch
from torch import embedding, nn
import torch.nn.functional as F
# from blue_bc.utils import gumbel_softmax_soft_hard
from models.encoders import EncoderRNN
import dgl
import dgl.nn as dglnn
from models.gnn.gnn import fully_connected, fully_connected_include_self, central_graph
from itertools import combinations
from dgl.nn import AvgPooling
from bc_utils import build_mlp, build_mlp_hidden_layers, reparameterize, reparameterize_sigmoid, evaluate_lop_pi, evaluate_lop_pi_para, atanh, calculate_log_pi


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])) # initialized as all zero

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))


class LSTMMDNBluePolicy(nn.Module):

    def __init__(self, num_features, num_actions, hidden_dims, mdn_num, hidden_act_func, output_func, device):
        super(LSTMMDNBluePolicy, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.mdn_num = mdn_num # m
        self.output_dim = num_actions # c
        self.lstm_layer_num = 1
        self.eps = 1e-8

        """MLP layer"""
        self.mlp0 = nn.Linear(num_features, self.hidden_dims[0])
        self.nonlinear0 = hidden_act_func
        """LSTM layer"""
        self.lstm1 = nn.LSTM(input_size=self.hidden_dims[0], hidden_size=self.hidden_dims[1], batch_first=True)
        self.nonlinear1 = hidden_act_func
        """MLP layer to generate alpha, sigma, mu"""
        self.alpha_layer = nn.Linear(self.hidden_dims[1], mdn_num)
        self.sigma_layer = nn.Linear(self.hidden_dims[1], self.output_dim * mdn_num)
        self.mu_layer = nn.Linear(self.hidden_dims[1], self.output_dim * mdn_num)
        """output layer"""
        self.output_layer = output_func
        """utils layer"""
        self.softmax = nn.Softmax(dim=2)


        # # self.h1 = nn.Linear(hidden_dim, hidden_dim)
        # self.out_heading = nn.Linear(hidden_dim, 1)
        # self.out_speed = nn.Linear(hidden_dim, 1)

        # self.embeddings_num = hidden_dim
        # self.alpha_layer = nn.Linear(self.embeddings_num, mdn_num)
        # self.sigma_layer = nn.Linear(self.embeddings_num, mdn_num)
        # self.mu_layer = nn.Linear(self.embeddings_num, self.output_dim * mdn_num)

        # self.nonlinear_relu = nn.ReLU()
        # self.nonlinear_tanh = nn.Tanh()
        # self.nonlinear_sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=2)
    def sample(self, x):
        x = x.unsqueeze(0)
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        alpha = alpha.squeeze()
        a = torch.tensor(range(self.mdn_num))
        # p = torch.tensor([0.1, 0.1, 0.1, 0.7])
        i = alpha.multinomial(num_samples=1, replacement=True)
        log_sigma_each_gs = log_sigma[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        reparameterize(mu_each_gs, log_sigma_each_gs)
        return 

    def forward(self, x):
        x = x.unsqueeze(0)
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        if alpha.numel() != 1:
            alpha = alpha.squeeze()
            i = alpha.multinomial(num_samples=1, replacement=True)
            mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
        else:
            mu_each_gs = mu.squeeze()
        return torch.tanh(mu_each_gs)

    def alpha_sigma_mu(self, x):
        # input of LSTM: tensor of shape (L, N, H_{in}) when batch_first is false, otherwise (N, L, H_{in})
        # output of LSTM: tensor of shape (L, N, D * H_{out}) when batch_first is false, otherwise (N, L, D * H_{out})
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        """pass into first MLP layer"""
        x = self.mlp0(x)
        x = self.nonlinear0(x)
        """initialize hidden states and cell states"""
        h0 = torch.zeros(self.lstm_layer_num, batch_size, self.hidden_dims[1]).requires_grad_().to(self.device)
        c0 = torch.zeros(self.lstm_layer_num, batch_size, self.hidden_dims[1]).requires_grad_().to(self.device)
        """pass into lstm layer"""
        hiddens, (hn, _) = self.lstm1(x, (h0, c0)) # hiddens: (N, L, D * H_{out}), L is the sequence length, batch first
        hiddens = self.nonlinear1(hiddens[:,seq_len-1:seq_len,:]) # (N, L, D * H_{out})
        """pass into MDN parameter calculation layers"""
        alpha = self.softmax(self.alpha_layer(hiddens)) # (N, L=1, m=5)
        log_sigma = self.sigma_layer(hiddens) # (N, L=1, c * m = 12 * 5)
        mu = self.mu_layer(hiddens) # (N, L=1, c * m = 12 * 5)
        return alpha, log_sigma, mu

    def evaluate_action_prob(self, x, y):
        alpha, log_sigma, mu = self.alpha_sigma_mu(x)
        sum_all_log_gs = torch.Tensor(np.array([0])).to(self.device)
        sum_all_gs = torch.Tensor(np.array([self.eps])).to(self.device)
        for i in range(self.mdn_num):
            alpha_each_gs = alpha[:,:,i]
            log_sigma_each_gs = log_sigma[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
            mu_each_gs = mu[:,:,self.output_dim*i:self.output_dim*(i+1)].squeeze()
            """noises should obey N(mu=0, sigma=1)"""
            noises = (atanh(y) - mu_each_gs) / (log_sigma_each_gs.exp() + self.eps)
            log_p_noises_standard_gs = calculate_log_pi(log_sigma_each_gs, noises, y)
            # sum_all_gs = sum_all_gs + alpha_each_gs * log_p_noises_standard_gs.exp()
            sum_all_log_gs = sum_all_log_gs + alpha_each_gs * log_p_noises_standard_gs
        neg_ln_mdn = -sum_all_log_gs.mean()
        mu_average = mu.mean()
        log_sigma_average = log_sigma.mean()
        # print("sum_all_gs_max = ", sum_all_gs.max())
        # neg_ln_mdn = -sum_all_gs.log().mean()
        return neg_ln_mdn, mu_average, log_sigma_average

    def mdn_gaussian_distribution(self, alpha, sigma, mu, y):
        prob = alpha * ((1.0 / (torch.pow(2 * torch.pi, torch.Tensor(np.array(self.output_dim / 2.0)).to(self.device)) * sigma))) * torch.exp(-torch.pow(torch.linalg.norm(y - mu, dim=1, keepdim=True), 2) / (2 * torch.pow(sigma, 2)))
        return prob

    def evaluate_loss(self, x, y):
        loss, mu_average, log_sigma_average = self.evaluate_action_prob(x, y)
        # loss = torch.mean(neg_ln_mdn)
        stats_dict = dict(neglogp=loss.item(), mu_average=mu_average, log_sigma_average=log_sigma_average)
        return loss, stats_dict

    def predict(self, x, deterministic=None):
        obs = torch.from_numpy(x).float().unsqueeze(0).to("cuda")
        return self.alpha_sigma_mu(obs).cpu().detach().numpy()

# class HighLevelPolicy(nn.Module):

#     def __init__(self, state_shape, agent_num, subpolicy_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.Tanh()):
#         super().__init__()
#         self.agent_num = agent_num
#         self.subpolicy_shape = subpolicy_shape
#         self.para_shape = para_shape
#         self.net = build_mlp_hidden_layers(
#             input_dim=state_shape[0],
#             hidden_units=hidden_units,
#             hidden_activation=hidden_activation
#         )
#         self.subpolicy_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
#         self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
#         self.para_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
#         self.para_fc2 = nn.Linear(hidden_units[-1], para_shape[0] * agent_num)
#         self.subpolicy_output = F.gumbel_softmax
#         self.para_output = nn.Sigmoid()
#         # self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])) # initialized as all zero

#     def forward(self, states):
#         x = self.net(states)
#         subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
#         subpolicy = self.subpolicy_output(subpolicy, tau=1, hard=True, eps=1e-10, dim=-1)
#         # subpolicy
#         paras = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
#         paras = self.para_output(paras)
#         # x = self.output(x)
#         print("subpolicy = ", subpolicy)
#         print("paras = ", paras)
#         return subpolicy, paras


class HighLevelPolicyIndependent(nn.Module):

    def __init__(self, state_shape, subpolicy_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.ReLU()):
        super().__init__()
        self.subpolicy_shape = subpolicy_shape
        self.para_shape = para_shape
        self.net = build_mlp_hidden_layers(
            input_dim=state_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.subpolicy_fc = build_mlp(input_dim=hidden_units[-1], output_dim=subpolicy_shape[0], 
                                      hidden_units=(hidden_units[-1],hidden_units[-1]), hidden_activation=hidden_activation, output_activation=None)
        # self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
        self.para_fc = build_mlp(input_dim=hidden_units[-1], output_dim=para_shape[0], 
                                 hidden_units=(hidden_units[-1],hidden_units[-1]), hidden_activation=hidden_activation, output_activation=None)
        # self.para_fc2 = nn.Linear(hidden_units[-1], para_shape[0] * agent_num)
        self.subpolicy_output = F.gumbel_softmax
        self.para_output = nn.Sigmoid()
        # self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])) # initialized as all zero

    def forward(self, states):
        x = self.net(states)
        subpolicy = (self.subpolicy_fc(x)).view(-1, self.subpolicy_shape[0])
        subpolicy = self.subpolicy_output(subpolicy, tau=1, hard=True, eps=1e-10, dim=-1)
        # subpolicy
        paras = (self.para_fc(x)).view(-1, self.para_shape[0])
        paras = self.para_output(paras)
        # x = self.output(x)
        print("subpolicy = ", subpolicy)
        print("paras = ", paras)
        return subpolicy, paras

class HighLevelPolicy(nn.Module):

    def __init__(self, state_shape, subpolicy_shape, hidden_units=(64, 64), hidden_activation=nn.ReLU()):
        super().__init__()
        self.subpolicy_shape = subpolicy_shape
        # self.net = build_mlp_hidden_layers(
        #     input_dim=state_shape[0],
        #     hidden_units=hidden_units,
        #     hidden_activation=hidden_activation
        # )
        self.subpolicy_fc = build_mlp(input_dim=state_shape[0], output_dim=subpolicy_shape[0], 
                                      hidden_units=hidden_units, hidden_activation=hidden_activation, output_activation=None)
        # self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
        self.subpolicy_output = F.gumbel_softmax

    def forward(self, states):
        # x = self.net(states)
        subpolicy = (self.subpolicy_fc(states)).view(-1, self.subpolicy_shape[0])
        # print("states_max = ", states.max())
        # print("subpolicy logits = ", subpolicy)
        subpolicy = self.subpolicy_output(subpolicy, tau=1, hard=True, eps=1e-10, dim=-1)
        # subpolicy_idx = torch.nonzero(subpolicy).detach()
        # subpolicy
        # print("subpolicy = ", subpolicy)
        return subpolicy

class LowLevelPolicy(nn.Module):

    def __init__(self, state_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.ReLU()):
        super().__init__()
        self.para_shape = para_shape
        # self.net = build_mlp_hidden_layers(
        #     input_dim=state_shape[0],
        #     hidden_units=hidden_units,
        #     hidden_activation=hidden_activation
        # )
        self.para_fc = build_mlp(input_dim=state_shape[0], output_dim=para_shape[0], 
                                 hidden_units=hidden_units, hidden_activation=hidden_activation, output_activation=None)
        self.para_output = nn.Sigmoid()

    def forward(self, states):
        # x = self.net(states)
        # subpolicy
        paras = (self.para_fc(states)).view(-1, self.para_shape[0])
        paras = self.para_output(paras)
        # x = self.output(x)
        # print("paras = ", paras)
        return paras

class MLPNetwork(nn.Module):
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
        super(MLPNetwork, self).__init__()

        self.in_fn = lambda x: x
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
            self.out_fn = lambda x: x
            
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

class hierMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, out_fn=None, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(hierMLPNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        # self.out_fn = lambda x: x
        if out_fn is not None:
            if out_fn is F.tanh:
                # initialize small to prevent saturation
                self.fc4.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = out_fn
        else:  # logits for discrete action (will softmax later)
            raise NotImplementedError
            # self.out_fn = lambda x: x

    def generate_embedding(self, X):
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        return h3

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h3 = self.generate_embedding(X)
        out = self.out_fn(self.fc4(h3))
        return out

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

    def get_embedding_loss(self, obs, next_obs, loss_func):
        obs_embedding = self.generate_embedding(obs)
        nextObs_embedding = self.generate_embedding(next_obs)
        loss = loss_func(obs_embedding, nextObs_embedding)
        return loss


class ObstacleQuasiGNNNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, constrain_out=True, nonlin=F.relu, device="cuda"):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ObstacleQuasiGNNNetwork, self).__init__()
        self.in_fn = lambda x: x
        self.device = device
        self.conv1 = dglnn.SAGEConv(
            in_feats=4, out_feats=hidden_dim, aggregator_type='pool', activation = nonlin)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type='pool')
        self.initialized_graphs = dict()
        self.avgpool = AvgPooling()

        self.obstacle_extracter1 = nn.Linear(4, hidden_dim)
        self.obstacle_extracter2 = nn.Linear(hidden_dim, hidden_dim)

        self.global_extracter1 = nn.Linear(16, hidden_dim)
        self.global_extracter2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim)
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

    def initialize_graph(self, n):
        # initialize graph just from 
        s, e = fully_connected_include_self(n)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e))
        g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        return g

    def generate_obstacle_embedding(self, X):
        x_dim = X.ndim
        if x_dim == 1:
            obstacles = X[16:].view(1, -1, 4)
        elif x_dim == 2:
            obstacles = X[...,16:].view(X.shape[0], -1, 4)
        else:
            NotImplementedError
        batches_detected_obstacle_n, batches_detected_obstacles = self.select_observed_obstacles(obstacles)
        graph_list = []
        graph_input = []
        for n, obstacles in zip(batches_detected_obstacle_n, batches_detected_obstacles):
            # assert n.item().is_integer()
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]
            # h = torch.vstack(batches_detected_obstacles)
            graph_list.append(g)
            graph_input.append(obstacles)
        batched_graphs = dgl.batch(graph_list).to(self.device)
        hn = torch.cat(graph_input, dim=0).to(self.device)
        res = self.conv1(batched_graphs, hn)
        res = self.conv2(batched_graphs, res)
        obstacle_embedding = self.avgpool(batched_graphs, res)

        # obstacle_features = self.nonlin(self.obstacle_extracter1(observed_obstacles))
        # obstacle_features = self.obstacle_extracter2(obstacle_features)
        # obstacle_embedding = torch.mean(obstacle_features, dim=-1)
        return obstacle_embedding.squeeze()

    def select_observed_obstacles(self, batches_obstacles):
        batches_detected_obstacle_n = []
        batches_detected_obstacles = []
        for batch_i, obstacles in enumerate(batches_obstacles):
            obstacle_detect_indicator = obstacles[:, 1]
            detected_obstacles = torch.cat((torch.zeros(1, 4), obstacles[obstacle_detect_indicator == 1, :]), dim=0)
            detected_obstacle_n = detected_obstacles.shape[0]
            batches_detected_obstacle_n.append(detected_obstacle_n)
            batches_detected_obstacles.append(detected_obstacles)
        return batches_detected_obstacle_n, batches_detected_obstacles

    def generate_globalInfo_embedding(self, X):
        global_info = X[...,0:16]
        global_features = self.nonlin(self.global_extracter1(global_info))
        global_features = self.global_extracter2(global_features)
        return global_features

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        globalInfo_embedding = self.generate_globalInfo_embedding(X)
        obstacle_embedding = self.generate_obstacle_embedding(X)
        combined_embedding = torch.cat((globalInfo_embedding, obstacle_embedding), dim=-1)
        combined_embedding = self.nonlin(self.fc1(combined_embedding))
        combined_embedding = self.nonlin(self.fc2(combined_embedding))
        out = self.out_fn(self.fc3(combined_embedding))
        return out

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

    def get_embedding_loss(self, obs, next_obs, loss_func):
        obs_embedding = self.generate_embedding(obs)
        nextObs_embedding = self.generate_embedding(next_obs)
        loss = loss_func(obs_embedding, nextObs_embedding)
        return loss


class CNNNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, constrain_out=True, nonlin=F.relu, device="cuda"):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CNNNetwork, self).__init__()

        self.device = device
        self.in_fn = lambda x: x
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=7, padding="same")
        # self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, padding="same")
        # self.pool2 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear(900, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.fc2.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
            
    def forward(self, X, A=None):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        batch_size = X.shape[0]
        h1 = (self.nonlin(self.conv1(self.in_fn(X))))
        h2 = ((self.conv2(h1)))
        h3 = self.nonlin(self.fc1(h2.view(batch_size, -1)))
        out = self.out_fn(self.fc2(h3))
        return out, torch.Tensor([]).to(self.device)

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

class MLPPolicyNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, comm_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True, device="cpu"):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPPolicyNetwork, self).__init__()
        self.act_dim = out_dim - comm_dim
        self.comm_dim = comm_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_comm1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_comm2 = nn.Linear(hidden_dim, self.comm_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.fc_comm3 = nn.Linear(1, 2)
        self.nonlin = nonlin
        self.bin = torch.Tensor([0, 1]).to(device)
        # set output function for action
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        # elif constrain_out and discrete_action:
        #     self.out_fn = lambda x: F.gumbel_softmax(x, hard=True, tau=1.0)
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        # set output function for communication
        self.comm_out_fn = F.tanh

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h1_comm = self.nonlin(self.fc_comm1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h2_comm = self.nonlin(self.fc_comm2(h1_comm))
        h2_comm = h2_comm.unsqueeze(-1)
        out_act = self.out_fn(self.fc3(h2))
        out_comm = self.comm_out_fn(self.fc_comm3(h2_comm))

        action_comm_part = F.gumbel_softmax(out_comm, hard=True, tau=1.0)
        out_comm = action_comm_part @ self.bin
        return out_act, out_comm

class GNNNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, concat_dim, out_dim, hidden_dim=64, gnn_hidden=8, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(GNNNetwork, self).__init__()

        self.act = nn.Tanh()
        self.conv1 = dglnn.SAGEConv(
            in_feats=input_dim, out_feats=gnn_hidden, aggregator_type='pool', activation = self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=gnn_hidden, out_feats=gnn_hidden, aggregator_type='pool')

        in_dim = concat_dim + gnn_hidden
        self.in_fn = lambda x: x
        self.fc1 = nn.Linear(in_dim, hidden_dim)
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
            self.out_fn = lambda x: x

        self.initialized_graphs = dict()
        self.avgpool = AvgPooling()
    
    def initialize_graph(self, n):
        s, e = central_graph(n)
        if n != 1:
            # Can't go to device if empty lists
            s = s.to(self.device)
            e = e.to(self.device)
            g = dgl.graph((s, e), num_nodes = n)
            g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        else:
            g = dgl.graph((s, e), num_nodes = n).to(self.device)
            # Need below to batch correctly
            g.ndata['_ID'] = torch.tensor([0], device=self.device)
            g.ndata['_TYPE'] = torch.tensor([0], device=self.device)
        return g

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, graph_obs, other_obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        graph_list = []
        graph_inputs = []
        for graph_tensors in graph_obs:
            n = len(graph_tensors)
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g 
            else:
                g = self.initialized_graphs[n]
            graph_list.append(g)
            h = torch.vstack(graph_tensors)
            graph_inputs.append(h)

        graph_inputs = torch.cat(graph_inputs, dim=0)
        batched_graphs = dgl.batch(graph_list).to(self.device)
        res = self.conv1(batched_graphs, graph_inputs)
        res = self.conv2(batched_graphs, res)
        res = self.avgpool(batched_graphs, res) # maybe don't need average pool?
        
        other_obs = torch.vstack(other_obs)
        res = torch.cat([res, other_obs], dim=1)

        h1 = self.nonlin(self.fc1(self.in_fn(res)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

class AttentionMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, gaussian_num=8, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(AttentionMLPNetwork, self).__init__()

        self.gaussian_num = gaussian_num
        self.coeff_input_dim = input_dim
        self.info_input_dim = self.coeff_input_dim - (gaussian_num - 1) * 5

        self.in_fn = lambda x: x
        self.coeff_fc1 = nn.Linear(self.coeff_input_dim, self.gaussian_num)

        self.info_fc1 = nn.Linear(self.info_input_dim, hidden_dim)
        self.info_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.info_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.info_fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.softmax = nn.Softmax(dim=-1)
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.info_fc4.weight.data.uniform_(-3e-3, 3e-3)
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
        if X.ndim == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        X_feat_num = X.shape[-1]

        base_blue_obs = X[:,0:(X_feat_num-5*self.gaussian_num)]
        gaussians = X[:,-5*self.gaussian_num:]
        pi = gaussians[:,0:self.gaussian_num]
        mu = gaussians[:,self.gaussian_num:3*self.gaussian_num]
        sigma = gaussians[:,3*self.gaussian_num:]
        agents_pi_mu_sigma = torch.cat([torch.cat((pi[:,i:i+1], mu[:,2*i:2*i+2], sigma[:,2*i:2*i+2]), dim=-1) for i in range(self.gaussian_num)], dim=-1)
        # splited_gaussians = torch.split(gaussians, self.gaussian_num, dim=-1)
        gaussians = agents_pi_mu_sigma.view(batch_size, self.gaussian_num, 5)
        base_blue_obs = base_blue_obs.unsqueeze(1).repeat(1,self.gaussian_num,1)
        blue_obs = torch.cat((base_blue_obs, gaussians), dim=-1)

        coeff_h1 = self.nonlin(self.coeff_fc1(X))
        coeff = self.softmax(coeff_h1).unsqueeze(2)

        info_h1 = self.nonlin(self.info_fc1(blue_obs))

        info_h1 = (coeff * info_h1).sum(dim=1).squeeze(1)
        info_h2 = self.nonlin(self.info_fc2(info_h1))
        info_h3 = self.nonlin(self.info_fc3(info_h2))
        out = self.out_fn(self.info_fc4(info_h3)).squeeze()
        return out

    def compute_loss(self, x, y, loss_func):
        y_pred = self.forward(x)
        loss = loss_func(y_pred, y)
        return loss

class AttentionEmbeddingNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, gaussian_num=8, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(AttentionEmbeddingNetwork, self).__init__()

        self.gaussian_num = gaussian_num
        self.coeff_input_dim = input_dim
        self.info_input_dim = self.coeff_input_dim - (gaussian_num - 1) * 5

        self.in_fn = lambda x: x
        self.coeff_fc1 = nn.Linear(self.coeff_input_dim, self.gaussian_num)

        self.info_fc1 = nn.Linear(self.info_input_dim, hidden_dim)
        self.info_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.info_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = nonlin
        self.softmax = nn.Softmax(dim=-1)
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.info_fc3.weight.data.uniform_(-3e-3, 3e-3)
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
        if X.ndim == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        X_feat_num = X.shape[-1]

        base_blue_obs_act = X[:,0:(X_feat_num-5*self.gaussian_num)]
        gaussians = X[:,-5*self.gaussian_num:]
        # probabilities = gaussians[:, ::5].unsqueeze(2)
        # probabilities = gaussians[:, 0:8].unsqueeze(2)

        # splited_gaussians = torch.split(gaussians, self.gaussian_num, dim=-1)
        gaussians = gaussians.view(batch_size, self.gaussian_num, 5)
        base_blue_obs_act = base_blue_obs_act.unsqueeze(1).repeat(1,self.gaussian_num,1)
        blue_obs_act_gaussian = torch.cat((base_blue_obs_act, gaussians), dim=-1)
        coeff_h1 = self.nonlin(self.coeff_fc1(X))
        coeff = self.softmax(coeff_h1).unsqueeze(2)
        # print("probabilities[0] = ", probabilities[0])

        info_h1 = self.nonlin(self.info_fc1(blue_obs_act_gaussian))
        info_h2 = self.nonlin(self.info_fc2(info_h1))
        info_h3 = (self.info_fc3(info_h2))
        info_h3 = (coeff * info_h3).sum(dim=1).squeeze(1)

        # info_h3 = X
        return info_h3

class AttentionCriticNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, gaussian_num=8, constrain_out=True, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(AttentionCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.softmax = nn.Softmax(dim=-1)
        # self.out_fn = lambda x: x
        if constrain_out:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, embeddings):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """

        h1 = self.nonlin(self.fc1(embeddings))
        h2 = self.nonlin(self.fc2(h1))
        output = self.out_fn(self.fc3(h2))
        return output

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class AverageActor(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, s_sizes, hidden_dim=32, comm_dim=16, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AverageActor, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = s_sizes
        self.comm_dim = comm_dim
        self.nagents = len(s_sizes)
        self.attend_heads = attend_heads

        self.actor_state_encoders = nn.ModuleList()
        self.actor_agg_encoder = nn.Linear(hidden_dim, comm_dim)
        self.actor_agg_nonlin = nn.LeakyReLU(hidden_dim, comm_dim)

        self.actor_agg_decoders = nn.ModuleList()

        # self.key_extractors = nn.ModuleList()
        # self.selector_extractors = nn.ModuleList()

        # iterate over agents
        for sdim in s_sizes:
            state_encoder = nn.Sequential()
            agg_decoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.actor_state_encoders.append(state_encoder)

            agg_decoder.add_module('agg_dec_fc1', nn.Linear(comm_dim, 2))
            agg_decoder.add_module('agg_dec_nl', nn.Tanh())

            self.actor_agg_decoders.append(agg_decoder)

            # self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            # self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))

    def forward(self, observations, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        agg_agent_num = len(agents)
        # extract state encoding for each agent policy
        s_encodings = [self.actor_state_encoders[a_i](obs_i) for (a_i, obs_i) in zip(agents, observations)]
        s_agg_encodings = [self.actor_agg_encoder(s_e).reshape(-1, self.comm_dim) for s_e in s_encodings]
        s_agg = (1/agg_agent_num) * self.actor_agg_nonlin(torch.sum(torch.cat(s_agg_encodings, dim=0), dim=0, keepdim=True))
        actions = [self.actor_agg_decoders[a_i](s_agg) for a_i in agents]

        return actions

class GNNMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden, search_parties_num=5, helicopters_num=1, mode="actor", device="cuda"):
        super(GNNMLP, self).__init__()
        self.total_agent_types_num = search_parties_num + helicopters_num
        self.agents_types = [0 for _ in range(search_parties_num)] + [1 for _ in range(helicopters_num)]
        self.mode = mode
        self.hidden_dim = hidden_dim
        # self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.SAGEConv(
            in_feats=input_dim, out_feats=hidden_dim, aggregator_type='pool', activation = self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=gnn_hidden, aggregator_type='pool')


        self.avgpool = AvgPooling()

        if self.mode == "critic":
            self.mlp_to_value = nn.ModuleList([MLPNetwork(input_dim=gnn_hidden, out_dim=1, hidden_dim=gnn_hidden, constrain_out=False).to(device) for _ in range(self.total_agent_types_num)])
        elif self.mode == "actor":
            self.mlp_to_actions = nn.ModuleList([MLPNetwork(input_dim=gnn_hidden, out_dim=2, hidden_dim=gnn_hidden).to(device) for _ in range(self.total_agent_types_num)])
        else:
            raise NotImplementedError

        self.batched_graphs = None
        self.initialized_graphs = dict()

    def initialize_graph(self, n):
        # initialize graph just from 
        s, e = fully_connected_include_self(n)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e))
        g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        return g

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, observations_groups, agents_groups, updating_agent_idx=None):
        # x is (batch_size, seq_len, num_agents, features)
        # batch_size, num_agents = extract_graph_configurations(observations_groups)
        # permuted = agent_obs.permute(0, 2, 1, 3) # (batch_size, num_agents, seq_len, features)
        # hn is of shape (batch_size * num_agents, hidden_dim)
        # .view(batch_size * lstm_input.shape[1], seq_len, features)

        # observations_groups = [observations_groups]
        # agents_groups = [agents_groups]

        graph_list = []
        lstm_input = []

        for agents_group, observations_group in zip(agents_groups, observations_groups):
            # assert n.item().is_integer()
            n = len(agents_group)
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]
            h = torch.vstack(observations_group)
            graph_list.append(g)
            lstm_input.append(h)
        batched_graphs = dgl.batch(graph_list).to(self.device)
        hn = torch.cat(lstm_input, dim=0)
        res = self.conv1(batched_graphs, hn)
        res = self.conv2(batched_graphs, res)
        res = self.avgpool(batched_graphs, res)
        # [B x hidden_dim]
        agents_groups_actions_or_values = []
        if self.mode == "actor":
            for group_i, agents_group in enumerate(agents_groups):
                agents_group_actions = []
                for agent_i in agents_group:
                    actor = self.mlp_to_actions[self.agents_types[agent_i]]
                    agents_group_actions.append(actor(res[group_i]))
                agents_groups_actions_or_values.append(agents_group_actions)
        elif self.mode == "critic":
            for group_i, agents_group in enumerate(agents_groups):
                critic = self.mlp_to_value[self.agents_types[updating_agent_idx]]
                agents_groups_actions_or_values.append(critic(res[group_i]))
            agents_groups_actions_or_values = torch.vstack(agents_groups_actions_or_values).to(self.device)
        else:
            raise NotImplementedError

        return agents_groups_actions_or_values

class AverageCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, comm_dim=16, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AverageCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.comm_dim = comm_dim
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_state_action_encoders = nn.ModuleList()
        self.critic_agg_encoder = nn.Linear(hidden_dim, comm_dim)
        self.critic_agg_nonlin = nn.LeakyReLU(hidden_dim, comm_dim)

        self.critic_agg_decoders = nn.ModuleList()

        # self.key_extractors = nn.ModuleList()
        # self.selector_extractors = nn.ModuleList()

        # iterate over agents
        for sa_dim in sa_sizes:
            state_action_encoder = nn.Sequential()
            agg_decoder = nn.Sequential()
            if norm_in:
                state_action_encoder.add_module('sa_enc_bn', nn.BatchNorm1d(
                                            sa_dim, affine=False))
            state_action_encoder.add_module('sa_enc_fc1', nn.Linear(sa_dim,
                                                            hidden_dim))
            state_action_encoder.add_module('sa_enc_nl', nn.LeakyReLU())
            self.critic_state_action_encoders.append(state_action_encoder)

            agg_decoder.add_module('agg_dec_fc1', nn.Linear(comm_dim, comm_dim))
            agg_decoder.add_module('agg_dec_fc2', nn.Linear(comm_dim, 1))

            self.critic_agg_decoders.append(agg_decoder)

            # self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            # self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))

    def forward(self, observations, updating_agent=None, agents=None):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        agg_agent_num = len(agents)
        # extract state encoding for each agent policy
        observations = torch.split(observations, self.sa_sizes[0], dim=-1)
        sa_encodings = [self.critic_state_action_encoders[a_i](obs_i) for (a_i, obs_i) in zip(agents, observations)]
        sa_agg_encodings = [self.critic_agg_encoder(sa_e).reshape(-1, self.comm_dim) for sa_e in sa_encodings]
        sa_agg = (1/agg_agent_num) * self.critic_agg_nonlin(torch.sum(torch.cat(sa_agg_encodings, dim=0), dim=0, keepdim=True))
        value = self.critic_agg_decoders[updating_agent](sa_agg)

        return value


class HierRandomPolicy(nn.Module):
    def __init__(self, state_shape, agent_num, subpolicy_shape, para_shape, hidden_units=(64, 64), hidden_activation=nn.ReLU()) -> None:
        super().__init__()
        self.agent_num = agent_num
        self.subpolicy_shape = subpolicy_shape
        self.para_shape = para_shape
        self.common_layer = build_mlp_hidden_layers(input_dim=state_shape[0], hidden_units=hidden_units, hidden_activation=hidden_activation)
        self.subpolicy_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.subpolicy_fc2 = nn.Linear(hidden_units[-1], subpolicy_shape[0] * agent_num)
        self.para_fc1 = build_mlp_hidden_layers(input_dim=hidden_units[-1], hidden_units=(hidden_units[-1],), hidden_activation=hidden_activation)
        self.para_fc2 = nn.Linear(hidden_units[-1], para_shape[0] * agent_num)
        self.subpolicy_output = gumbel_softmax_soft_hard
        self.para_output = nn.Sigmoid()
        self.log_stds = nn.Parameter(torch.zeros(1, self.agent_num * para_shape[0])) # initialized as all zero

    def forward(self, states):
        # x = self.common_layer(states)
        # subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
        # subpolicy_prob, subpolicy = self.subpolicy_output(subpolicy, tau=1, eps=1e-10, dim=-1)
        # # subpolicy
        # paras_mean_to_sigmoid = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
        # paras = self.para_output(paras_mean_to_sigmoid)
        subpolicy_prob, subpolicy, paras_mean_to_sigmoid = self.generate_dist(states)
        paras, log_paras = reparameterize_sigmoid(paras_mean_to_sigmoid, self.log_stds)
        paras = paras.view(-1, self.agent_num, self.para_shape[0])
        log_subpolicy = torch.log(subpolicy_prob[subpolicy==1] + 1e-6).sum()
        log_pi = log_subpolicy
        # print("subpolicy_prob, subpolicy = ", (subpolicy_prob, subpolicy))
        # print("subpolicy_prob = ", subpolicy_prob)
        # print("log_subpolicy = %f, log_paras = %f" % (log_subpolicy, log_paras))
        return subpolicy, paras, log_pi

    def generate_dist(self, states):
        x = self.common_layer(states)
        subpolicy = self.subpolicy_fc2(self.subpolicy_fc1(x)).view(-1, self.agent_num, self.subpolicy_shape[0])
        subpolicy_prob, subpolicy = self.subpolicy_output(subpolicy, tau=3, eps=1e-10, dim=-1)
        # subpolicy
        paras_mean_to_sigmoid = self.para_fc2(self.para_fc1(x)).view(-1, self.agent_num, self.para_shape[0])
        return subpolicy_prob, subpolicy, paras_mean_to_sigmoid

    # def sample(self, states):
    #     return reparameterize(self.net(states), self.log_stds)

    # def evaluate_log_subpolicy_para_pi(self, states, subpolicy_prob, subpolicy, paras_mean_to_sigmoid):
    #     act_policy_prob = subpolicy_prob[subpolicy==1]
    #     act_para_prob = evaluate_lop_pi_para(paras_mean_to_sigmoid, self.log_stds, paras_act)
    #     # subpolicy_prob, paras_mean_to_sigmoid = self.generate_dist(states)
    #     # subpolicy_act = actions[0]
    #     # paras_act = actions[1]

    #     return 

    # def evaluate_log_subpolicy_pi()

def gumbel_softmax_soft_hard(logits, tau=1, eps=1e-10, dim=-1):
    probs = F.gumbel_softmax(logits, tau, hard=False, eps=eps, dim=dim)
    index = probs.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    one_hot = y_hard - probs.detach() + probs
    return probs, one_hot.detach()
       



