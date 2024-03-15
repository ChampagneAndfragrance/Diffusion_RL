from turtle import down
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from bc_utils import build_mlp, reparameterize, evaluate_lop_pi, atanh, asigmoid, calculate_log_pi
import time


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



class ConvEncoderPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(9, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 1)

        self.net = build_mlp(
            input_dim=2*12*12,
            output_dim=2,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, 2)) # initialized as all zero

    def forward(self, costmap_observation_layers):


        return torch.tanh(self.conv_fc(costmap_observation_layers))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def conv_fc(self, costmap_observation_layers):

        x = F.relu(self.conv1(costmap_observation_layers))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # x = F.relu(self.conv5(x))
        # x = self.pool(x)
        x = self.net(x.view(x.shape[0], 2*12*12))
        # print(x.shape)
        return x
    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.conv_fc(states), self.log_stds, actions)


class ConvFCPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        self.pool8 = nn.MaxPool2d(8, 8)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(1, 1)

        obs_hidden = (64, 32)
        self.net_for_obs = build_mlp(input_dim=120, output_dim=16, hidden_units=obs_hidden, hidden_activation=hidden_activation)
        self.net = build_mlp(
            input_dim=1385,
            output_dim=2,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, 2)) # initialized as all zero

    def forward(self, costmap, observations):


        return torch.tanh(self.conv_fc(costmap, observations))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def conv_fc(self, costmap, observations):
        forward_start = time.time()

        batch_size = observations.shape[0]
        obs_emb_start = time.time()
        x_obs = self.net_for_obs(observations)
        obs_emb_end = time.time()
        # print("obs embeddings generation uses: ", obs_emb_end-obs_emb_start)

        downsample_start = time.time()
        # costmap = self.pool8(costmap)
        downsample_send = time.time()
        # print("downsample uses: ", downsample_send-downsample_start)

        conv1_start = time.time() 
        x = F.relu(self.conv1(costmap))
        conv1_end = time.time() 
        # print("conv1 uses: ", conv1_end-conv1_start)

        x = self.pool8(x)
        x = F.relu(self.conv2(x))
        x = self.pool4(x)
        # x = x + x_obs.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        # x = F.relu(self.conv5(x))
        # x = self.pool(x)

        x = x.view(1, 1*37*37).repeat(batch_size, 1)
        x = torch.cat((x, x_obs), dim=1)

        x = self.net(x)
        # print(x.shape)
        forward_end = time.time()
        # print("The forward process comsumes:", forward_end-forward_start)
        return x

    def evaluate_log_pi(self, costmap, observations, actions):
        return evaluate_lop_pi(self.conv_fc(costmap, observations), self.log_stds, actions)


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 1)

        # # Decoder
        self.t_conv1 = nn.ConvTranspose2d(1, 4, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 1,kernel_size=2, stride=2)

    def forward(self, costmap, observation):
        x = F.relu(self.conv1(costmap))



    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv3(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv4(x))
    #     x = self.pool2(x)
        
        # deconvolutions
        # x = F.relu(self.t_conv1(x))
        # x = F.relu(self.t_conv2(x))
        # x = torch.sigmoid(self.t_conv3(x))
        # return x

    # def encoder(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv3(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv4(x))
    #     x = self.pool2(x)
    #     return x

class LSTMMDNBluePolicy(nn.Module):

    def __init__(self, num_features, num_actions, hidden_dims, mdn_num, hidden_act_func, output_func, device):
        super(LSTMMDNBluePolicy, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.mdn_num = mdn_num # m
        self.output_dim = num_actions # c
        self.lstm_layer_num = 1
        self.eps = 1e-8
        self.speed_dim = num_actions // 2
        self.angle_dim = num_actions // 2

        """MLP layer"""
        self.mlp0 = nn.Linear(num_features, self.hidden_dims[0])
        self.nonlinear0 = hidden_act_func
        """LSTM layer"""
        self.lstm1 = nn.LSTM(input_size=self.hidden_dims[0], hidden_size=self.hidden_dims[1], batch_first=True)
        self.nonlinear1 = hidden_act_func
        """MLP layer to generate alpha, sigma, mu"""
        self.alpha_layer = nn.Linear(self.hidden_dims[1], mdn_num)
        self.sigma_layer_speed = nn.Linear(self.hidden_dims[1], self.speed_dim * mdn_num)
        self.sigma_layer_angle = nn.Linear(self.hidden_dims[1], self.angle_dim * mdn_num)
        self.mu_layer_speed = nn.Linear(self.hidden_dims[1], self.speed_dim * mdn_num)
        self.mu_layer_angle = nn.Linear(self.hidden_dims[1], self.angle_dim * mdn_num)
        """output layer"""
        self.output_layer = output_func
        """utils layer"""
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


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
        red_action = torch.zeros(self.output_dim)
        x = x.unsqueeze(0)
        alpha, log_sigma_speed, log_sigma_angle, mu_speed, mu_angle = self.alpha_sigma_mu(x)
        if alpha.numel() != 1:
            alpha = alpha.squeeze()
            i = alpha.multinomial(num_samples=1, replacement=True)
            mu_speed_each_gs = mu_speed[:,:,self.speed_dim*i:self.speed_dim*(i+1)].squeeze()
            mu_angle_each_gs = mu_angle[:,:,self.angle_dim*i:self.angle_dim*(i+1)].squeeze()
        else:
            mu_speed_each_gs = mu_speed.squeeze()
            mu_angle_each_gs = mu_angle.squeeze()

        red_action[0::2] = torch.sigmoid(mu_speed_each_gs)
        red_action[1::2] = torch.tanh(mu_angle_each_gs)

        return red_action

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
        log_sigma_speed = self.sigma_layer_speed(hiddens) # (N, L=1, c * m = 1 * 5)
        log_sigma_angle = self.sigma_layer_angle(hiddens) # (N, L=1, c * m = 1 * 5)
        mu_speed = self.mu_layer_speed(hiddens) # (N, L=1, c * m = 1 * 5)
        mu_angle = self.mu_layer_angle(hiddens) # (N, L=1, c * m = 1 * 5)
        return alpha, log_sigma_speed, log_sigma_angle, mu_speed, mu_angle

    def evaluate_action_prob(self, x, y):
        alpha, log_sigma_speed, log_sigma_angle, mu_speed, mu_angle = self.alpha_sigma_mu(x)
        sum_all_log_gs = torch.Tensor(np.array([0])).to(self.device)
        sum_all_gs = torch.Tensor(np.array([self.eps])).to(self.device)
        for i in range(self.mdn_num):
            alpha_each_gs = alpha[:,:,i]
            log_sigma_speed_each_gs = log_sigma_speed[:,:,self.speed_dim*i:self.speed_dim*(i+1)].squeeze()
            log_sigma_angle_each_gs = log_sigma_angle[:,:,self.angle_dim*i:self.angle_dim*(i+1)].squeeze()
            mu_speed_each_gs = mu_speed[:,:,self.speed_dim*i:self.speed_dim*(i+1)].squeeze()
            mu_angle_each_gs = mu_angle[:,:,self.angle_dim*i:self.angle_dim*(i+1)].squeeze()
            """split y to y_speed and y_angle"""
            y_speed = y[:, 0::2]
            y_angle = y[:, 1::2]
            """noises should obey N(mu=0, sigma=1)"""

            noises_speed = (asigmoid(y_speed) - mu_speed_each_gs) / (log_sigma_speed_each_gs.exp() + self.eps)
            noises_angle = (atanh(y_angle) - mu_angle_each_gs) / (log_sigma_angle_each_gs.exp() + self.eps) 

            log_p_noises_standard_gs = calculate_log_pi(log_sigma_speed_each_gs, noises_speed, y_speed) + calculate_log_pi(log_sigma_angle_each_gs, noises_angle, y_angle)
            # sum_all_gs = sum_all_gs + alpha_each_gs * log_p_noises_standard_gs.exp()
            sum_all_log_gs = sum_all_log_gs + alpha_each_gs * log_p_noises_standard_gs
        neg_ln_mdn = -sum_all_log_gs.mean()
        mu_speed_average = mu_speed.mean()
        mu_angle_average = mu_angle.mean()
        log_sigma_speed_average = log_sigma_speed.mean()
        log_sigma_angle_average = log_sigma_angle.mean()
        # print("sum_all_gs_max = ", sum_all_gs.max())
        # neg_ln_mdn = -sum_all_gs.log().mean()
        return neg_ln_mdn, mu_speed_average, mu_angle_average, log_sigma_speed_average, log_sigma_angle_average

    def mdn_gaussian_distribution(self, alpha, sigma, mu, y):
        prob = alpha * ((1.0 / (torch.pow(2 * torch.pi, torch.Tensor(np.array(self.output_dim / 2.0)).to(self.device)) * sigma))) * torch.exp(-torch.pow(torch.linalg.norm(y - mu, dim=1, keepdim=True), 2) / (2 * torch.pow(sigma, 2)))
        return prob

    def evaluate_loss(self, x, y):
        loss, mu_speed_average, mu_angle_average, log_sigma_speed_average, log_sigma_angle_average = self.evaluate_action_prob(x, y)
        # loss = torch.mean(neg_ln_mdn)
        stats_dict = dict(neglogp=loss.item(), mu_speed_average=mu_speed_average, mu_angle_average=mu_angle_average,
                          log_sigma_speed_average=log_sigma_speed_average, log_sigma_angle_average=log_sigma_angle_average)
        return loss, stats_dict

    def predict(self, x, deterministic=None):
        obs = torch.from_numpy(x).float().unsqueeze(0).to("cuda")
        return self.alpha_sigma_mu(obs).cpu().detach().numpy()