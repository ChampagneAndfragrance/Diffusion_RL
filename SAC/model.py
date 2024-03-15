import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class ThreatModel(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ThreatModel, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, detection_in):
        x1 = F.relu(self.linear1(detection_in))
        x1 = F.relu(self.linear2(x1))
        x1 = F.sigmoid(self.linear3(x1))
        return x1

    def loss(self, detection_in, detection_out_gt, dones_sample):
        detection_in, detection_out_gt, dones_sample = detection_in[0], detection_out_gt[0], dones_sample[0]
        batch_size = detection_in.shape[0]
        detection_out_logits = self.forward(detection_in.view(batch_size,-1))
        loss_func = torch.nn.BCELoss(reduction='none')
        loss = loss_func(detection_out_logits, torch.any(detection_out_gt==1, dim=1).float()) * (1 - dones_sample.float())
        return loss.mean()

class HeuThreatModel(nn.Module):
    def __init__(self, seq_len_in, seq_len_out):
        super(HeuThreatModel, self).__init__()
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.apply(weights_init_)

    def forward(self, detection_in):
        detection_in = detection_in.view(detection_in.shape[0], self.seq_len_in, -1)

        red_loc = detection_in[:,-1,:2]
        red_vel = detection_in[:,-1,2:4]
        red_path_pt = self.predict_next_locations(red_loc, red_vel, num_steps=self.seq_len_out)

        heli_relative_loc = detection_in[:,-1,4:6]
        heli_abs_vel = detection_in[:,-1,6:8]
        heli_abs_loc = heli_relative_loc + red_loc
        heli_path_pt = self.predict_next_locations(heli_abs_loc, heli_abs_vel, num_steps=self.seq_len_out)
        detected_by_heli = torch.linalg.norm(heli_path_pt-red_path_pt, dim=-1) < 0.05

        sp_relative_loc = detection_in[:,-1,8:10]
        sp_abs_vel = detection_in[:,-1,10:12]
        sp_abs_loc = sp_relative_loc + red_loc
        sp_path_pt = self.predict_next_locations(sp_abs_loc, sp_abs_vel, num_steps=self.seq_len_out)
        detected_by_sp = torch.linalg.norm(sp_path_pt-red_path_pt, dim=-1) < 0.05

        return torch.concat((detected_by_heli, detected_by_sp), dim=-1)

    def predict_next_locations(self, location_batch, velocity_batch, num_steps=10):
        batch_size = location_batch.shape[0]

        # INFO: Ensure both tensors have the same shape
        assert location_batch.shape == velocity_batch.shape, "Shape mismatch between current_location and velocities"

        # INFO: Create a time_steps tensor for each step
        time_steps = torch.arange(1, num_steps + 1, dtype=location_batch.dtype, device=location_batch.device)
        
        # INFO: Use the linear motion model to predict the next locations for all steps
        predicted_locations = location_batch.unsqueeze(-1).repeat(1,1,num_steps) + velocity_batch.unsqueeze(-1) * time_steps.repeat(batch_size,2,1)

        return predicted_locations.transpose(1, 2)

    # def get_red_blue_state(self):
    #     red_loc = np.array(self.prisoner.location) / self.dim_x
    #     red_vel = self.current_prisoner_velocity / self.max_timesteps
    #     blue_relatives = np.concatenate(self.get_relative_hs_locVels(), axis=0) / self.dim_x
    #     return np.concatenate((red_loc, red_vel, blue_relatives), axis=0)

    # def get_relative_hs_locVels(self):
    #     helicopter_locVels = []
    #     for helicopter in self.helicopters_list:
    #         helicopter_locVel = helicopter.location.copy() + helicopter.step_dist_xy.tolist()
    #         helicopter_locVels.append(np.array(helicopter_locVel)-np.array(self.prisoner.location+[0,0]))
    #     search_party_locVels = []
    #     for search_party in self.search_parties_list:
    #         search_party_locVel = search_party.location.copy() + search_party.step_dist_xy.tolist()
    #         search_party_locVels.append(np.array(search_party_locVel)-np.array(self.prisoner.location+[0,0]))
    #     return helicopter_locVels + search_party_locVels

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, bound=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if bound is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (bound[1] - bound[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (bound[1] + bound[0]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        # INFO: generate mean and log_std from the policy network
        mean, log_std = self.forward(state)
        # INFO: recover the std from its log
        std = log_std.exp()
        # INFO: get a normal distribution from the mean and the variance
        normal = Normal(mean, std)
        # INFO: for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # y_t = F.softmax(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # INFO: Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DiscretePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, bound=None):
        super(DiscretePolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if bound is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (bound[1] - bound[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (bound[1] + bound[0]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

    def sample(self, state):
        # INFO: generate logits from the policy network
        logits = self.forward(state)
        # INFO: gumbel-softmax for differentialble actions
        y_t = F.gumbel_softmax(logits, hard=True)
        action = y_t * self.action_scale + self.action_bias
        # INFO: get the probability of each discrete action
        log_prob = (F.log_softmax(logits+1e-8) * y_t.detach()).sum(1, keepdim=True)
        # INFO: no exploration
        max_index = torch.argmax(logits, dim=-1)
        # Create a one-hot tensor
        one_hot = torch.zeros_like(logits)
        # Set the element at max_index to 1
        one_hot.scatter_(dim=-1, index=max_index.unsqueeze(-1), value=1)
        return action, log_prob, one_hot

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DiscretePolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)