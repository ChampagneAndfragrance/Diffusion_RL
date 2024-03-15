import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import copy
import sys
import torch.utils.data as data
# from train_le import Le_Dataset
from blue_bc.buffer import ReplayBuffer, Buffer
from enum import Enum, auto
from models.configure_model import configure_model
from collections import OrderedDict

from collections import namedtuple
from visualize.render_utils import plot_mog_heatmap

# Declaring namedtuple()
BlueObsStruct = namedtuple('id', 'obs')

sys.path.append("./blue_bc")
from policy import MLPNetwork

class Le_Dataset(data.Dataset):#data.Dataset
    def __init__(self, parent_folder):
        # TODO
        # 1. Initialize file path or list of file names.
        self.parent_folder = parent_folder
        self.x = []
        self.y = []
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return (torch.Tensor(self.x[index]), torch.Tensor(self.y[index]))
        # return (torch.Tensor(np.concatenate((self.x[index][0:2]*2428, self.x[index][2:4]*4320, self.x[index][4:6]*2428, self.x[index][6:8]*4320))), 2428*self.y[index].float())
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.x)

    def push(self, x, y):
        self.x.extend(x)
        self.y.extend(y)
        return

    def save(self, train_test):
        x_path = self.parent_folder / train_test / "x.npy"
        y_path = self.parent_folder / train_test / "y.npy"
        np.save(x_path, np.array(self.x))
        np.save(y_path, np.array(self.y))

    def load(self, train_test):
        x_path = self.parent_folder / train_test / "x.npy"
        y_path = self.parent_folder / train_test / "y.npy"
        self.x = np.load(x_path).tolist()
        self.y = np.load(y_path).tolist()

def save_video(ims, filename, fps=30.0):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def gumbel_softmax_soft_hard(logits, tau=1, eps=1e-10, dim=-1):
    probs = F.gumbel_softmax(logits, tau, hard=False, eps=eps, dim=dim)
    one_hot = F.gumbel_softmax(logits, tau, hard=True, eps=eps, dim=dim)
    return probs, one_hot.detach()


class HierScheduler(object):
    def __init__(self, env, timesteps) -> None:
        self.env = env
        self.subpolicy = None
        self.paras = None
        self.agent_num = self.env.num_search_parties + self.env.num_helicopters 
        # self.option_num = subpolicy.shape[-1]
        self.option_names = ["plan_path_to_stop", "plan_path_to_loc_para", "plan_spiral_para", "plan_pid_para"]
        self.para_indices = [[], [[0],[1,2]], [0], [0]]
        self.predicted_detection = None
        # self.detection_history = detection_history
        self.timesteps = timesteps
    
    @property
    def update_path_flag(self):
        if self.predicted_detection is not None:
            flag = True
        else:
            flag = False
        return flag

    @property
    def options(self):
        # opt = self.subpolicy.squeeze()
        options = np.nonzero(self.subpolicy)[-1]
        return options

    def update(self, subpolicy, paras, predicted_detection):
        self.subpolicy = subpolicy.squeeze().detach().cpu().numpy()
        self.paras = paras.squeeze().detach().cpu().numpy()    
        self.predicted_detection = predicted_detection

    def plan_path(self):
        # if self.update_path_flag:
        if True:
            blue_ag_idx = 0
            for search_party in self.env.search_parties_list:
                opt_idx = self.options[blue_ag_idx]
                self.command_agent(opt_idx, blue_ag_idx, search_party)
                # if self.para_indices[opt_idx] == []:
                #     getattr(search_party, self.option_names[opt_idx])()
                # else:
                #     getattr(search_party, self.option_names[opt_idx])(*([self.paras[blue_ag_idx,self.para_indices[opt_idx][i]] for i in range(len(self.para_indices[opt_idx]))]))
                blue_ag_idx = blue_ag_idx + 1

            if self.env.is_helicopter_operating():
                for helicopter in self.env.helicopters_list:
                    opt_idx = self.options[blue_ag_idx]
                    self.command_agent(opt_idx, blue_ag_idx, helicopter)
                    # if self.para_indices[opt_idx] == []:
                    #     getattr(helicopter, self.option_names[opt_idx])()
                    # else:
                    #     getattr(search_party, self.option_names[opt_idx])(*([self.paras[blue_ag_idx,self.para_indices[opt_idx][i]] for i in range(len(self.para_indices[opt_idx]))]))
                    blue_ag_idx = blue_ag_idx + 1
        else:
            pass
         
    def command_agent(self, opt_idx, blue_ag_idx, blue_agent):
        if opt_idx == 0:
            blue_agent.plan_path_to_stop_para()
        elif opt_idx == 1:
            blue_agent.plan_path_to_loc_para(self.paras[blue_ag_idx,0], self.paras[blue_ag_idx,[1,2]])
        elif opt_idx == 2:
            blue_agent.plan_spiral_para(self.paras[blue_ag_idx,0])
        elif opt_idx == 3:
            # vector = np.array(self.new_detection) - np.array(self.detection_history[-2][0])
            # speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
            # direction = np.arctan2(vector[1], vector[0])
            blue_agent.plan_pid_para(self.paras[blue_ag_idx,0], self.predicted_detection)
        else:
            raise NotImplementedError

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan_para())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan_para())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')
class BaseTrainer(object):
    def __init__(self, blue_policy, input_dim, out_dim, hidden_dim, update_target_period, lr, device) -> None:
        self.blue_policy = blue_policy.to(device)
        self.target_blue_policy = copy.deepcopy(self.blue_policy).to(device)
        self.blue_critic = MLPNetwork(input_dim, out_dim, hidden_dim, nonlin=torch.nn.functional.relu, constrain_out=False, norm_in=False, discrete_action=False).to(device)
        self.target_blue_critic = copy.deepcopy(self.blue_critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.blue_policy.parameters(), lr=lr) # original: lr
        self.critic_optimizer = torch.optim.Adam(self.blue_critic.parameters(), lr=lr)

        self.niter = 0
        self.update_target_period = update_target_period
        self.gamma = 0.95
        self.tau = 0.01

    def update(self, sample, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, hier_act, rewards, dones, next_obs = sample
        batch_size = obs.shape[0]
        vf_in = torch.cat((obs, hier_act.detach()), dim=1)
        actual_value = self.blue_critic(vf_in) # reward_prediction(from t)

        trgt_acs = self.target_blue_policy(next_obs)
        # trgt_acs = torch.cat((trgt_acs[0].view(batch_size, -1), trgt_acs[1].view(batch_size, -1)), dim=1)
        trgt_vf_in = torch.cat((next_obs, trgt_acs), dim=1)
        target_value = (rewards.view(-1, 1) + self.gamma * self.target_blue_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_critic.parameters(), 0.5)
        self.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        curr_pol_out = self.blue_policy(obs)
        # curr_pol_out = torch.cat((curr_pol_out[0].view(batch_size, -1), curr_pol_out[1].view(batch_size, -1)), dim=1)
        vf_in = torch.cat((obs, curr_pol_out), dim=1)
        pol_loss = -self.blue_critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_policy.parameters(), 0.5)
        self.policy_optimizer.step()

        # if self.niter % self.update_target_period == 0:
        self.update_all_targets()

        if logger is not None:
            logger.add_scalars('blue/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        self.soft_update(self.target_blue_critic, self.blue_critic, self.tau)
        self.soft_update(self.target_blue_policy, self.blue_policy, self.tau)
        self.niter += 1

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class HierTrainer(object):
    def __init__(self, blue_policy, input_dim, out_dim, hidden_dim, update_target_period, lr, device) -> None:
        self.blue_policy = blue_policy.to(device)
        self.target_blue_policy = copy.deepcopy(self.blue_policy).to(device)
        self.blue_critic = MLPNetwork(input_dim, out_dim, hidden_dim, nonlin=torch.nn.functional.relu, constrain_out=False, norm_in=False, discrete_action=False).to(device)
        self.target_blue_critic = copy.deepcopy(self.blue_critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.blue_policy.parameters(), lr=0.1 * lr) # original: lr
        self.critic_optimizer = torch.optim.Adam(self.blue_critic.parameters(), lr=lr)

        self.niter = 0
        self.update_target_period = update_target_period
        self.gamma = 0.95
        self.tau = 0.01

    def update(self, sample, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, hier_act, rewards, dones, next_obs = sample
        batch_size = obs.shape[0]
        vf_in = torch.cat((obs, hier_act.detach()), dim=1)
        actual_value = self.blue_critic(vf_in) # reward_prediction(from t)

        trgt_acs = self.target_blue_policy(next_obs)
        trgt_acs = torch.cat((trgt_acs[0].view(batch_size, -1), trgt_acs[1].view(batch_size, -1)), dim=1)
        trgt_vf_in = torch.cat((next_obs, trgt_acs), dim=1)
        target_value = (rewards.view(-1, 1) + self.gamma * self.target_blue_critic(trgt_vf_in)) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_critic.parameters(), 0.5)
        self.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        curr_pol_out = self.blue_policy(obs)
        curr_pol_out = torch.cat((curr_pol_out[0].view(batch_size, -1), curr_pol_out[1].view(batch_size, -1)), dim=1)
        vf_in = torch.cat((obs, curr_pol_out), dim=1)
        pol_loss = -self.blue_critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(self.blue_policy.parameters(), 0.5)
        self.policy_optimizer.step()

        # if self.niter % self.update_target_period == 0:
        self.update_all_targets()

        if logger is not None:
            logger.add_scalars('blue/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        self.soft_update(self.target_blue_critic, self.blue_critic, self.tau)
        self.soft_update(self.target_blue_policy, self.blue_policy, self.tau)
        self.niter += 1

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Estimator(Enum):
    DETECTIONS = auto()
    LEARNED_ESTIMATOR = auto()
    LINEAR_ESTIMATOR = auto()
    MDN_ESTIMATOR = auto()
    NO_DETECTIONS = auto()
    FLAT_SEQUENCE = auto()

def blue_obs_type_from_estimator(estimator_name, estimator_enum):
    if estimator_name in ["linear_estimator"]:
        blue_obs_type = estimator_enum.LINEAR_ESTIMATOR
        print("Using Linear Estimator")
    elif estimator_name in ["no_estimator", "predicting"]:
        print("Using No Detections")
        blue_obs_type = estimator_enum.NO_DETECTIONS
    elif estimator_name in ["detections", "ground_truth", "det_filt"]:
        print("using Localized Detections")
        blue_obs_type = estimator_enum.DETECTIONS
    elif estimator_name in ["filtering"]:
        print("using MDN filtering model")
        blue_obs_type = estimator_enum.MDN_ESTIMATOR
    elif estimator_name in ["seq_fc"]:
        blue_obs_type = estimator_enum.FLAT_SEQUENCE
    else:
        raise ValueError("Estimator should be no_estimator, linear_estimator, filtering, predicting or ground_truth")
    return blue_obs_type

def get_modified_blue_obs(env, blue_obs_type, estimator_enum):
    if blue_obs_type == estimator_enum.LINEAR_ESTIMATOR:
        blue_observation = env.get_modified_blue_observation_linear_estimator()
    elif blue_obs_type == estimator_enum.DETECTIONS:
        blue_observation = env.get_modified_blue_obs_last_detections()
    elif blue_obs_type == estimator_enum.NO_DETECTIONS:
        blue_observation = env.get_modified_blue_obs_no_detections()
    elif blue_obs_type == estimator_enum.MDN_ESTIMATOR:
        blue_observation = env.get_modified_blue_obs_no_detections_with_gaussians()    
    elif blue_obs_type == estimator_enum.FLAT_SEQUENCE:
        blue_observation = env.get_flat_seq_blue_obs_no_detections()
    return blue_observation

def get_modified_blue_obs_high(env, blue_obs_type, estimator_enum):
    if blue_obs_type == estimator_enum.LINEAR_ESTIMATOR:
        blue_observation = env.get_modified_blue_observation_linear_estimator()
    elif blue_obs_type == estimator_enum.LEARNED_ESTIMATOR:
        blue_observation = env.get_modified_blue_observation_learned_estimator_high()
    elif blue_obs_type == estimator_enum.DETECTIONS:
        blue_observation = env.get_modified_blue_obs_last_detections()
    elif blue_obs_type == estimator_enum.NO_DETECTIONS:
        blue_observation = env.get_modified_blue_obs_no_detections_with_high_actions()
    elif blue_obs_type == estimator_enum.MDN_ESTIMATOR:
        blue_observation = env.get_modified_blue_obs_no_detections_with_gaussians()   
    elif blue_obs_type == estimator_enum.FLAT_SEQUENCE:
        blue_observation = env.get_flat_seq_blue_obs_no_detections()
    return blue_observation

def get_modified_blue_obs_low(env, blue_obs_type, estimator_enum):
    if blue_obs_type == estimator_enum.LINEAR_ESTIMATOR:
        blue_observation = env.get_modified_blue_observation_linear_estimator()
    elif blue_obs_type == estimator_enum.LEARNED_ESTIMATOR:
        blue_observation = env.get_modified_blue_observation_learned_estimator_low()
    elif blue_obs_type == estimator_enum.DETECTIONS:
        blue_observation = env.get_modified_blue_obs_last_detections()
    elif blue_obs_type == estimator_enum.NO_DETECTIONS:
        blue_observation = env.get_modified_blue_obs_map()
    elif blue_obs_type == estimator_enum.MDN_ESTIMATOR:
        blue_observation = env.get_modified_blue_obs_no_detections_with_gaussians()   
    elif blue_obs_type == estimator_enum.FLAT_SEQUENCE:
        blue_observation = env.get_flat_seq_blue_obs_no_detections()
    return blue_observation

def convert_to_shared_parameters(blue_observation):
    """ Convert the blue observation from separate agents to shared parameters """
    # Currently the helicopter is the last agent
    # Search parties are 0 - 4

    blue_obs = []
    # Add id for each of the agent observations
    for i, obs in enumerate(blue_observation):
        blue_obs.append(BlueObsStruct(id=i, obs=obs))
    
    return blue_obs

def sort_filtering_output(filtering_output):
    pi, mu, sigma = filtering_output
    batch_size = pi.shape[0]
    sorted_idx = torch.argsort(pi, axis=1)
    pi=torch.sort(pi, axis=1).values
    sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    return pi, mu, sigma

def get_localized_trgt_gaussian_locations(torch_obs, agent_actions_high, non_localized_filtering_output_high):
    localized_trgt_gaussian_locations = []
    localized_trgt_gaussians = []
    for ob, ac in zip(torch_obs, agent_actions_high):
        ob = torch.Tensor(ob) if type(ob) is np.ndarray else ob
        ob = ob.to("cuda") if non_localized_filtering_output_high[0].is_cuda else ob
        agent_locations = ob[1:3].unsqueeze(0)
        # blue_agents_locations.append(agent_locations)
        localized_filtering_out = localize_filtering_mu(non_localized_filtering_output_high, agent_locations)
        pi, mu, sigma = sort_filtering_output(localized_filtering_out)
        high_action_idx = np.nonzero(ac)[0]
        trgt_gaussian_pi = pi[0,high_action_idx]
        trgt_gaussian_mu = mu[0,high_action_idx,:]
        trgt_gaussian_sigma = sigma[0,high_action_idx,:]
        trgt_gaussian = torch.concat((trgt_gaussian_pi.unsqueeze(0), trgt_gaussian_mu, trgt_gaussian_sigma), axis=-1).detach().cpu().numpy()
        localized_trgt_gaussian_locations.append(trgt_gaussian_mu)
        localized_trgt_gaussians.append(trgt_gaussian)
    return localized_trgt_gaussian_locations, localized_trgt_gaussians

def convert_buffer_to_dataloader(replay_buffer, config, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    le_train_dataset = Le_Dataset(parent_folder=data_dir)
    sample = replay_buffer.trace_back_sample(N=1e4, to_gpu=config["environment"]["cuda"], norm_rews=True)
    filtering_obs, prisoner_loc = sample
    le_train_dataset.x = filtering_obs[0].tolist()
    le_train_dataset.y = prisoner_loc[0].tolist()
    # le_train_dataset.save("")
    train_dataloader = data.DataLoader(le_train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    return train_dataloader

def load_filter(filtering_model_config, filtering_model_path, device):
    from models.configure_model import configure_model
    with open(filtering_model_config, 'r') as stream:
        config = yaml.safe_load(stream)
    # load the prior model here
    prior_or_combine = "prior_model"
    prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
    # construct the combined model with the loaded prior_model here
    prior_or_combine = "combined_model"
    filtering_model = configure_model(config, prior_or_combine, prior_network=prior_model)
    # convert to cuda or cpu
    filtering_model.to(device)
    if filtering_model_path is not None:
        filtering_model.load_state_dict(torch.load(filtering_model_path))
        print("Loaded filtering model from {}".format(filtering_model_path))
    return filtering_model

def split_filtering_input(filtering_input):
    prior_input = filtering_input[..., 0:3]
    dynamic_input = filtering_input[..., 3:]
    sel_input = filtering_input
    return [prior_input, dynamic_input, sel_input]

def localize_filtering_mu(filtering_output, agent_location):
    pi, mu, sigma = filtering_output
    agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
    mu_localize = mu - agent_location
    return pi, mu_localize, sigma

def convert_filtering_output_to_each_gaussian_tensor(filtering_output):
    pi, mu, sigma = filtering_output
    batch_size = pi.shape[0]
    gaussian_num = pi.shape[1]
    sorted_idx = torch.argsort(pi, axis=1)
    pi=torch.sort(pi, axis=1).values
    sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    sigma = sigma.reshape(batch_size, -1)
    mu = mu.reshape(batch_size, -1)

    agents_pi_mu_sigma = [torch.cat((pi[:,i:i+1], mu[:,2*i:2*i+2], sigma[:,2*i:2*i+2]), dim=-1) for i in range(gaussian_num)]
    return agents_pi_mu_sigma

def sort_filtering_output(filtering_output):
    pi, mu, sigma = filtering_output
    batch_size = pi.shape[0]
    sorted_idx = torch.argsort(pi, axis=1)
    pi=torch.sort(pi, axis=1).values
    sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
    return pi, mu, sigma

def get_localized_trgt_gaussian_locations(torch_obs, non_localized_filtering_output_high):
    non_localized_filtering_output_high = [out if torch.is_tensor(out) else torch.Tensor(out) for out in (non_localized_filtering_output_high)]
    localized_trgt_gaussian_locations = []
    for ob in (torch_obs):
        ob = torch.Tensor(ob) if type(ob) is np.ndarray else ob
        ob = ob.to("cuda") if non_localized_filtering_output_high[0].is_cuda else ob
        agent_locations = ob[1:3].unsqueeze(0)
        # blue_agents_locations.append(agent_locations)
        localized_filtering_out = localize_filtering_mu(non_localized_filtering_output_high, agent_locations)
        _, mu, _ = sort_filtering_output(localized_filtering_out)
        high_action_idx = -1
        trgt_gaussian_mu = mu[0,high_action_idx:,:]
        localized_trgt_gaussian_locations.append(trgt_gaussian_mu)
    return localized_trgt_gaussian_locations

def split_directions_to_direction_speed(directions, search_party_v_limit):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    # search_party_v_limit = 20
    helicopter_v_limit = 127
    for idx in range(len(blue_actions_directions)):
        if idx < 5:
            search_party_direction = blue_actions_directions[idx]
            if np.linalg.norm(search_party_direction) > 1:
                search_party_direction = search_party_direction / np.linalg.norm(search_party_direction)
            search_party_speed = np.maximum(np.linalg.norm(search_party_direction), 1.0) * search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = np.maximum(np.linalg.norm(helicopter_direction), 1.0) * helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel 

def classify_groups(torch_obs, low_tracking_idx, next_localized_trgt_gaussians, agent_actions_high):
    gaussians_possible_num = agent_actions_high[0].shape[0]
    torch_obs_low_group = [[] for _ in range(gaussians_possible_num)]
    agents_groups = [[] for _ in range(gaussians_possible_num)]
    for lti in low_tracking_idx:
        group_idx = torch.nonzero(agent_actions_high[lti])[0]
        torch_obs_low_group[group_idx].append(torch.concat((torch_obs[lti], next_localized_trgt_gaussians[lti].squeeze(), group_idx), dim=-1))
        agents_groups[group_idx].append(lti)
    return torch_obs_low_group, agents_groups

def assemble_obs_with_selIdx(torch_obs, low_tracking_idx, next_localized_trgt_gaussians, agent_actions_high):
    agents_num = len(torch_obs)
    for agent_i in range(agents_num):
        group_idx = torch.nonzero(agent_actions_high[agent_i])[0]
        torch_obs[agent_i] = torch.concat((torch_obs[agent_i], next_localized_trgt_gaussians[agent_i].squeeze(), group_idx), dim=-1).detach().cpu().numpy()
        if (agent_i in low_tracking_idx) == False:
            torch_obs[agent_i][-1] = -1
    return torch_obs

def disassemble_obs_with_selIdx(torch_obs, next_torch_obs, updating_agent_idx):
    torch_obs_low_group = []
    next_torch_obs_low_group = []
    agents_groups = []
    updating_agent_group_idx = torch_obs[updating_agent_idx][0,-1]
    for agent_i, (agent_torch_obs, next_agent_torch_obs) in enumerate(zip(torch_obs, next_torch_obs)): 
        group_idx = agent_torch_obs[0,-1]
        if group_idx == updating_agent_group_idx:
            torch_obs_low_group.append(agent_torch_obs)
            next_torch_obs_low_group.append(next_agent_torch_obs)
            agents_groups.append(agent_i)
        else:
            pass
    return torch_obs_low_group, next_torch_obs_low_group, agents_groups

def disassemble_obs(torch_obs, next_torch_obs, acs, updating_agent_idx, possible_gaussians_num):
    torch_obs_low_groups = []
    next_torch_obs_low_groups = []
    acs_low_groups = []
    agents_groups = []
    updating_agent_tracking_batch_idx = []
    # updating_agent_group_idx = torch_obs[updating_agent_idx][0,-1]
    # possible_gaussians_num = x
    batch_size = torch_obs[0].shape[0]
    agents_num = len(torch_obs)
    for batch_idx in range(batch_size): 
        agents_group = [[] for _ in range(possible_gaussians_num)]
        torch_obs_low_group = [[] for _ in range(possible_gaussians_num)]
        next_torch_obs_low_group = [[] for _ in range(possible_gaussians_num)]
        acs_low_group = [[] for _ in range(possible_gaussians_num)]

        updating_agent_group_idx = torch_obs[updating_agent_idx][batch_idx,-1]
        if updating_agent_group_idx != -1:
            updating_agent_tracking_batch_idx.append(batch_idx)

        for agent_i in range(agents_num):
            group_idx = torch_obs[agent_i][batch_idx][-1].int()
            if group_idx == updating_agent_group_idx and group_idx != -1:
                agents_group[group_idx].append(agent_i)
                torch_obs_low_group[group_idx].append(torch_obs[agent_i][batch_idx])
                next_torch_obs_low_group[group_idx].append(next_torch_obs[agent_i][batch_idx])
                acs_low_group[group_idx].append(acs[agent_i][batch_idx])
        agents_groups.extend(remove_empty_list(agents_group))
        torch_obs_low_groups.extend(remove_empty_list(torch_obs_low_group))
        next_torch_obs_low_groups.extend(remove_empty_list(next_torch_obs_low_group))
        acs_low_groups.extend(remove_empty_list(acs_low_group))
    return torch_obs_low_groups, next_torch_obs_low_groups, acs_low_groups, agents_groups, np.array(updating_agent_tracking_batch_idx)

def remove_empty_list(list_with_empty):
    list_without_empty = [ele for ele in list_with_empty if ele != []]
    return list_without_empty

def combine_obs_acs(obs, acs, gradient_passing_agents_idx):
    obs_acs = copy.deepcopy(obs)
    batch_size = len(obs)
    for batch_idx in range(batch_size):
        agents_in_group_num = len(obs[batch_idx])
        for agent_idx in range(agents_in_group_num):
            if agent_idx in gradient_passing_agents_idx:
                obs_acs[batch_idx][agent_idx] = torch.cat((obs[batch_idx][agent_idx], acs[batch_idx][agent_idx]), dim=-1)
            else:
                obs_acs[batch_idx][agent_idx] = torch.cat((obs[batch_idx][agent_idx].detach(), acs[batch_idx][agent_idx].detach()), dim=-1).detach()
    return obs_acs

def localize_filtering_mu(filtering_output, agent_location):
    pi, mu, sigma = filtering_output
    if type(agent_location) is np.ndarray:
        agent_location = np.repeat(np.expand_dims(agent_location, axis=1), repeats=8, axis=1)
    elif torch.is_tensor(agent_location):
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
    mu_localize = mu - agent_location
    return pi, mu_localize, sigma

def get_probability_grid(nn_output, true_location=None, res=10):
    pi, mu, sigma = nn_output
    if torch.is_tensor(pi):
        pi = pi.detach().cpu().numpy() 
        sigma = sigma.detach().cpu().numpy()
        mu = mu.detach().cpu().numpy()
    grid = plot_mog_heatmap(mu[0], sigma[0], pi[0], res)
    return grid

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)





def load_filter(filtering_model_config, filtering_model_path, device):
    with open(filtering_model_config, 'r') as stream:
        config = yaml.safe_load(stream)
    prior_or_combine = "prior_model"
    prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
    prior_or_combine = "combined_model"
    filtering_model = configure_model(config, prior_or_combine, prior_model)
    filtering_model.to(device)

    if filtering_model_path is not None:
        if isinstance(torch.load(filtering_model_path), OrderedDict):
            filtering_model.load_state_dict(torch.load(filtering_model_path))
        else:
            filtering_model = torch.load(filtering_model_path).to(device)
        
        print("Loaded filtering model from {}".format(filtering_model_path))
    return filtering_model

def traj_data_collector(env, blue_actions, robotarium_size=1.8, ROI_size=0.3):

    scale = robotarium_size / ROI_size
    robot_range = robotarium_size/2.0 + 0.01

    blue_agents_velocity = np.stack(blue_actions, axis=1)
    blue_agents_velocity[2,:] = blue_agents_velocity[2,:] / env.dim_x * robotarium_size
    blue_agents_normalized_velocity = blue_agents_velocity

    blue_agents_normalized_detection_range = np.stack([np.array(ag.detection_range) / env.dim_x * scale for ag in env.search_parties_list+env.helicopters_list], axis=0)

    blue_agents_localized_location = np.stack([np.clip((np.array(ag.location)-np.array(env.prisoner.location)) / env.dim_x * scale, -robot_range, robot_range) for ag in env.search_parties_list+env.helicopters_list], axis=1)

    blue_agents_normalized_location = np.stack([np.clip((np.array(ag.location)-env.dim_x/2.0) / env.dim_x * robotarium_size, -robot_range, robot_range) for ag in env.search_parties_list+env.helicopters_list], axis=1)

    prisoners_normalized_location = np.array(env.prisoner.location) / env.dim_x

    hideout_locations = np.stack(env.known_hideout_locations_on_map+env.unknown_hideout_locations_on_map, axis=0)

    return blue_agents_normalized_detection_range, blue_agents_normalized_velocity, blue_agents_localized_location, prisoners_normalized_location, hideout_locations

import torch

def temperature_softmax(logits, temperature=0.001):
    """
    Apply temperature-scaled softmax to input logits.

    Args:
        logits (torch.Tensor): Input logits.
        temperature (float): Temperature parameter for softmax.

    Returns:
        torch.Tensor: Softmax probabilities.
    """
    if temperature == 1.0:
        return torch.nn.functional.softmax(logits, dim=-1)
    else:
        scaled_logits = logits / temperature
        max_logits, _ = torch.max(scaled_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(scaled_logits - max_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
        probabilities = exp_logits / sum_exp_logits
        return probabilities

def identity_map(x):
    return x