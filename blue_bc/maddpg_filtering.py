import enum
import copy
import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from policy import LowLevelPolicy, MLPNetwork, MLPPolicyNetwork, HighLevelPolicy, HighLevelPolicyIndependent, AttentionMLPNetwork, AttentionEmbeddingNetwork, AttentionCriticNetwork, MLPPolicyNetwork, GNNNetwork, CNNNetwork
# from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, two_hot_encode
import yaml
from models.configure_model import configure_model
from blue_bc.maddpg import BaseMADDPG, BaseDDPG, BaseDDPGAgent, onehot_from_logits, gumbel_softmax, hard_update
from blue_bc.utils import convert_buffer_to_dataloader
from tqdm import tqdm

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')

class DDPGFiltering(BaseDDPG):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    device="cuda"):
        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma, tau, critic_lr, policy_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        self.filtering_model = configure_model(config)
        self.filtering_model.to(device)

        if filtering_model_path is not None:
            self.filtering_model.load_state_dict(torch.load(filtering_model_path))
            print("Loaded filtering model from {}".format(filtering_model_path))

        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = filtering_output_dim + num_in_critic # include filtering input with critic input # 2 + num_in_critic

        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [FilteringDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

    def step(self, observations, filtering_input, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        filtering_output = self.filtering_model(filtering_input)
        return [a.step(obs, filtering_output, explore=explore) for a, obs in zip(self.agents, observations)]

    def test_loc_est_error(self, normalized_prisoner_gt_loc, filtering_input):
        filtering_output = self.filtering_model(filtering_input)
        input_vf = self.convert_filtering_output_max_pi_mu_to_tensor(filtering_output)
        input_vf = input_vf.detach().cpu().numpy()
        return np.array(normalized_prisoner_gt_loc-input_vf)


    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        # print("pi = ", pi)
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        # print("cols = ", cols)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        # max_prob_mu_err = torch.norm((max_prob_mu - prisoner_loc), dim=-1)
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 4, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def get_localized_prisoner_loc_est(self, obs, filtering_output):
        agent_locations = obs[:, 1:3]
        localized_next_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
        localized_prisoner_loc_est = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
        return localized_prisoner_loc_est

    def update(self, sample, filtering_sample, agent_i, train_option="regular", logger=None):
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
        if train_option == "regular":
            obs, acs, rews, next_obs, dones = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        curr_agent = self.agents[agent_i]
        filtering_obs, next_filtering_obs, prisoner_loc = filtering_sample

        self.filtering_model.train()
        # localize each filtering output for the current observations
        f_filtering_output = self.filtering_model(filtering_obs)
        curr_filtering_loss = self.filtering_model.compute_loss(f_filtering_output, prisoner_loc)
        """Update the current filter"""
        curr_agent.filter_optimizer.zero_grad()
        curr_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.5)
        curr_agent.filter_optimizer.step()

        
        c_next_filtering_output = self.filtering_model(next_filtering_obs)

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            trgt_c_prisoner_loc_est = self.get_localized_prisoner_loc_est(next_obs[agent_i], c_next_filtering_output)

        trgt_vf_in = torch.cat((next_obs[agent_i], curr_agent.target_policy(torch.cat((next_obs[agent_i], trgt_c_prisoner_loc_est), dim=-1)), trgt_c_prisoner_loc_est), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)




        c_filtering_output = self.filtering_model(filtering_obs)
        c_prisoner_loc_est = self.get_localized_prisoner_loc_est(obs[agent_i], c_filtering_output)
        vf_in = torch.cat((obs[agent_i], acs[agent_i], c_prisoner_loc_est), dim=1)

        actual_value = curr_agent.critic(vf_in.detach()) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        # vf_filtering_loss = 0.01 * filtering_loss_vf + vf_loss
        vf_filtering_loss = vf_loss

        """Update the current critic"""
        curr_agent.critic_optimizer.zero_grad()
        vf_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #### ----------------------------Update Policy Below ----------------------------- ###

        p_filtering_output = self.filtering_model(filtering_obs)
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            p_prisoner_loc_est = self.get_localized_prisoner_loc_est(obs[agent_i], p_filtering_output)
            input_policy_tensor = torch.cat((obs[agent_i], p_prisoner_loc_est), dim=-1)
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        p_vf_in = torch.cat((obs[agent_i], curr_pol_vf_in, p_prisoner_loc_est.detach()), dim=1)

        pol_loss = -curr_agent.critic(p_vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(p_filtering_output, prisoner_loc)
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'curr_filtering_loss': curr_filtering_loss,
                                'max_pi': max_pi,
                                'mu_error_of_max_pi': mu_error_of_max_pi,
                                'sigma_of_max_pi': sigma_of_max_pi,
                                # 'filtering_loss_pol': filtering_loss_pol,
                                # 'filtering_loss_vf': filtering_loss_vf
                                },
                               self.niter)
        return td_error_abs_each

    def collect_distribution_error(self, filtering_output, prisoner_loc):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        # print("max_prob_mu = ", max_prob_mu[50])
        # print("prisoner_loc = ", prisoner_loc[50])
        max_prob_mu_err = torch.norm((max_prob_mu - prisoner_loc), dim=-1)
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols].mean(dim=-1)
        return torch.max(pi, dim=1)[0].mean(), max_prob_mu_err.mean(), max_prob_sigma.mean()

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents], 
                     'filtering_model': self.filtering_model.state_dict()}
        torch.save(save_dict, filename)

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])

    def init_filter_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])



class MADDPGFiltering(BaseMADDPG):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    filtering_model=None,
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    device="cuda"):

        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma, tau, critic_lr, policy_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        # prior_or_combine = "prior_model"
        # prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
        # prior_or_combine = "combined_model"
        # self.filtering_model = configure_model(config, prior_or_combine, prior_model)
        # self.filtering_model.to(device)
        
        # if filtering_model_path is not None:
        #     self.filtering_model.load_state_dict(torch.load(filtering_model_path))
        #     print("Loaded filtering model from {}".format(filtering_model_path))
        if filtering_model is not None:
            self.filtering_model = filtering_model.to(device)
            self.trgt_filtering_model = copy.deepcopy(self.filtering_model).to(device)
        
        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic

        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [FilteringDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

        self.filtering_loss = -6

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def get_filter_loss(self, filtering_input, prisoner_loc):
        curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_input), torch.Tensor(prisoner_loc).unsqueeze(0))
        filtering_loss = curr_filtering_loss.detach().cpu().numpy()
        return filtering_loss

    def step(self, observations, filtering_input, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """

        # filtering_output = self.filtering_model(filtering_input)
        filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_input))
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]

    def split_filtering_input(self, filtering_input):
        prior_input = filtering_input[..., 0:3]
        dynamic_input = filtering_input[..., 3:]
        sel_input = filtering_input
        return [prior_input, dynamic_input, sel_input]

    def split_new_pmc_input(self, pmc_input):
        prior_input = pmc_input[..., 0:3]
        dynamic_input = pmc_input[..., 3:6]
        sel_input = pmc_input[..., 6:]
        return [prior_input, dynamic_input, sel_input]

    def get_prior_input(self, filtering_input):
        prior_input = filtering_input[..., 0:3]
        return prior_input

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        # print("pi = ", pi)
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        # print("cols = ", cols)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def update_filter(self, replay_buffer, config, data_dir, curr_episode, epoch_num=100):
        train_data_loader = convert_buffer_to_dataloader(replay_buffer, config, data_dir=data_dir/str(curr_episode))
        batch_size = config["train"]["batch_size"]
        curr_agent = self.agents[0]
        for epoch in (range(1, epoch_num+1)):
            batch_loss = 0
            num_batches = 0
            for x_train, y_train in train_data_loader:
                if x_train.size(0) != batch_size:
                    continue
                num_batches += 1
                prior_input = x_train[...,0:3]
                curr_prior_loss = self.filtering_model.prior_network.compute_loss(prior_input, y_train)
                curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(x_train), y_train)
                curr_agent.filter_optimizer.zero_grad()
                curr_prior_loss.backward()
                curr_filtering_loss.backward()
                curr_agent.filter_optimizer.step()
                batch_loss += curr_filtering_loss.item()   

    def update(self, sample, agent_i, train_option="regular", logger=None):
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
        if train_option == "regular":
            # obs, acs, rews, next_obs, dones = sample
            obs, acs, rews, next_obs, dones, filtering_obs, next_filtering_obs, prisoner_loc = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        # filtering_obs, next_filtering_obs, prisoner_loc = filtering_sample


        self.filtering_model.train()

        curr_agent = self.agents[agent_i]
        
        # next_filtering_output = self.filtering_model(next_filtering_obs[agent_i])
        # next_prior_output = self.filtering_model.prior_network(self.get_prior_input(next_filtering_obs[agent_i]))
        next_filtering_output = self.trgt_filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs in zip(self.target_policies, next_obs):
                agent_locations = nobs[:, 1:3]
                localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                next_input = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
                input_tensor = torch.cat((nobs, next_input), dim=-1)
                all_trgt_acs.append(pi(input_tensor))
                all_next_filter_ins.append(next_input)

        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs, *all_next_filter_ins), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)


        # localize each filtering output for the current observations
        filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc)
        # curr_filtering_loss = self.filtering_model.compute_loss(filtering_obs[agent_i], prisoner_loc[agent_i])
        curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # self.filtering_loss = curr_filtering_loss.detach()
        # target_value = target_value + ((-curr_filtering_loss) - 6)
        """Update the current filter"""
        # if agent_i == 0:
        curr_agent.filter_optimizer.zero_grad()
        curr_prior_loss.backward()
        curr_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
        curr_agent.filter_optimizer.step()

        
        all_filter_ins = []
        for ob in obs:
            agent_locations = ob[:, 1:3]
            localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
            all_filter_ins.append(input_vf.detach())

        # need to include filtering output to vf input here as well
        vf_in = torch.cat((*obs, *acs, *all_filter_ins), dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        # vf_filtering_loss = 0.01 * filtering_loss_vf + vf_loss
        vf_filtering_loss = vf_loss

        """Update the current critic"""
        curr_agent.critic_optimizer.zero_grad()
        vf_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #### ----------------------------Update Policy Below ----------------------------- ###

        filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            agent_locations = obs[agent_i][:, 1:3]
            localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            localized_filter_tensor = self.convert_filtering_output_to_tensor(localized_filtering_out)

            input_policy_tensor = torch.cat((obs[agent_i], localized_filter_tensor), dim=-1)
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
                all_pol_filter_ins.append(localized_filter_tensor.detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                agent_locations = ob[:, 1:3]
                localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                input_tensor_policy = torch.cat((ob, input_filter), dim=-1)
                all_pol_acs.append(pi(input_tensor_policy).detach())
                all_pol_filter_ins.append(input_filter.detach())

        vf_in = torch.cat((*obs, *all_pol_acs, *all_pol_filter_ins), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'curr_filtering_loss': curr_filtering_loss,
                                'max_pi': max_pi,
                                'mu_error_of_max_pi': mu_error_of_max_pi,
                                'sigma_of_max_pi': sigma_of_max_pi,
                                # 'filtering_loss_pol': filtering_loss_pol,
                                # 'filtering_loss_vf': filtering_loss_vf
                                },
                               self.niter)
        return td_error_abs_each

    # def mu_to_local_frame(self, mu, obs):
    #     batch_size = mu.shape[0]
    #     agents_normalizedLoc = obs[:, 1:3]
    #     agents_normalizedLoc = agents_normalizedLoc.unsqueeze(1).repeat(1, 4, 1)
    #     mu_in_local_frame = mu - agents_normalizedLoc
    #     mu_in_local_frame = mu_in_local_frame.reshape(batch_size, -1)
    #     # mus = torch.split(mu, split_size_or_sections=)
    #     # agents_normalizedLoc = np.concatenate([obs[i][:,1:3] for i in range(self.nagents)], axis=1)
    #     # mu_in_local_frame = mu - agents_normalizedLoc
    #     return mu_in_local_frame

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        soft_update(self.trgt_filtering_model, self.filtering_model, self.tau)
        self.niter += 1

    def collect_distribution_error(self, filtering_output, prisoner_loc):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        # print("max_prob_mu = ", max_prob_mu[50])
        # print("prisoner_loc = ", prisoner_loc[50])
        max_prob_mu_err = torch.norm((max_prob_mu - prisoner_loc), dim=-1)
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols].mean(dim=-1)
        return torch.max(pi, dim=1)[0].mean(), max_prob_mu_err.mean(), max_prob_sigma.mean()

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.policy = fn(a.policy)
        self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents], 
                     'filtering_model': self.filtering_model.state_dict()}
        torch.save(save_dict, filename)

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)
            # a.policy.to(self.pol_dev)
            # a.critic.to(self.critic_dev)
            # a.target_policy.to(self.trgt_critic_dev)
            # a.target_critic.to(self.trgt_critic_dev)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])
        # self.filtering_model.to(self.pol_dev)

class MADDPGCommFiltering(MADDPGFiltering):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    comm_dim,
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    filtering_model=None,
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    device="cuda"):

        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, filtering_model_config, filtering_model_path, filtering_model, gamma, tau, critic_lr, policy_lr, filter_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        # prior_or_combine = "prior_model"
        # prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
        # prior_or_combine = "combined_model"
        # self.filtering_model = configure_model(config, prior_or_combine, prior_model)
        # self.filtering_model.to(device)

        # if filtering_model_path is not None:
        #     self.filtering_model.load_state_dict(torch.load(filtering_model_path))
        #     print("Loaded filtering model from {}".format(filtering_model_path))
        self.filtering_model = filtering_model.to(device)
        self.trgt_filtering_model = copy.deepcopy(self.filtering_model).to(device)

        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic
        self.comm_dim = comm_dim
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [FilteringCommDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, comm_dim=self.comm_dim, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

        self.filtering_loss = -6

    def step(self, obs_filter_out, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """

        # filtering_output = self.filtering_model(filtering_input)
        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_input))
        agents_actions = []
        agents_comms = []
        agents_actionsComms = []
        for a, obs in zip(self.agents, obs_filter_out):
            agent_actions, agent_comms = a.step(obs, explore=explore)
            agents_actions.append(agent_actions)
            agents_comms.append(agent_comms)
            agents_actionsComms.append(torch.cat((agent_actions, agent_comms), dim=-1))
        return agents_actions, agents_comms

    # def split_action_comm(self, agent_actionsComms):

    def update(self, sample, agent_i, train_option="regular", logger=None):
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
        if train_option == "regular":
            # obs, acs, rews, next_obs, dones = sample
            obs, acs, rews, next_obs, dones, filtering_obs, next_filtering_obs, prisoner_loc = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        # filtering_obs, next_filtering_obs, prisoner_loc = filtering_sample


        self.filtering_model.train()

        curr_agent = self.agents[agent_i]
        
        # next_filtering_output = self.filtering_model(next_filtering_obs[agent_i])
        # next_prior_output = self.filtering_model.prior_network(self.get_prior_input(next_filtering_obs[agent_i]))
        # next_filtering_output = self.trgt_filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs in zip(self.target_policies, next_obs):
                # agent_locations = nobs[:, 1:3]
                # localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                # next_input = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
                # input_tensor = torch.cat((nobs, next_input), dim=-1)
                input_tensor = nobs
                all_trgt_acs.append(torch.cat(pi(input_tensor), dim=-1))
                # all_next_filter_ins.append(next_input)

        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)


        # localize each filtering output for the current observations
        filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc)
        # curr_filtering_loss = self.filtering_model.compute_loss(filtering_obs[agent_i], prisoner_loc[agent_i])
        curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # self.filtering_loss = curr_filtering_loss.detach()
        # target_value = target_value + ((-curr_filtering_loss) - 6)

        """Update the current filter"""
        # if agent_i == 0:
        # curr_agent.filter_optimizer.zero_grad()
        # # curr_prior_loss.backward()
        # curr_filtering_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.01)
        # curr_agent.filter_optimizer.step()

        
        # all_filter_ins = []
        # for ob in obs:
        #     agent_locations = ob[:, 1:3]
        #     localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
        #     input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
        #     all_filter_ins.append(input_vf.detach())

        # need to include filtering output to vf input here as well
        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        # vf_filtering_loss = 0.01 * filtering_loss_vf + vf_loss
        vf_filtering_loss = vf_loss
        # print("vf_filtering_loss = ", vf_filtering_loss)

        """Update the current critic"""
        curr_agent.critic_optimizer.zero_grad()
        vf_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #### ----------------------------Update Policy Below ----------------------------- ###

        filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            # agent_locations = obs[agent_i][:, 1:3]
            # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            # localized_filter_tensor = self.convert_filtering_output_to_tensor(localized_filtering_out)

            # input_policy_tensor = torch.cat((obs[agent_i], localized_filter_tensor), dim=-1)
            input_policy_tensor = obs[agent_i]
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(torch.cat(curr_pol_vf_in, dim=-1))
                # all_pol_filter_ins.append(localized_filter_tensor.detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                # agent_locations = ob[:, 1:3]
                # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                # input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                # input_tensor_policy = torch.cat((ob, input_filter), dim=-1)
                input_tensor_policy = ob
                all_pol_acs.append(torch.cat(pi(input_tensor_policy), dim=-1).detach()) 
                # all_pol_filter_ins.append(input_filter.detach())

        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        # print("pol_loss = ", pol_loss)
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'curr_filtering_loss': curr_filtering_loss,
                                'max_pi': max_pi,
                                'mu_error_of_max_pi': mu_error_of_max_pi,
                                'sigma_of_max_pi': sigma_of_max_pi,
                                # 'filtering_loss_pol': filtering_loss_pol,
                                # 'filtering_loss_vf': filtering_loss_vf
                                },
                               self.niter)
        return td_error_abs_each

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents], 
                     'filtering_model': self.filtering_model.state_dict(),
                     'trgt_filtering_model': self.trgt_filtering_model.state_dict()}
        torch.save(save_dict, filename)

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])
        self.trgt_filtering_model.load_state_dict(save_dict['trgt_filtering_model'])

class AttentionMADDPGFiltering(MADDPGFiltering):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    device="cuda"):    
        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, filtering_model_config, filtering_model_path, gamma, tau, critic_lr, policy_lr, filter_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = filtering_output_dim + num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = filtering_output_dim * 6 + num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic

        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [AttentionFileringDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

class ActorAttentionCritcFiltering(MADDPGFiltering):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    device="cuda"):    
        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, filtering_model_config, filtering_model_path, gamma, tau, critic_lr, policy_lr, filter_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = filtering_output_dim + num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = filtering_output_dim * 6 + num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic

        self.gaussian_num = config["combined_model"]["number_gaussians"]
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.device = device
        self.agents = [AACFileringDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        return torch.cat((pi, mu, sigma), dim=-1)

    def convert_filtering_output_to_each_agent_tensor(self, filtering_output):
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
        return torch.cat(agents_pi_mu_sigma, dim=-1)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_critic_embedding, a.critic_embedding, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def update(self, sample, agent_i, train_option="regular", logger=None):
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
        if train_option == "regular":
            # obs, acs, rews, next_obs, dones = sample
            obs, acs, rews, next_obs, dones, filtering_obs, next_filtering_obs, prisoner_loc = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        self.filtering_model.train()

        curr_agent = self.agents[agent_i]

        next_filtering_output = self.filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            all_next_trgt_embeddings = []
            for i, emb, pi, nobs in zip(range(self.nagents), self.target_embeddings_generator, self.target_policies, next_obs):
                agent_locations = nobs[:, 1:3]
                localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                next_input = self.convert_filtering_output_to_each_agent_tensor(localized_next_filtering_out)
                input_tensor = torch.cat((nobs, next_input), dim=-1)

                curr_agent_trgt_next_action = pi(input_tensor)
                curr_agent_next_filter_in = next_input
                # all_trgt_acs.append(pi(input_tensor))
                all_next_filter_ins.append(next_input)
                curr_agent_trgt_critic_next_emb_in = torch.cat((nobs, curr_agent_trgt_next_action, curr_agent_next_filter_in), dim=-1)
                curr_agent_trgt_critic_next_embeddings = emb(curr_agent_trgt_critic_next_emb_in)
                if i == agent_i:
                    all_next_trgt_embeddings.append(curr_agent_trgt_critic_next_embeddings)



        trgt_vf_in = torch.cat(all_next_trgt_embeddings, dim=1) # (pi, next_obs, target_critic)->reward_prediction


        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)


        # localize each filtering output for the current observations
        filtering_output = self.filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc)
        # curr_filtering_loss = self.filtering_model.compute_loss(filtering_obs[agent_i], prisoner_loc[agent_i])
        curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        """Update the current filter"""
        # if agent_i == 0:
        curr_agent.filter_optimizer.zero_grad()
        curr_prior_loss.backward()
        curr_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
        curr_agent.filter_optimizer.step()

        
        all_filter_ins = []
        all_trgt_embeddings = []
        for i, emb, ob, ac in zip(range(self.nagents), self.embeddings_generator, obs, acs):
            agent_locations = ob[:, 1:3]
            localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            input_vf = self.convert_filtering_output_to_each_agent_tensor(localized_filtering_out)
            # all_filter_ins.append(input_vf.detach())


            curr_agent_trgt_action = ac
            curr_agent_filter_in = input_vf.detach()
            # all_next_filter_ins.append(next_input)
            curr_agent_trgt_next_emb_in = torch.cat((ob, curr_agent_trgt_action, curr_agent_filter_in), dim=-1)
            curr_agent_trgt_critic_embeddings = emb(curr_agent_trgt_next_emb_in)
            # if i != agent_i:
            #     curr_agent_trgt_critic_embeddings = curr_agent_trgt_critic_embeddings.detach()
            if i == agent_i:
                all_trgt_embeddings.append(curr_agent_trgt_critic_embeddings)



        # need to include filtering output to vf input here as well
        vf_in = torch.cat(all_trgt_embeddings, dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        # vf_filtering_loss = 0.01 * filtering_loss_vf + vf_loss
        vf_filtering_loss = vf_loss

        """Update the current critic"""
        curr_agent.critic_optimizer.zero_grad()
        vf_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #### ----------------------------Update Policy Below ----------------------------- ###

        filtering_output = self.filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            agent_locations = obs[agent_i][:, 1:3]
            localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            localized_filter_tensor = self.convert_filtering_output_to_each_agent_tensor(localized_filtering_out)

            input_policy_tensor = torch.cat((obs[agent_i], localized_filter_tensor), dim=-1)
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        all_pol_embeddings = []
        for i, pi, emb, ob in zip(range(self.nagents), self.policies, self.embeddings_generator, obs):
            if i == agent_i:
                pol_acs = curr_pol_vf_in
                pol_filter_in = localized_filter_tensor.detach()
                pol_emb_in = torch.cat((ob, pol_acs, pol_filter_in), dim=-1)
                pol_embeddings = emb(pol_emb_in)
                all_pol_embeddings.append(pol_embeddings)
                # all_pol_acs.append(curr_pol_vf_in)
                # all_pol_filter_ins.append(localized_filter_tensor.detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            # else:
                # agent_locations = ob[:, 1:3]
                # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                # input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                # input_tensor_policy = torch.cat((ob, input_filter), dim=-1)

                # # all_pol_acs.append(pi(input_tensor_policy).detach())
                # # all_pol_filter_ins.append(input_filter.detach())

                # pol_acs = pi(input_tensor_policy).detach()
                # pol_filter_in = input_filter.detach()

                # pol_emb_in = torch.cat((ob, pol_acs, pol_filter_in), dim=-1)
                # pol_embeddings = emb(pol_emb_in)
                # all_pol_embeddings.append(pol_embeddings)

        vf_in = torch.cat(all_pol_embeddings, dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'curr_filtering_loss': curr_filtering_loss,
                                'max_pi': max_pi,
                                'mu_error_of_max_pi': mu_error_of_max_pi,
                                'sigma_of_max_pi': sigma_of_max_pi,
                                # 'filtering_loss_pol': filtering_loss_pol,
                                # 'filtering_loss_vf': filtering_loss_vf
                                },
                               self.niter)
        return td_error_abs_each

    @property
    def target_embeddings_generator(self):
        return [a.critic_embedding.to(self.device) for a in self.agents]

    @property
    def embeddings_generator(self):
        return [a.target_critic_embedding.to(self.device) for a in self.agents]

class FilteringDDPGAgent(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    comm_dim=0,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)
        self.policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                 comm_dim=comm_dim,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 device=device,
                                #  discrete_action=discrete_action
                                 ).to(device)
        self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                        comm_dim=comm_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        device=device,
                                        # discrete_action=discrete_action
                                        ).to(device)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic) # should be handled already

        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def step(self, obs_filter_out, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # localized_filtering_out = self.localize_filtering_mu(filtering_output, obs[1:3].unsqueeze(0))
        # input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
        # print("input_vf = ", input_vf)
        tensor_in = torch.Tensor(obs_filter_out).to(self.device) if type(obs_filter_out) == np.ndarray else obs_filter_out
        action = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])
         
class FilteringCommDDPGAgent(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    comm_dim=32,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)
        self.hs_state = 0
        self.steps_per_side = 5
        self.steps_already_taken = 0
        self.device = device
        self.policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                 comm_dim=comm_dim,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 device=device
                                 ).to(device)
        self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                        comm_dim=comm_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        device=device
                                        ).to(device)
        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def heuristic_search(self, speed):
        if self.hs_state == 0:
            vel = torch.Tensor([0, 1]).to(self.device) * speed
            self.steps_already_taken = self.steps_already_taken + 1
            if self.steps_already_taken >= self.steps_per_side:
                self.hs_state = 1
        elif self.hs_state == 1:
            vel = torch.Tensor([0, -1]).to(self.device) * speed
            self.steps_already_taken = self.steps_already_taken + 1
            if self.steps_already_taken >= self.steps_per_side:
                self.hs_state = 2
        elif self.hs_state == 2:
            vel = torch.Tensor([-1, 0]).to(self.device) * speed
            self.steps_already_taken = self.steps_already_taken + 1
            if self.steps_already_taken >= self.steps_per_side:
                self.hs_state = 3
        elif self.hs_state == 3:
            vel = torch.Tensor([1, 0]).to(self.device) * speed
            self.steps_already_taken = self.steps_already_taken + 1
            if self.steps_already_taken >= self.steps_per_side:
                self.hs_state = 0
        else:
            raise NotImplementedError
        return vel

    def step(self, obs_filter_out, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # localized_filtering_out = self.localize_filtering_mu(filtering_output, obs[1:3].unsqueeze(0))
        # input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
        # # print("input_vf = ", input_vf)

        # tensor_in = torch.concat([obs, input_vf.squeeze()], dim=-1)
        tensor_in = torch.Tensor(obs_filter_out).to(self.device) if type(obs_filter_out) == np.ndarray else obs_filter_out
        out_act, out_comm = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                out_act = gumbel_softmax(out_act.unsqueeze(0), hard=True).squeeze()
            else:
                out_act = onehot_from_logits(out_act.unsqueeze(0)).squeeze()
        else:  # continuous action
            if explore:
                out_act = out_act + Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            out_act = out_act.clamp(-1, 1)
        return out_act, out_comm

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class FilteringCommDDPGAgentHier(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    comm_dim=32,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)
        self.policy = MLPPolicyNetwork(num_in_pol, 
                                    num_out_pol,
                                    comm_dim=comm_dim,
                                    hidden_dim=hidden_dim,
                                    constrain_out=True,
                                    discrete_action=discrete_action,
                                    device=device
                                    ).to(device)
        self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                        comm_dim=comm_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        device=device
                                        ).to(device)
        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

        # save high and low level actions into agent class property
        self.high_level_action = np.array([0,0,0,0,0,0,0,1])
        self.low_level_action = np.array([0,0,0,0,0,0,0,1])

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def step(self, obs, filtering_output, level="high", explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        input_vf = filtering_output
        # print("input_vf = ", input_vf)
        if level == "high":
            tensor_in = torch.concat([obs.unsqueeze(0), *input_vf], dim=-1).squeeze()
        elif level == "low":
            tensor_in = torch.concat([obs.unsqueeze(0), input_vf], dim=-1).squeeze()

        out_act, out_comm = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                out_act = gumbel_softmax(out_act.unsqueeze(0), hard=True).squeeze()
            else:
                out_act = onehot_from_logits(out_act.unsqueeze(0)).squeeze()
        else:  # continuous action
            if explore:
                out_act = out_act + Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            out_act = out_act.clamp(-1, 1)

        if level == "high":
            self.high_level_action = out_act.detach().cpu().numpy()
        elif level == "low":
            self.low_level_action = out_act.detach().cpu().numpy()
        return out_act, out_comm

    def init_step(self, obs, filtering_output, level="high", explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        out_act, out_comm = self.step(obs, filtering_output, level="high", explore=False)
        self.high_level_action = np.array([0,0,0,0,0,0,0,1])
        self.low_level_action = np.array([0,0,0,0,0,0,0,1])
        return out_act, out_comm

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class FilteringCommDDPGAgentHierMap(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    comm_dim=32,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)
        self.policy = CNNNetwork(num_in_pol, 
                                    num_out_pol,
                                    hidden_dim=hidden_dim,
                                    constrain_out=True,
                                    # discrete_action=discrete_action,
                                    # device=device
                                    ).to(device)
        self.target_policy = CNNNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        # discrete_action=discrete_action,
                                        # device=device
                                        ).to(device)
        self.critic = CNNNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                #  discrete_action=discrete_action,
                                #  device=device
                                 ).to(device)
        self.target_critic = CNNNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                #  discrete_action=discrete_action,
                                #  device=device
                                 ).to(device)
        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic) # should be handled already

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def obs_to_map(self, observation):
        map = observation.view(-1, 6, 15, 15)
        return map

    def step(self, obs, filtering_output, level="high", explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        input_vf = filtering_output
        # print("input_vf = ", input_vf)
        if level == "high":
            tensor_in = self.obs_to_map(obs)
        elif level == "low":
            tensor_in = self.obs_to_map(obs)

        out_act, out_comm = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                out_act = gumbel_softmax(out_act.unsqueeze(0), hard=True).squeeze()
            else:
                out_act = onehot_from_logits(out_act.unsqueeze(0)).squeeze()
        else:  # continuous action
            if explore:
                out_act = out_act + Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            out_act = out_act.clamp(-1, 1)
        return out_act, out_comm

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class FilteringCommDDPGAgentHierGNN(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    comm_dim=32,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)


        
        # self.policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
        #                          comm_dim=comm_dim,
        #                          hidden_dim=hidden_dim,
        #                          constrain_out=True,
        #                          discrete_action=discrete_action,
        #                          device=device
        #                          ).to(device)
        # self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
        #                                 comm_dim=comm_dim,
        #                                 hidden_dim=hidden_dim,
        #                                 constrain_out=True,
        #                                 discrete_action=discrete_action,
        #                                 device=device
        #                                 ).to(device)
        # include filtering model parameters here
        
        
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def step(self, obs, filtering_output, level="high", explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        input_vf = filtering_output
        # print("input_vf = ", input_vf)
        if level == "high":
            tensor_in = torch.concat([obs.unsqueeze(0), *input_vf], dim=-1).squeeze()
        elif level == "low":
            tensor_in = torch.concat([obs.unsqueeze(0), input_vf], dim=-1).squeeze()

        out_act, out_comm = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                out_act = gumbel_softmax(out_act.unsqueeze(0), hard=True).squeeze()
            else:
                out_act = onehot_from_logits(out_act.unsqueeze(0)).squeeze()
        else:  # continuous action
            if explore:
                out_act = out_act + Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            out_act = out_act.clamp(-1, 1)
        return out_act, out_comm

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class FilteringCommDDPGAgentHierGNNv2(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)


        
        self.policy = GNNNetwork(input_dim = 2, 
                                 concat_dim = 10,
                                 out_dim = num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 ).to(device)
        
        self.target_policy = GNNNetwork(
                                        input_dim = 2,
                                        concat_dim = 10,
                                        out_dim = num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        ).to(device)

        self.critic = GNNNetwork(input_dim = 4, 
                                 concat_dim = 10,
                                 out_dim = 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 ).to(device)
        
        self.target_critic = GNNNetwork(
                                        input_dim = 4,
                                        concat_dim = 10,
                                        out_dim = 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        ).to(device)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)
        self.discrete_action = False
    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def step(self, graph_obs, rest_obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        # input_vf = filtering_output
        # print("input_vf = ", input_vf)
        # if level == "high":
            # tensor_in = torch.concat([obs.unsqueeze(0), *input_vf], dim=-1).squeeze()
        # elif level == "low":
        
        # Level is always low
        # tensor_in = torch.concat([obs.unsqueeze(0), input_vf], dim=-1).squeeze()

        out_act = self.policy(graph_obs, rest_obs)
        if self.discrete_action:
            if explore:
                out_act = gumbel_softmax(out_act.unsqueeze(0), hard=True).squeeze()
            else:
                out_act = onehot_from_logits(out_act.unsqueeze(0)).squeeze()
        else:  # continuous action
            if explore:
                out_act = out_act + Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            out_act = out_act.clamp(-1, 1)
        return out_act

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class AttentionFileringDDPGAgent(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)

        self.policy = AttentionMLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)
        self.target_policy = AttentionMLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, constrain_out=True)

        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((self.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sorted_idx = torch.argsort(pi, axis=1)
        pi=torch.sort(pi, axis=1).values
        sigma = sigma[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        mu = mu[np.expand_dims(np.arange(0,batch_size), axis=1).repeat(8, axis=1),sorted_idx,:]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)

        return torch.cat((pi, mu, sigma), dim=-1)

    def step(self, obs, filtering_output, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        localized_filtering_out = self.localize_filtering_mu(filtering_output, obs[1:3].unsqueeze(0))
        input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
        # print("input_vf = ", input_vf)

        tensor_in = torch.concat([obs, input_vf.squeeze()], dim=-1)

        action = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class AACFileringDDPGAgent(BaseDDPGAgent):
    def __init__(self, num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    filtering_model,
                    hidden_dim=64, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    discrete_action=True, 
                    device="cpu"):
        super().__init__(num_in_pol, num_out_pol, num_in_critic, hidden_dim, critic_lr, policy_lr, discrete_action, device)

        self.critic_embedding = AttentionEmbeddingNetwork(int(num_in_critic / 6), num_out_pol, hidden_dim=hidden_dim, constrain_out=False)
        self.target_critic_embedding = AttentionEmbeddingNetwork(int(num_in_critic / 6), num_out_pol, hidden_dim=hidden_dim, constrain_out=False)
        self.critic = AttentionCriticNetwork(hidden_dim, num_out_pol, hidden_dim=hidden_dim, constrain_out=False)
        self.target_critic = AttentionCriticNetwork(hidden_dim, num_out_pol, hidden_dim=hidden_dim, constrain_out=False)
        

        # include filtering model parameters here
        self.policy_optimizer = torch.optim.Adam((self.policy.parameters()), lr=policy_lr) # original: lr
        self.critic_optimizer = torch.optim.Adam((list(self.critic.parameters())+list(self.critic_embedding.parameters())), lr=critic_lr) #  + list(filtering_model.parameters())
        self.filter_optimizer = torch.optim.Adam((filtering_model.parameters()), lr=filter_lr, weight_decay=1e-5)

    def convert_filtering_output_max_pi_mu_to_tensor(self, filtering_output):
        pi, mu, sigma = filtering_output
        batch_size = pi.shape[0]
        sigma = sigma.reshape(batch_size, -1)
        mu = mu.reshape(batch_size, -1)
        max_prob_idices = torch.argmax(pi, dim=1, keepdim=True)
        cols = torch.cat([2*max_prob_idices, 2*max_prob_idices+1], dim=-1)
        rows = torch.cat([torch.arange(batch_size).unsqueeze(1), torch.arange(batch_size).unsqueeze(1)], dim=-1)
        max_prob_mu = mu.view(batch_size, -1)[rows, cols]
        max_prob_sigma = sigma.view(batch_size, -1)[rows, cols]
        max_prob_mu_sigma = torch.cat((max_prob_mu, max_prob_sigma), dim=-1)
        return max_prob_mu_sigma

    def localize_filtering_mu(self, filtering_output, agent_location):
        pi, mu, sigma = filtering_output
        agent_location = agent_location.unsqueeze(1).repeat(1, 8, 1)
        mu_localize = mu - agent_location
        return pi, mu_localize, sigma

    def convert_filtering_output_to_each_agent_tensor(self, filtering_output):
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
        return torch.cat(agents_pi_mu_sigma, dim=-1)

    def step(self, obs, filtering_output, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        localized_filtering_out = self.localize_filtering_mu(filtering_output, obs[1:3].unsqueeze(0))
        input_vf = self.convert_filtering_output_to_each_agent_tensor(localized_filtering_out)
        # print("input_vf = ", input_vf)

        tensor_in = torch.concat([obs, input_vf.squeeze()], dim=-1)

        action = self.policy(tensor_in)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()).to(self.device),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

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