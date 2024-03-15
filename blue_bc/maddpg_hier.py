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
from policy import LowLevelPolicy, MLPNetwork, HighLevelPolicy, HighLevelPolicyIndependent, AttentionMLPNetwork, AttentionEmbeddingNetwork, AttentionCriticNetwork, MLPPolicyNetwork
# from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, two_hot_encode
import yaml
from models.configure_model import configure_model
from blue_bc.maddpg import BaseMADDPG, BaseDDPG, BaseDDPGAgent, onehot_from_logits, gumbel_softmax, hard_update
from blue_bc.maddpg_filtering import MADDPGFiltering, FilteringCommDDPGAgent, FilteringCommDDPGAgentHier, FilteringCommDDPGAgentHierMap
from blue_bc.utils import convert_buffer_to_dataloader
from tqdm import tqdm
from blue_bc.maddpg_filtering import FilteringDDPGAgent

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')

class MADDPGCommFilteringHier(MADDPGFiltering):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    comm_dim,
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    level="high",
                    device="cuda"):

        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, filtering_model_config, filtering_model_path, gamma, tau, critic_lr, policy_lr, filter_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        prior_or_combine = "prior_model"
        prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
        prior_or_combine = "combined_model"
        self.filtering_model = configure_model(config, prior_or_combine, prior_model)
        self.filtering_model.to(device)

        if filtering_model_path is not None:
            self.filtering_model.load_state_dict(torch.load(filtering_model_path))
            print("Loaded filtering model from {}".format(filtering_model_path))

        self.trgt_filtering_model = copy.deepcopy(self.filtering_model)

        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 if level == "high" else 1 * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5

        agent_input_dim = filtering_output_dim + num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = filtering_output_dim * 6 + num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic
        self.comm_dim = comm_dim
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [FilteringCommDDPGAgentHier(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, comm_dim=self.comm_dim, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        
        # self.policy = [MLPPolicyNetwork(agent_input_dim, num_out_pol,
        #                             comm_dim=comm_dim,
        #                             hidden_dim=hidden_dim,
        #                             constrain_out=True,
        #                             discrete_action=discrete_action,
        #                             device=device
        #                             ).to(device) for _ in range(2)] 
        # self.critic = [MLPNetwork(num_in_critic_filter, 1,
        #                          hidden_dim=hidden_dim,
        #                          constrain_out=False) for _ in range(2)]   

        # for agent_i, agent in enumerate(self.agents):
        #     if agent_i < 5:
        #         agent.policy = self.policy[0]
        #         hard_update(agent.target_policy, agent.policy)
        #         agent.critic = self.critic[0]
        #         hard_update(agent.target_critic, agent.critic)
        #         agent.policy_optimizer = torch.optim.Adam((agent.policy.parameters()), lr=policy_lr) # original: lr
        #         agent.critic_optimizer = torch.optim.Adam((agent.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())
                
        #     else:
        #         agent.policy = self.policy[1]
        #         hard_update(agent.target_policy, agent.policy)
        #         agent.critic = self.critic[1]
        #         hard_update(agent.target_critic, agent.critic)
        #         agent.policy_optimizer = torch.optim.Adam((agent.policy.parameters()), lr=policy_lr) # original: lr
        #         agent.critic_optimizer = torch.optim.Adam((agent.critic.parameters()), lr=critic_lr) #  + list(filtering_model.parameters())  

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

    def step(self, observations, filtering_output, level="high", explore=False):
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
        for agent_i, (a, obs) in enumerate(zip(self.agents, observations)):
            if level == "high":
                agent_actions, agent_comms = a.step(obs, filtering_output, level=level, explore=explore)
            elif level=="low":
                agent_actions, agent_comms = a.step(obs, filtering_output[agent_i], level=level, explore=explore)
            agents_actions.append(agent_actions)
            agents_comms.append(agent_comms)
            agents_actionsComms.append(torch.cat((agent_actions, agent_comms), dim=-1))
        return agents_actions, agents_comms, filtering_output

    def init_step(self, observations, filtering_output, level="high", explore=False):
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
        for agent_i, (a, obs) in enumerate(zip(self.agents, observations)):
            if level == "high":
                agent_actions, agent_comms = a.init_step(obs, filtering_output, level=level, explore=explore)
            elif level=="low":
                agent_actions, agent_comms = a.init_step(obs, filtering_output[agent_i], level=level, explore=explore)
            agents_actions.append(agent_actions)
            agents_comms.append(agent_comms)
            agents_actionsComms.append(torch.cat((agent_actions, agent_comms), dim=-1))
        return agents_actions, agents_comms, filtering_output

    # def split_action_comm(self, agent_actionsComms):

    def update(self, sample, agent_i, level="high", train_option="regular", train_mode="maddpg", logger=None):
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
            obs, acs, rews, next_obs, dones, filtering_out, next_filtering_out, prisoner_loc = sample
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
            # all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
            #                 zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs, nfo in zip(self.target_policies, next_obs, next_filtering_out):
                agent_locations = nobs[:, 1:3]
                input_tensor = torch.cat((nobs, nfo), dim=-1)
                all_trgt_acs.append(torch.cat((onehot_from_logits(pi(input_tensor)[0]), pi(input_tensor)[1]), dim=-1))
                all_next_filter_ins.append(nfo)
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs, nfo in zip(self.target_policies, next_obs, next_filtering_out):
                agent_locations = nobs[:, 1:3]
                # localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                # next_input = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
                input_tensor = torch.cat((nobs, nfo), dim=-1)
                all_trgt_acs.append(torch.cat(pi(input_tensor), dim=-1))
                all_next_filter_ins.append(nfo)
        if train_mode == "maddpg":
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs, *all_next_filter_ins), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        elif train_mode == "ddpg":
            trgt_vf_in = torch.cat((next_obs[agent_i], all_trgt_acs[agent_i], all_next_filter_ins[agent_i]), dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        # """Update the current filter"""
        # localize each filtering output for the current observations
        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_agent.filter_optimizer.zero_grad()
        # # curr_prior_loss.backward()
        # curr_filtering_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
        # curr_agent.filter_optimizer.step()
        
        all_filter_ins = []
        for fo in filtering_out:
            # agent_locations = ob[:, 1:3]
            # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            # input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
            all_filter_ins.append(fo.detach())

        # need to include filtering output to vf input here as well
        if train_mode == "maddpg":
            vf_in = torch.cat((*obs, *acs, *all_filter_ins), dim=1)
        elif train_mode == "ddpg":
            vf_in = torch.cat((obs[agent_i], acs[agent_i], all_filter_ins[agent_i]), dim=1)
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

        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            
            curr_pol_out = curr_agent.policy(torch.cat((obs[agent_i], filtering_out[agent_i]), dim=-1).detach())
            curr_pol_vf_in = (gumbel_softmax(curr_pol_out[0], hard=True), curr_pol_out[1])
        else:
            agent_locations = obs[agent_i][:, 1:3]
            # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            # localized_filter_tensor = self.convert_filtering_output_to_tensor(localized_filtering_out)

            input_policy_tensor = torch.cat((obs[agent_i], filtering_out[agent_i]), dim=-1)
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(torch.cat(curr_pol_vf_in, dim=-1))
                all_pol_filter_ins.append(filtering_out[i].detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                input_tensor = torch.cat((ob, filtering_out[i]), dim=-1)
                all_pol_acs.append(torch.cat((onehot_from_logits(pi(input_tensor)[0]), pi(input_tensor)[1]), dim=-1))
                all_pol_filter_ins.append(filtering_out[i].detach()) # add the localized filtering input of current agent
            else:
                agent_locations = ob[:, 1:3]
                # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                # input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                input_tensor_policy = torch.cat((ob, filtering_out[i]), dim=-1)
                all_pol_acs.append(torch.cat(pi(input_tensor_policy), dim=-1).detach()) 
                all_pol_filter_ins.append(filtering_out[i].detach())

        if train_mode == "maddpg":
            vf_in = torch.cat((*obs, *all_pol_acs, *all_pol_filter_ins), dim=1)
        elif train_mode == "ddpg":
            vf_in = torch.cat((obs[agent_i], all_pol_acs[agent_i], all_pol_filter_ins[agent_i]), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/%s_losses' % (agent_i, level),
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                # 'curr_filtering_loss': curr_filtering_loss,
                                # 'max_pi': max_pi,
                                # 'mu_error_of_max_pi': mu_error_of_max_pi,
                                # 'sigma_of_max_pi': sigma_of_max_pi,
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


class MADDPGCommFilteringHierMap(MADDPGFiltering):
    def __init__(self, agent_num, 
                    num_in_pol, 
                    num_out_pol, 
                    num_in_critic, 
                    comm_dim,
                    discrete_action, 
                    filtering_model_config, 
                    filtering_model_path=None, 
                    gamma=0.95, 
                    tau=0.01, 
                    critic_lr=0.01, 
                    policy_lr=0.01, 
                    filter_lr=0.0001,
                    hidden_dim=64, 
                    level="high",
                    device="cuda"):

        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, filtering_model_config, filtering_model_path, gamma, tau, critic_lr, policy_lr, filter_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        prior_or_combine = "prior_model"
        prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
        prior_or_combine = "combined_model"
        self.filtering_model = configure_model(config, prior_or_combine, prior_model)
        self.filtering_model.to(device)

        if filtering_model_path is not None:
            self.filtering_model.load_state_dict(torch.load(filtering_model_path))
            print("Loaded filtering model from {}".format(filtering_model_path))

        self.trgt_filtering_model = copy.deepcopy(self.filtering_model)

        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 if level == "high" else 1 * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5

        agent_input_dim = filtering_output_dim + num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = filtering_output_dim * 6 + num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic
        self.comm_dim = comm_dim
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [FilteringCommDDPGAgentHierMap(num_in_pol, num_out_pol, num_in_critic_filter, self.filtering_model, comm_dim=self.comm_dim, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]


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

    def step(self, observations, filtering_output, level="high", explore=False):
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
        for agent_i, (a, obs) in enumerate(zip(self.agents, observations)):
            if level == "high":
                agent_actions, agent_comms = a.step(obs, filtering_output, level=level, explore=explore)
            elif level=="low":
                agent_actions, agent_comms = a.step(obs, filtering_output[agent_i], level=level, explore=explore)
            agents_actions.append(agent_actions)
            agents_comms.append(agent_comms)
            agents_actionsComms.append(torch.cat((agent_actions, agent_comms), dim=-1))
        return agents_actions, agents_comms, filtering_output

    def construct_map_from_obs_acs(self, obs_map, ac):
        # coord_x, coord_y = obs[-2:]
        # map = obs.view(-1, 5, 15, 15)
        obs_map_copy = copy.deepcopy(obs_map)
        mask = torch.nonzero(obs_map_copy[:,1,:,:] == 1)
        batch_idx_with_ag = mask[:,0]
        obs_map_copy[mask[:,0],3,mask[:,1],mask[:,2]] = ac[batch_idx_with_ag,0]
        obs_map_copy[mask[:,0],4,mask[:,1],mask[:,2]] = ac[batch_idx_with_ag,1]
        return obs_map_copy

    def update(self, sample, agent_i, level="high", train_option="regular", train_mode="maddpg", logger=None):
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
            obs, acs, rews, next_obs, dones, filtering_out, next_filtering_out, prisoner_loc = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        # filtering_obs, next_filtering_obs, prisoner_loc = filtering_sample
        

        self.filtering_model.train()

        curr_agent = self.agents[agent_i]
        for i, (ob, nob) in enumerate(zip(obs, next_obs)):
            obs[i] = curr_agent.obs_to_map(ob)
            next_obs[i] = curr_agent.obs_to_map(nob)
        
        # next_filtering_output = self.filtering_model(next_filtering_obs[agent_i])
        # next_prior_output = self.filtering_model.prior_network(self.get_prior_input(next_filtering_obs[agent_i]))
        # next_filtering_output = self.trgt_filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))

        if self.discrete_action: # one-hot encode action
            # all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
            #                 zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs, nfo in zip(self.target_policies, next_obs, next_filtering_out):
                agent_locations = nobs[:, 1:3]
                input_tensor = torch.cat((nobs, nfo), dim=-1)
                all_trgt_acs.append(torch.cat((onehot_from_logits(pi(input_tensor)[0]), pi(input_tensor)[1]), dim=-1))
                all_next_filter_ins.append(nfo)
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            for pi, nobs, nfo in zip(self.target_policies, next_obs, next_filtering_out):
                agent_locations = nobs[:, 1:3]
                # localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                # next_input = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
                input_tensor = nobs
                all_trgt_acs.append(torch.cat(pi(input_tensor), dim=-1))
                all_next_filter_ins.append(nfo)
        if train_mode == "maddpg":
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs, *all_next_filter_ins), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        elif train_mode == "ddpg":
            trgt_vf_in = torch.cat((next_obs[agent_i], all_trgt_acs[agent_i], all_next_filter_ins[agent_i]), dim=1)
        elif train_mode == "map":
            trgt_vf_in = self.construct_map_from_obs_acs(next_obs[agent_i], all_trgt_acs[agent_i])


        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in)[0] *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        # """Update the current filter"""
        # localize each filtering output for the current observations
        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_agent.filter_optimizer.zero_grad()
        # # curr_prior_loss.backward()
        # curr_filtering_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
        # curr_agent.filter_optimizer.step()
        
        all_filter_ins = []
        for fo in filtering_out:
            # agent_locations = ob[:, 1:3]
            # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            # input_vf = self.convert_filtering_output_to_tensor(localized_filtering_out)
            all_filter_ins.append(fo.detach())

        # need to include filtering output to vf input here as well
        if train_mode == "maddpg":
            vf_in = torch.cat((*obs, *acs, *all_filter_ins), dim=1)
        elif train_mode == "ddpg":
            vf_in = torch.cat((obs[agent_i], acs[agent_i], all_filter_ins[agent_i]), dim=1)
        elif train_mode == "map":
            vf_in = self.construct_map_from_obs_acs(obs[agent_i], acs[agent_i])
        actual_value = curr_agent.critic(vf_in)[0] # reward_prediction(from t)

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

        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            
            curr_pol_out = curr_agent.policy(torch.cat((obs[agent_i], filtering_out[agent_i]), dim=-1).detach())
            curr_pol_vf_in = (gumbel_softmax(curr_pol_out[0], hard=True), curr_pol_out[1])
        else:
            agent_locations = obs[agent_i][:, 1:3]
            # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
            # localized_filter_tensor = self.convert_filtering_output_to_tensor(localized_filtering_out)

            input_policy_tensor = obs[agent_i]
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(torch.cat(curr_pol_vf_in, dim=-1))
                all_pol_filter_ins.append(filtering_out[i].detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                input_tensor = torch.cat((ob, filtering_out[i]), dim=-1)
                all_pol_acs.append(torch.cat((onehot_from_logits(pi(input_tensor)[0]), pi(input_tensor)[1]), dim=-1))
                all_pol_filter_ins.append(filtering_out[i].detach()) # add the localized filtering input of current agent
            else:
                agent_locations = ob[:, 1:3]
                # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                # input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                input_tensor_policy = ob
                all_pol_acs.append(torch.cat(pi(input_tensor_policy), dim=-1).detach()) 
                all_pol_filter_ins.append(filtering_out[i].detach())

        if train_mode == "maddpg":
            vf_in = torch.cat((*obs, *all_pol_acs, *all_pol_filter_ins), dim=1)
        elif train_mode == "ddpg":
            vf_in = torch.cat((obs[agent_i], all_pol_acs[agent_i], all_pol_filter_ins[agent_i]), dim=1)
        elif train_mode == "map":
            vf_in = self.construct_map_from_obs_acs(obs[agent_i], all_pol_acs[agent_i])

        pol_loss = -curr_agent.critic(vf_in)[0].mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/%s_losses' % (agent_i, level),
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                # 'curr_filtering_loss': curr_filtering_loss,
                                # 'max_pi': max_pi,
                                # 'mu_error_of_max_pi': mu_error_of_max_pi,
                                # 'sigma_of_max_pi': sigma_of_max_pi,
                                # 'filtering_loss_pol': filtering_loss_pol,
                                # 'filtering_loss_vf': filtering_loss_vf
                                },
                               self.niter)
        return td_error_abs_each

    def update_low(self, sample, agent_i, train_option="regular", train_mode="maddpg", logger=None):
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
            obs, acs, rews, next_obs, dones, filtering_out, next_filtering_out, prisoner_loc = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        # filtering_obs, next_filtering_obs, prisoner_loc = filtering_sample
        

        self.filtering_model.train()

        curr_agent = self.agents[agent_i]

        obs = curr_agent.obs_to_map(obs)
        next_obs = curr_agent.obs_to_map(next_obs)
        
        # next_filtering_output = self.filtering_model(next_filtering_obs[agent_i])
        # next_prior_output = self.filtering_model.prior_network(self.get_prior_input(next_filtering_obs[agent_i]))
        # next_filtering_output = self.trgt_filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))

        all_trgt_acs = torch.cat(curr_agent.policy(next_obs), dim=-1)

        if train_mode == "map":
            trgt_vf_in = self.construct_map_from_obs_acs(next_obs, all_trgt_acs)
        else:
            raise NotImplementedError


        target_value = (rews.view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in)[0] *
                        (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        # """Update the current filter"""
        # localize each filtering output for the current observations
        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # curr_agent.filter_optimizer.zero_grad()
        # # curr_prior_loss.backward()
        # curr_filtering_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
        # curr_agent.filter_optimizer.step()
        

        # need to include filtering output to vf input here as well
        if train_mode == "map":
            vf_in = self.construct_map_from_obs_acs(obs, acs)
        else:
            raise NotImplementedError
        actual_value = curr_agent.critic(vf_in)[0] # reward_prediction(from t)

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

        # filtering_output = self.trgt_filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
        curr_pol_out = curr_agent.policy(obs.detach())
        curr_pol_vf_in = curr_pol_out

        all_pol_acs = torch.cat(curr_pol_vf_in, dim=-1)

        if train_mode == "map":
            vf_in = self.construct_map_from_obs_acs(obs, all_pol_acs)
        else:
            raise NotImplementedError

        pol_loss = -curr_agent.critic(vf_in)[0].mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        # pol_filtering_loss = pol_loss + 0.01 * filtering_loss_pol
        pol_filtering_loss = pol_loss
        """Update the current policy"""
        curr_agent.policy_optimizer.zero_grad()
        pol_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc[agent_i])
        if logger is not None:
            logger.add_scalars('agent%i/%s_losses' % (agent_i, "low"),
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                # 'curr_filtering_loss': curr_filtering_loss,
                                # 'max_pi': max_pi,
                                # 'mu_error_of_max_pi': mu_error_of_max_pi,
                                # 'sigma_of_max_pi': sigma_of_max_pi,
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