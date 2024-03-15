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
from blue_bc.maddpg import BaseMADDPG, BaseDDPG, BaseDDPGAgent, onehot_from_logits, gumbel_softmax
from blue_bc.utils import convert_buffer_to_dataloader
from tqdm import tqdm
from blue_bc.maddpg_filtering import FilteringDDPGAgent, MADDPGFiltering, FilteringCommDDPGAgentHierMap

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')

class MADDPGFilteringShared(BaseMADDPG):
    """ Rather than have different networks for each agent number, maintain a single network 
    for each agent type
    """
    def __init__(self, 
                    type_dict,
                    agent_num, 
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
                    device="cuda",
                    agent_types = 2):

        super().__init__(agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma, tau, critic_lr, policy_lr, hidden_dim, device)
        with open(filtering_model_config, 'r') as stream:
            config = yaml.safe_load(stream)

        # load the prior model here
        prior_or_combine = "prior_model"
        prior_model = configure_model(config, prior_or_combine, prior_network=None).to(device)
        # construct the combined model with the loaded prior_model here
        prior_or_combine = "combined_model"
        self.filtering_model = configure_model(config, prior_or_combine, prior_network=prior_model)
        # convert to cuda or cpu
        self.filtering_model.to(device)

        if filtering_model_path is not None:
            self.filtering_model.load_state_dict(torch.load(filtering_model_path))
            print("Loaded filtering model from {}".format(filtering_model_path))

        # filtering_hidden_dim = config["model"]["hidden_dim"]
        filtering_output_dim = config["combined_model"]["number_gaussians"] * 5 # pi is 1, mu is 2, sigma is 2 (1 + 2 + 2) = 5
        agent_input_dim = num_in_pol # filtering_output_dim + num_in_pol # 2 + num_in_pol

        # 6 is for total number of agents
        num_in_critic_filter = num_in_critic # include filtering input with critic input # 2 * 6 + num_in_critic

        self.agent_types = agent_types
        self.nagents = agent_num
        self.discrete_action = discrete_action
        # self.agents = [FilteringDDPGAgent(agent_input_dim, num_out_pol, num_in_critic_filter, self.filtering_model, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        
        self.agents = {}
        for i in range(self.agent_types):
            self.agents[i] = FilteringDDPGAgent(agent_input_dim, 
                num_out_pol, 
                num_in_critic_filter, 
                self.filtering_model, 
                hidden_dim=hidden_dim, 
                critic_lr=critic_lr, 
                policy_lr=policy_lr, 
                filter_lr=filter_lr, 
                discrete_action=discrete_action, 
                device=device)
        
        
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0
        self.agent_type_dict = type_dict

        self.filtering_loss = -6

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for _, a in self.agents.items():
            a.scale_noise(scale)

    def reset_noise(self):
        for _, a in self.agents.items():
            a.reset_noise()

    def get_filter_loss(self, filtering_input, prisoner_loc):
        curr_filtering_loss = self.filtering_model.compute_loss(filtering_input, torch.Tensor(prisoner_loc).unsqueeze(0))
        filtering_loss = curr_filtering_loss.detach().cpu().numpy()
        return filtering_loss

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """

        
        # filtering_output = self.filtering_model(*self.split_filtering_input(filtering_input))
        # original code

        actions = []
        # for agent_id in id_type_dict:
        for i, obs in enumerate(observations):
            agent_type = self.agent_type_dict[i]
            action = self.agents[agent_type].step(obs, explore=explore)
            actions.append(action)
        return actions

    def split_filtering_input(self, filtering_input):
        prior_input = filtering_input[..., 0:3]
        dynamic_input = filtering_input[..., 3:]
        sel_input = filtering_input
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
                # prior_input = x_train[...,0:3]
                # curr_prior_loss = self.filtering_model.prior_network.compute_loss(prior_input, y_train)
                curr_filtering_loss = self.filtering_model.compute_loss(x_train, y_train)
                curr_agent.filter_optimizer.zero_grad()
                # curr_prior_loss.backward()
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
        agent_type = self.agent_type_dict[agent_i]
        curr_agent = self.agents[agent_type]
        
        # next_filtering_output = self.filtering_model(next_filtering_obs[agent_i])
        # next_prior_output = self.filtering_model.prior_network(self.get_prior_input(next_filtering_obs[agent_i]))
        next_filtering_output = self.filtering_model(*self.split_filtering_input(next_filtering_obs[agent_i]))

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = []
            all_next_filter_ins = []
            # for pi, nobs in zip(self.target_policies, next_obs):
            for j, nobs in enumerate(next_obs):
                # j represents the agent id and we need the actual agent target policy
                pi = self.agents[self.agent_type_dict[j]].target_policy
                # agent_locations = nobs[:, 1:3]
                # localized_next_filtering_out = self.localize_filtering_mu(next_filtering_output, agent_locations)
                # next_input = self.convert_filtering_output_to_tensor(localized_next_filtering_out)
                input_tensor = nobs
                all_trgt_acs.append(pi(input_tensor))
                # all_next_filter_ins.append(next_input)

        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs, *all_next_filter_ins), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)


        # localize each filtering output for the current observations
        filtering_output = self.filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        # max_pi, mu_error_of_max_pi, sigma_of_max_pi = self.collect_distribution_error(filtering_output, prisoner_loc)
        # curr_filtering_loss = self.filtering_model.compute_loss(filtering_obs[agent_i], prisoner_loc[agent_i])
        curr_prior_loss = self.filtering_model.prior_network.compute_loss(self.get_prior_input(next_filtering_obs[agent_i]), prisoner_loc[agent_i])
        curr_filtering_loss = self.filtering_model.compute_loss(*self.split_filtering_input(filtering_obs[agent_i]), prisoner_loc[agent_i])
        # self.filtering_loss = curr_filtering_loss.detach()
        # target_value = target_value + ((-curr_filtering_loss) - 6)

        """Update the current filter"""
        # # if agent_i == 0:
        # curr_agent.filter_optimizer.zero_grad()
        # # curr_prior_loss.backward()
        # curr_filtering_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.filtering_model.parameters(), 0.1)
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

        """Update the current critic"""
        curr_agent.critic_optimizer.zero_grad()
        vf_filtering_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        #### ----------------------------Update Policy Below ----------------------------- ###

        # filtering_output = self.filtering_model(*self.split_filtering_input(filtering_obs[agent_i]))
        
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

            input_policy_tensor = obs[agent_i]
            curr_pol_out = curr_agent.policy(input_policy_tensor.detach())
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        all_pol_filter_ins = []
        # for i, pi, ob in zip(range(self.nagents), self.policies, obs):
        for i in range(self.nagents):
            pi = self.agents[self.agent_type_dict[i]].policy
            ob = obs[i]
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
                # all_pol_filter_ins.append(localized_filter_tensor.detach()) # add the localized filtering input of current agent
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                # agent_locations = ob[:, 1:3]
                # localized_filtering_out = self.localize_filtering_mu(filtering_output, agent_locations)
                # input_filter = self.convert_filtering_output_to_tensor(localized_filtering_out)
                input_tensor_policy = ob
                all_pol_acs.append(pi(input_tensor_policy).detach())
                # all_pol_filter_ins.append(input_filter.detach())

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
        for _, a in self.agents.items():
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
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

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for _, a in self.agents.items()], 
                     'filtering_model': self.filtering_model.state_dict()}
        torch.save(save_dict, filename)

    def prep_training(self, device='gpu'):
        for _, a in self.agents.items():
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        
        if not self.pol_dev == device:
            for _, a in self.agents.items():
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for _, a in self.agents.items():
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for _, a in self.agents.items():
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for _, a in self.agents.items():
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for _, a in self.agents.items():
            a.policy.eval()
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for _, a in self.agents.items():
            a.policy = fn(a.policy)
        self.pol_dev = device
        
    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        agents = [agent for _, agent in self.agents.items()]
        for a, params in zip(agents, save_dict['agent_params']):
            a.load_params(params)
            # a.policy.to(self.pol_dev)
            # a.critic.to(self.critic_dev)
            # a.target_policy.to(self.trgt_critic_dev)
            # a.target_critic.to(self.trgt_critic_dev)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])
        # self.filtering_model.to(self.pol_dev)

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


class MADDPGCommFilteringHierMapShared(MADDPGFiltering):
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
        # self.agents = [FilteringCommDDPGAgentHierMap(num_in_pol, num_out_pol, num_in_critic_filter, self.filtering_model, comm_dim=self.comm_dim, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        self.agent_types = 2
        self.agents = {}
        for i in range(self.agent_types):
            self.agents[i] = FilteringCommDDPGAgentHierMap(num_in_pol, num_out_pol, num_in_critic_filter, self.filtering_model, 
                                                           comm_dim=self.comm_dim, hidden_dim=hidden_dim, 
                                                           critic_lr=critic_lr, policy_lr=policy_lr, filter_lr=filter_lr, 
                                                           discrete_action=discrete_action, device=device)
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
        self.agent_type_dict = [0, 0, 0, 0, 0, 1]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for _, a in self.agents.items():
            a.scale_noise(scale)

    def reset_noise(self):
        for _, a in self.agents.items():
            a.reset_noise()

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
        for agent_i, obs in enumerate(observations):
            agent_type = self.agent_type_dict[agent_i]
            if level == "high":
                agent_actions, agent_comms = self.agents[agent_type].step(obs, filtering_output, level=level, explore=explore)
            elif level=="low":
                agent_actions, agent_comms = self.agents[agent_type].step(obs, filtering_output[agent_i], level=level, explore=explore)
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

        agent_type = self.agent_type_dict[agent_i]
        curr_agent = self.agents[agent_type]

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
            for i, nobs, nfo in (zip(range(self.nagents), next_obs, next_filtering_out)):
                pi = self.agents[self.agent_type_dict[i]].target_policy
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
        for i, ob in zip(range(self.nagents), obs):
            pi = self.agents[self.agent_type_dict[i]].policy
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
        agent_type = self.agent_type_dict[agent_i]
        curr_agent = self.agents[agent_type]

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

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for _, a in self.agents.items():
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for _, a in self.agents.items()], 
                     'filtering_model': self.filtering_model.state_dict()}
        torch.save(save_dict, filename)

    def prep_training(self, device='gpu'):
        for _, a in self.agents.items():
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        
        if not self.pol_dev == device:
            for _, a in self.agents.items():
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for _, a in self.agents.items():
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for _, a in self.agents.items():
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for _, a in self.agents.items():
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for _, a in self.agents.items():
            a.policy.eval()
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for _, a in self.agents.items():
            a.policy = fn(a.policy)
        self.pol_dev = device
        
    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=self.pol_dev)
        agents = [agent for _, agent in self.agents.items()]
        for a, params in zip(agents, save_dict['agent_params']):
            a.load_params(params)
            # a.policy.to(self.pol_dev)
            # a.critic.to(self.critic_dev)
            # a.target_policy.to(self.trgt_critic_dev)
            # a.target_critic.to(self.trgt_critic_dev)
        self.filtering_model.load_state_dict(save_dict['filtering_model'])
        # self.filtering_model.to(self.pol_dev)

