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
from policy import LowLevelPolicy, MLPNetwork, hierMLPNetwork, HighLevelPolicy, HighLevelPolicyIndependent, ObstacleQuasiGNNNetwork
from blue_bc.utils import temperature_softmax, identity_map
# from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, two_hot_encode
import yaml
from models.configure_model import configure_model

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')
class BaseDDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma=0.95, tau=0.01, critic_lr=0.01, policy_lr=0.01, hidden_dim=64, device="cuda"):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [BaseDDPGAgent(num_in_pol, num_out_pol, num_in_critic, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, discrete_action=discrete_action, device=device) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]


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
            obs, acs, rews, next_obs, dones = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        # obs_noised = [(obs[i] + 0.1*(torch.rand_like(obs[i])-0.5)).detach() for i in range(len(obs))]
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                            next_obs)]

        trgt_vf_in = torch.cat((next_obs[agent_i], curr_agent.target_policy(next_obs[agent_i])), dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        # vf_in_noised = torch.cat((*obs_noised, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)
        # actual_value_noised = curr_agent.critic(vf_in_noised) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach()) 
        # + 0.1 * MSELoss_each(actual_value, actual_value_noised)
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            # curr_pol_out_noised = curr_agent.policy(obs_noised[agent_i])
            curr_pol_vf_in = curr_pol_out
            # curr_pol_vf_noised_in = curr_pol_out_noised

        all_pol_acs = []
        all_pol_acs_noised = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
                # all_pol_acs_noised.append(curr_pol_vf_noised_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
                # all_pol_acs_noised.append(onehot_from_logits(pi(ob_n)))
            else:
                all_pol_acs.append(pi(ob))
                # all_pol_acs_noised.append(pi(ob_n))

        vf_in = torch.cat((obs[agent_i], curr_pol_vf_in), dim=1)
        all_pol_acs_torch = torch.cat(all_pol_acs, dim=1)
        # all_pol_acs_noised_torch = torch.cat(all_pol_acs_noised, dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        # + 0.2 * (curr_pol_vf_in - curr_pol_vf_noised_in).abs().mean()
        # print("noise = ", (obs_noised[0]-obs[0]).abs().mean())
        # print("policy loss = ", -curr_agent.critic(vf_in).mean())
        # print("smooth loss = ", (all_pol_acs_torch - all_pol_acs_noised_torch).abs().mean())
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)
        return td_error_abs_each

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

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
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)

class BaseHierCtrl(object):
    def __init__(self, agent_num, num_in_pol, high_num_out_pol, low_num_out_pol, num_in_critic, gamma=0.95, tau=0.01, critic_lr=0.01, policy_lr=0.01, 
                    hidden_dim=64, device="cuda", subpolicy_num = 4):
        self.subpolicy_num = subpolicy_num
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.nagents = agent_num
        self.niter = 0
        self.high_policy = MLPNetwork(num_in_pol, high_num_out_pol, hidden_dim=hidden_dim, out_fn=temperature_softmax).to(device)
        self.high_trgt_policy = MLPNetwork(num_in_pol, high_num_out_pol, hidden_dim=hidden_dim, out_fn=temperature_softmax).to(device)
        self.critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, out_fn=identity_map).to(device)
        self.trgt_critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, out_fn=identity_map).to(device)
        self.low_policies = []
        self.low_trgt_policies = []
        for _ in range(subpolicy_num):
            self.low_policies.append(MLPNetwork(num_in_pol, low_num_out_pol, hidden_dim=hidden_dim, out_fn=F.tanh).to(device))
            self.low_trgt_policies.append(MLPNetwork(num_in_pol, low_num_out_pol, hidden_dim=hidden_dim, out_fn=F.tanh).to(device))
        # INFO: Make sure target policies/critics the same with policies/critics at the begining
        hard_update(self.high_trgt_policy, self.high_policy)
        hard_update(self.trgt_critic, self.critic)
        hard_update(self.low_trgt_policies, self.low_policies)
        # INFO: Define some optimizers
        self.high_policy_optimizer = Adam(self.high_policy.parameters(), lr=policy_lr)
        self.low_policy_optimizers = [Adam(pi.parameters(), lr=policy_lr) for pi in self.low_policies]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # INFO: Define the exploration noise
        self.exploration = OUNoise(2, scale=0.1, mu=0, theta=0.15, sigma=15)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, step_option="high", explore = False):
        if step_option == "high":
            high_action = self.high_policy(obs[0])
            subpolicy_coeffs = high_action.unsqueeze(0).transpose(0,1)
            candidate_actions = []
            for subpolicy_idx in range(self.subpolicy_num):
                low_action = self.low_policies[subpolicy_idx](obs[0])
                candidate_actions.append(low_action)
            agent_actions = torch.sum((subpolicy_coeffs * torch.vstack(candidate_actions)), dim=0)
        elif "goal" in step_option or step_option == "collision":
            # INFO: Decide which goal it is going to
            if "goal" in step_option:
                splited_list = step_option.split('_')
                sub_idx = int(splited_list[-1])
            else:
                sub_idx = 3
            agent_actions = self.low_policies[sub_idx](obs[0])
        else:
            raise NotImplementedError
        if explore:
            agent_actions += Variable(Tensor(self.exploration.noise()).to(self.device), requires_grad=False)
            agent_actions = agent_actions.clamp(-1, 1)
        return agent_actions

    def update(self, sample, agent_i, train_option="high", logger=None):
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

        obs, acs, rews, next_obs, dones = sample

        if train_option == "high":
            # INFO: Update the critic
            self.critic_optimizer.zero_grad()
            high_trgt_actions = self.high_trgt_policy(next_obs[0])
            # Calculate the low level candidate actions
            candidate_actions = []
            for subpolicy_idx in range(self.subpolicy_num):
                low_action = self.low_trgt_policies[subpolicy_idx](next_obs)
                candidate_actions.append(low_action[0])
            all_trgt_acs = (high_trgt_actions * torch.vstack(candidate_actions)).sum(axis=0)
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma * self.trgt_critic(trgt_vf_in) * (1 - dones[agent_i].view(-1, 1)))
            vf_in = torch.cat((*obs, *acs), dim=1)
            actual_value = self.critic(vf_in)
            vf_loss_each = MSELoss_each(actual_value, target_value.detach())
            td_error_each = target_value - actual_value
            td_error_abs_each = torch.abs(td_error_each)
            vf_loss = torch.mean(vf_loss_each.squeeze())
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            # INFO: Update the policy
            self.high_policy_optimizer.zero_grad()
            curr_high_pol_out = self.high_policy(obs[0])
            candidate_actions = []
            for subpolicy_idx in range(self.subpolicy_num):
                low_action = self.low_policies[subpolicy_idx](obs)
                candidate_actions.append(low_action[0])
            curr_pol_out = (curr_high_pol_out * torch.vstack(candidate_actions)).sum(axis=0)
            curr_pol_vf_in = curr_pol_out
            vf_in = torch.cat((*obs, *curr_pol_vf_in), dim=1)
            entropy = -torch.sum(F.softmax(curr_pol_out, dim=-1) * torch.log(F.softmax(curr_pol_out, dim=-1)), axis=-1).mean()
            pol_loss = -self.critic(vf_in).mean() + (-0.00*entropy)
            pol_loss.backward()
            torch.nn.utils.clip_grad_norm(self.high_policy.parameters(), 0.5)
            self.high_policy_optimizer.step()
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                {'vf_loss': vf_loss,
                                    'pol_loss': pol_loss,
                                    'entropy': entropy,
                                    # 'embedding_loss': embedding_loss
                                    },
                                self.niter)
            return td_error_abs_each
        elif "goal" in train_option or train_option == "collision":
            # INFO: Decide which goal it is going to
            if "goal" in train_option:
                splited_list = train_option.split('_')
                sub_idx = int(splited_list[-1])
            else:
                sub_idx = 3
            # INFO: Update the critic
            self.critic_optimizer.zero_grad()
            all_trgt_acs = self.low_trgt_policies[sub_idx](next_obs[agent_i])
            trgt_vf_in = torch.cat((*next_obs, all_trgt_acs), dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma * self.trgt_critic(trgt_vf_in) * (1 - dones[agent_i].view(-1, 1)))
            vf_in = torch.cat((*obs, *acs), dim=1)
            actual_value = self.critic(vf_in)
            vf_loss_each = MSELoss_each(actual_value, target_value.detach())
            td_error_each = target_value - actual_value
            td_error_abs_each = torch.abs(td_error_each)
            vf_loss = torch.mean(vf_loss_each.squeeze())
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            # INFO: Update the policy
            self.low_policy_optimizers[sub_idx].zero_grad()
            curr_low_pol_out = self.low_policies[sub_idx](obs[agent_i])
            curr_pol_vf_in = curr_low_pol_out
            vf_in = torch.cat((*obs, curr_pol_vf_in), dim=1)
            entropy = -torch.sum(F.softmax(curr_low_pol_out, dim=-1) * torch.log(F.softmax(curr_low_pol_out, dim=-1)), axis=-1).mean()
            pol_loss = -self.critic(vf_in).mean()
            pol_loss.backward()
            torch.nn.utils.clip_grad_norm(self.low_policies[sub_idx].parameters(), 0.5)
            self.low_policy_optimizers[sub_idx].step()
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                {'vf_loss': vf_loss,
                                    'pol_loss': pol_loss,
                                    'entropy': entropy,
                                    # 'embedding_loss': embedding_loss
                                    },
                                self.niter)
            return td_error_abs_each
        else:
            raise NotImplementedError

    def init_from_save(self, filename, load_option="high"):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        if load_option == "high":
            self.high_policy = torch.load(filename)
        elif "goal" in load_option or load_option == "collision":
            # INFO: Decide which goal it is going to
            if "goal" in load_option:
                splited_list = load_option.split('_')
                sub_idx = int(splited_list[-1])
            else:
                sub_idx = 3
            # self.prep_training(device='cpu')  # move parameters to CPU before saving
            self.low_policies[sub_idx] = torch.load(filename)


        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params) 

    def save(self, filename, save_option="high"):
        if save_option == "high":
            # self.prep_training(device='cpu')  # move parameters to CPU before saving
            torch.save(self.high_policy.train(), filename)
        elif "goal" in save_option or save_option == "collision":
            # INFO: Decide which goal it is going to
            if "goal" in save_option:
                splited_list = save_option.split('_')
                sub_idx = int(splited_list[-1])
            else:
                sub_idx = 3
            # self.prep_training(device='cpu')  # move parameters to CPU before saving
            torch.save(self.low_policies[sub_idx].train(), filename)

    def prep_training(self, device='gpu'):
        self.high_policy.train()
        self.high_trgt_policy.train()
        for low_policy, trgt_low_policy in zip(self.low_policies, self.low_trgt_policies):
            low_policy.train()
            trgt_low_policy.train()
        self.critic.train()
        self.trgt_critic.train()
        return
            

    def prep_rollouts(self, device='cpu'):
        self.high_policy.eval()
        # self.high_trgt_policy.eval()
        for low_policy, trgt_low_policy in zip(self.low_policies, self.low_trgt_policies):
            low_policy.eval()
            # trgt_low_policy.eval()
        # self.critic.eval()
        # self.trgt_critic.eval()
        return

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.trgt_critic, self.critic, self.tau)
        soft_update(self.high_trgt_policy, self.high_policy, self.tau)
        soft_update(self.low_trgt_policies, self.low_policies, self.tau)
        self.niter += 1

class BaseMADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma=0.95, tau=0.01, critic_lr=0.01, policy_lr=0.01, hidden_dim=64, device="cuda", constrained=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [BaseDDPGAgent(num_in_pol, num_out_pol, num_in_critic, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, discrete_action=discrete_action, device=device, constrained=constrained) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]

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
            obs, acs, rews, next_obs, dones = sample
        elif train_option == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
        else:
            raise NotImplementedError

        obs_noised = [(obs[i] + 0.1*(torch.rand_like(obs[i])-0.5)).detach() for i in range(len(obs))]
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                            next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_in = torch.cat((*obs, *acs), dim=1)
        vf_in_noised = torch.cat((*obs_noised, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)
        actual_value_noised = curr_agent.critic(vf_in_noised) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach()) 
        # + 0.1 * MSELoss_each(actual_value, actual_value_noised)
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            curr_pol_vf_noised_in = curr_pol_vf_in
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_out_noised = curr_agent.policy(obs_noised[agent_i])
            curr_pol_vf_in = curr_pol_out
            curr_pol_vf_noised_in = curr_pol_out_noised

        all_pol_acs = []
        all_pol_acs_noised = []
        for i, pi, ob, ob_n in zip(range(self.nagents), self.policies, obs, obs_noised):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
                all_pol_acs_noised.append(curr_pol_vf_noised_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
                all_pol_acs_noised.append(onehot_from_logits(pi(ob_n)))
            else:
                all_pol_acs.append(pi(ob))
                all_pol_acs_noised.append(pi(ob_n))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        all_pol_acs_torch = torch.cat(all_pol_acs, dim=1)
        all_pol_acs_noised_torch = torch.cat(all_pol_acs_noised, dim=1)

        # embedding_loss = 100 * curr_agent.policy.get_embedding_loss(*obs, *next_obs, loss_func=MSELoss)
        entropy = -torch.sum(F.softmax(curr_pol_out, dim=-1) * torch.log(F.softmax(curr_pol_out, dim=-1)), axis=-1).mean()
        pol_loss = -curr_agent.critic(vf_in).mean() # + (-0.02*entropy)
        # + 0.2 * (curr_pol_vf_in - curr_pol_vf_noised_in).abs().mean()
        # print("noise = ", (obs_noised[0]-obs[0]).abs().mean())
        # print("policy loss = ", -curr_agent.critic(vf_in).mean())
        # print("smooth loss = ", (all_pol_acs_torch - all_pol_acs_noised_torch).abs().mean())
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'entropy': entropy,
                                # 'embedding_loss': embedding_loss
                                },
                               self.niter)
        return td_error_abs_each

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

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
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)

class BaseDDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64, critic_lr=0.01, policy_lr=0.01, discrete_action=True, device="cpu", constrained=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=constrained).to(device)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False).to(device)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=constrained).to(device)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False).to(device)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        if not discrete_action:
            self.exploration = OUNoise(2, scale=0.1, mu=0, theta=0.15, sigma=15)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.device = device

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
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

class hierMADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_num, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma=0.95, tau=0.01, critic_lr=0.01, policy_lr=0.01, hidden_dim=64, device="cuda", constrained=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.num_out_pol = num_out_pol
        self.nagents = agent_num
        self.discrete_action = discrete_action
        self.agents = [hierDDPGAgent(num_in_pol, num_out_pol, num_in_critic, hidden_dim=hidden_dim, critic_lr=critic_lr, policy_lr=policy_lr, discrete_action=discrete_action, device=device, constrained=constrained) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, self.niter, explore=explore) for a, obs in zip(self.agents, observations)]

    def get_agent_action(self, obs, low_maddpgs, high_policy, train_low = False):
        batch_size = obs[0].shape[0]
        candidate_actions = []
        for sub_i, subpolicy in enumerate(low_maddpgs):
            if sub_i < 3:
                torch_agent_actions = subpolicy.agents[0].policy(obs[0][...,:16])
            elif sub_i < 4:
                torch_agent_actions = subpolicy.agents[0].policy(obs[0])
            else:
                raise NotImplementedError
            # step([obs[0][...,:16]], explore=False)
            candidate_actions.append(torch_agent_actions)
        torch_candidate_actions = torch.cat(candidate_actions, dim=-1)
        if train_low == False:
            torch_candidate_actions = torch_candidate_actions.detach()
        # INFO: Augment the high observation with low proposed actions
        high_observation = torch.cat((obs[0], torch_candidate_actions), dim=-1)
        subpolicy_coeffs = high_policy(high_observation).view(batch_size,-1, 1) # temperature_softmax(high_policy(high_observation), temperature=1.0).view(batch_size,-1, 1) 
        action = torch.sum((subpolicy_coeffs * torch_candidate_actions.view(batch_size,-1,2)), dim=1)
        return high_observation.detach(), action.clamp(-1, 1)


    def update(self, sample, agent_i, low_maddpgs, train_option="regular", logger=None, train_low = False):
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
        curr_agent.critic_optimizer.zero_grad()

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction
        else: 
            augmented_nextObs, all_trgt_acs = self.get_agent_action(next_obs, low_maddpgs, curr_agent.target_policy, train_low)
            augmented_obs, all_acs = self.get_agent_action(obs, low_maddpgs, curr_agent.policy, train_low)
        trgt_vf_in = torch.cat((augmented_nextObs, all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_in = torch.cat((augmented_obs, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach()) 
        # + 0.1 * MSELoss_each(actual_value, actual_value_noised)
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze())
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        if train_low:
            for subpolicy in low_maddpgs:
                subpolicy.agents[0].policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            curr_pol_vf_noised_in = curr_pol_vf_in
        else:
            curr_pol_vf_in = all_acs

        all_pol_acs = []
        all_pol_acs_noised = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
        vf_in = torch.cat((augmented_obs, *all_pol_acs), dim=1)
        all_pol_acs_torch = torch.cat(all_pol_acs, dim=1)

        # embedding_loss = 100 * curr_agent.policy.get_embedding_loss(*obs, *next_obs, loss_func=MSELoss)
        # entropy = -torch.sum(F.softmax(curr_pol_out, dim=-1) * torch.log(F.softmax(curr_pol_out, dim=-1)), axis=-1).mean()
        pol_loss = -curr_agent.critic(vf_in).mean() # + (-0.02*entropy)
        # + 0.2 * (curr_pol_vf_in - curr_pol_vf_noised_in).abs().mean()
        # print("noise = ", (obs_noised[0]-obs[0]).abs().mean())
        # print("policy loss = ", -curr_agent.critic(vf_in).mean())
        # print("smooth loss = ", (all_pol_acs_torch - all_pol_acs_noised_torch).abs().mean())
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if train_low:
            for subpolicy in low_maddpgs:
                torch.nn.utils.clip_grad_norm(subpolicy.agents[0].policy.parameters(), 0.5)
                subpolicy.agents[0].policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                # 'entropy': entropy,
                                # 'embedding_loss': embedding_loss
                                },
                               self.niter)
        return td_error_abs_each

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

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
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    def init_from_save(self, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_num, num_in_high_pol, num_in_low_pol, subpolicy_num_out_pol, para_num_out_pol, num_in_high_critic, num_in_low_critic, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=(64,64), device="cuda"):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = agent_num
        self.device = device
        self.agents = [DDPGAgent(num_in_high_pol, num_in_low_pol, subpolicy_num_out_pol, para_num_out_pol, num_in_high_critic, num_in_low_critic, gamma, tau, lr, hidden_dim, device=device) for _ in range(self.nagents)]
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.high_niter = 0
        self.low_niter = 0
        

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step_high(self, observations, explore=False):
        subpolicies_list = []
        for i, a in enumerate(self.agents):
            subpolicy = a.step_high(observations[i], target=False, explore=explore)
            subpolicies_list.append(subpolicy)
        return torch.cat(subpolicies_list)

    def step_low(self, observations, explore=False):
        paras_list = []
        for i, a in enumerate(self.agents):
            paras = a.step_low(observations[i], target=False, explore=explore)
            paras_list.append(paras)
        return torch.cat(paras_list)

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        subpolicies_list = []
        paras_list = []
        for i, a in enumerate(self.agents):
            subpolicy, subpolicy_idx, paras = a.step(observations, explore=explore)
            subpolicies_list.append(subpolicy)
            if i == 5:
                paras = torch.Tensor([[0.1, *observations[-4:-2]]])
            else:
                paras = torch.Tensor([[1.0, *observations[-4:-2]]])
            paras_list.append(paras)
        return torch.cat(subpolicies_list), torch.cat(paras_list)

    def update_high(self, sample, agent_i, logger=None):
        """"Split samples agent to agent"""
        obs, hier_high_act, rewards, dones, next_obs = sample
        obs = self.split_sample(obs)
        hier_high_act = self.split_sample(hier_high_act)
        next_obs = self.split_sample(next_obs)
        """Get current agent"""
        curr_agent = self.agents[agent_i]
        # batch_size = obs.shape[0]
        """Prepare current agent high cirtic training (zero the grad)"""
        curr_agent.high_critic_optimizer.zero_grad()
        """Calculate the target value (from target critic)"""
        all_trgt_acs = []
        for i in range(self.nagents):
            subpolicy = self.agents[i].step_high(next_obs[i], target=True, explore=False)
            all_trgt_acs.append(subpolicy)             
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        target_value = (rewards[:,agent_i].view(-1, 1) + self.gamma * curr_agent.target_high_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)
        """Calculate the actual value (from actual critic)"""
        vf_in = torch.cat((*obs, *hier_high_act), dim=1)
        actual_value = curr_agent.high_critic(vf_in) # reward_prediction(from t)
        """calculate the critic TD loss"""
        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        vf_loss = torch.mean(vf_loss_each.squeeze().to(device=self.device))
        vf_loss.backward()
        """step the high level critic"""
        torch.nn.utils.clip_grad_norm(curr_agent.high_critic.parameters(), 0.5)
        curr_agent.high_critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()
        """Prepare policy actions (policy(obs)) for all agents, but only train the current policy"""
        curr_agent.high_level_policy.zero_grad()
        all_pol_acs = []
        updating_ag_subpolicy = curr_agent.step_high(obs[agent_i], target=False, explore=False)
        for i in range(self.nagents):
            if i == agent_i:
                all_pol_acs.append(updating_ag_subpolicy)
            else:
                subpolicy = self.agents[i].step_high(obs[i], target=False, explore=False)
                all_pol_acs.append(subpolicy) 
        # all_pol_acs = [pi(obs) for pi in self.policies]                    
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.high_critic(vf_in).mean()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.high_level_policy.parameters(), 0.5)
        curr_agent.high_policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/high/losses' % agent_i,
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.high_niter)

        return td_error_each

    def split_sample(self, sample_item):
        chunk_size = sample_item.shape[-1] // self.nagents
        splited_sample_items = torch.split(sample_item, chunk_size, dim=-1)
        return splited_sample_items

    def update_low(self, sample, agent_i, logger=None):
        obs, hier_low_act, rewards, dones, next_obs = sample
        obs = self.split_sample(obs)
        hier_low_act = self.split_sample(hier_low_act)
        next_obs = self.split_sample(next_obs)

        curr_agent = self.agents[agent_i]
        # batch_size = obs.shape[0]
        curr_agent.low_critic_optimizer.zero_grad()

        all_trgt_acs = []
        for i in range(self.nagents):
            paras = self.agents[i].step_low(next_obs[i], target=True, explore=False)
            all_trgt_acs.append(paras)             
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        target_value = (rewards[:,agent_i].view(-1, 1) + self.gamma * curr_agent.target_low_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_in = torch.cat((*obs, *hier_low_act), dim=1)
        actual_value = curr_agent.low_critic(vf_in) # reward_prediction(from t)
        """calculate the critic TD loss"""
        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        vf_loss = torch.mean(vf_loss_each.squeeze().to(device=self.device))
        vf_loss.backward()
        """step the low level critic"""
        torch.nn.utils.clip_grad_norm(curr_agent.low_critic.parameters(), 0.5)
        curr_agent.low_critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        all_pol_acs = []
        updating_ag_paras = curr_agent.step_low(obs[agent_i], target=False, explore=False)
        curr_agent.low_level_policy.zero_grad()
        for i in range(self.nagents):
            if i == agent_i:
                all_pol_acs.append(updating_ag_paras)
            else:
                paras = self.agents[i].step_low(obs[i], target=False, explore=False)
                all_pol_acs.append(paras) 
        # all_pol_acs = [pi(obs) for pi in self.policies]                    
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.low_critic(vf_in).mean()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.low_level_policy.parameters(), 0.5)
        curr_agent.low_policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/low/losses' % agent_i,
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.low_niter)

        return td_error_each


    def update(self, sample, agent_i, logger=None):
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
        curr_agent = self.agents[agent_i]
        batch_size = obs.shape[0]
        curr_agent.critic_optimizer.zero_grad()

        all_trgt_acs = []
        for i in range(self.nagents):
            # subpolicy = self.agents[i].step_high(obs, target=True, explore=False)
            subpolicy, _, paras = self.agents[i].step(next_obs, target=True, explore=False)
            all_trgt_acs.append(subpolicy)
            all_trgt_acs.append(paras.detach()) 
            # all_trgt_acs.append(pi(next_obs)[0])
            # all_trgt_acs.append(pi(next_obs)[1])              
        trgt_vf_in = torch.cat((next_obs, *all_trgt_acs), dim=1) # (pi, next_obs, target_critic)->reward_prediction

        target_value = (rewards[:,agent_i].view(-1, 1) + self.gamma * curr_agent.target_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        vf_in = torch.cat((obs, hier_act), dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze().to(device=self.device))
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        
        all_pol_acs = []
        subpolicy, subpolicy_idx, paras = curr_agent.step(obs, target=False, explore=False)
        for i in range(5):
            curr_agent.low_policy_optimizers[i].zero_grad()
        curr_agent.high_level_policy.zero_grad()

        for i in range(self.nagents):
            if i == agent_i:
                all_pol_acs.append(subpolicy)
                all_pol_acs.append(paras.detach())
            else:
                subpolicy, _, paras = self.agents[i].step(obs, target=False, explore=False)
                all_pol_acs.append(subpolicy)
                all_pol_acs.append(paras.detach())  
        # all_pol_acs = [pi(obs) for pi in self.policies]                    
        vf_in = torch.cat((obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss.backward()
        for i in range(5):
            torch.nn.utils.clip_grad_norm(curr_agent.low_level_policies[i].parameters(), 0.5)
            curr_agent.low_policy_optimizers[i].step()
        torch.nn.utils.clip_grad_norm(curr_agent.high_level_policy.parameters(), 0.5)
        curr_agent.high_policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

        return td_error_abs_each

    # def update_candidate(self, sample, agent_i, rl_ratio, mode="homo", parallel=False, logger=None, is_debug=True):
    #     """
    #     Update parameters of agent model based on sample from replay buffer
    #     Inputs:
    #         sample: tuple of (observations, actions, rewards, next
    #                 observations, and episode end masks) sampled randomly from
    #                 the replay buffer. Each is a list with entries
    #                 corresponding to each agent
    #         agent_i (int): index of agent to update
    #         parallel (bool): If true, will average gradients across threads
    #         logger (SummaryWriter from Tensorboard-Pytorch):
    #             If passed in, important quantities will be logged
    #     """
    #     curr_agent = self.agents[agent_i]

    #     obs, hier_act, rewards, dones, next_obs = sample
    #     batch_size = obs.shape[0]
    #     vf_in = torch.cat((obs, hier_act.detach()), dim=1)
    #     actual_value = self.blue_critic(vf_in) # reward_prediction(from t)

    #     trgt_acs = self.target_blue_policy(next_obs)
    #     # trgt_acs = torch.cat((trgt_acs[0].view(batch_size, -1), trgt_acs[1].view(batch_size, -1)), dim=1)
    #     trgt_vf_in = torch.cat((next_obs, trgt_acs), dim=1)
    #     target_value = (rewards.view(-1, 1) + self.gamma * self.target_blue_critic(trgt_vf_in) * (1 - dones.view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

    #     vf_loss_each = MSELoss_each(actual_value, target_value.detach())
    #     td_error_each = target_value - actual_value
    #     td_error_abs_each = torch.abs(td_error_each)

    #     # vf_loss = MSELoss(actual_value, target_value.detach())
    #     vf_loss = torch.mean(vf_loss_each.squeeze())
    #     self.critic_optimizer.zero_grad()
    #     vf_loss.backward()
    #     torch.nn.utils.clip_grad_norm(self.blue_critic.parameters(), 0.5)
    #     self.critic_optimizer.step()
    #     # curr_agent.scheduler_critic_optimizer.step()

    #     curr_pol_out = self.blue_policy(obs)
    #     # curr_pol_out = torch.cat((curr_pol_out[0].view(batch_size, -1), curr_pol_out[1].view(batch_size, -1)), dim=1)
    #     vf_in = torch.cat((obs, curr_pol_out), dim=1)
    #     pol_loss = -self.blue_critic(vf_in).mean()
    #     # pol_loss += (curr_pol_out**2).mean() * 1e-3
    #     self.policy_optimizer.zero_grad()
    #     pol_loss.backward()
    #     torch.nn.utils.clip_grad_norm(self.blue_policy.parameters(), 0.5)
    #     self.policy_optimizer.step()

    #     # if self.niter % self.update_target_period == 0:
    #     self.update_all_targets()

    #     if logger is not None:
    #         logger.add_scalars('blue/losses',
    #                            {'vf_loss': torch.mean(vf_loss_each),
    #                             'td_error': torch.mean(td_error_each),
    #                             'pol_loss': pol_loss},
    #                            self.niter)

    def update_high_targets(self):
        for a in self.agents:
            self.soft_update(a.target_high_critic, a.high_critic, self.tau)
            self.soft_update(a.target_high_policy, a.high_level_policy, self.tau)
        self.high_niter += 1

    def update_low_targets(self):
        for a in self.agents:
            self.soft_update(a.target_low_critic, a.low_critic, self.tau)
            self.soft_update(a.target_low_policy, a.low_level_policy, self.tau)
        self.low_niter += 1

    # def update_all_targets(self):
    #     """
    #     Update all target networks (called after normal updates have been
    #     performed for each agent)
    #     """
    #     for a in self.agents:
    #         self.soft_update(a.target_critic, a.critic, self.tau)
    #         self.soft_update(a.target_high_policy, a.high_level_policy, self.tau)
    #         for i in range(5):
    #             self.soft_update(a.target_low_policies[i], a.low_level_policies[i], self.tau)
    #     self.niter += 1

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

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.policy.eval()
            a.critic.eval()
            a.target_policy.eval()
            a.target_critic.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
                a.target_policy = fn(a.target_policy)
                a.critic = fn(a.critic)
                a.target_critic = fn(a.target_critic)
            self.pol_dev = device
            self.trgt_pol_dev = device
            self.critic_dev = device
            self.trgt_critic_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def init_agent_from_save(self, agent_ind, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        # instance = cls(**save_dict['init_dict'])
        # instance.init_dict = save_dict['init_dict']
        self.agents[agent_ind].load_params(save_dict['agent_params'][agent_ind])
        # for a, params in zip(instance.agents, save_dict['agent_params']):
        #     a.load_params(params)
        # return instance

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, is_cuda="true"):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = []
        agent_types = []
        agent_type_dict = {'adversary': adversary_alg, 'capturer': adversary_alg, 'prey': agent_alg}
        for atype in env.agent_types:
            alg_types.append(agent_type_dict[atype])
            agent_types.append(atype)
            
        for acsp, obsp, algtype, agent_type in zip(env.action_space, env.observation_space, alg_types, agent_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)

            """Split the communication dim in action"""
            if isinstance(acsp, MultiDiscrete):
                # num_in_comm = acsp.nvec[1]
                num_in_comm = acsp.high[1] + 1
            else:
                num_in_comm = 0

            if algtype == "MADDPG":
                num_in_critic = 0
                for ind, oobsp in enumerate(env.observation_space): # need to investigate whose observation spaces we should include
                    if agent_types[ind] == agent_type:
                        num_in_critic += oobsp.shape[0] # add observation dimensions agent by agent FOR THE SAME TYPE (22+22+22/20/22+22)
                for ind, oacsp in enumerate(env.action_space): # need to investigate whose action spaces we should include
                    if agent_types[ind] == agent_type:
                        num_in_critic += get_shape(oacsp) # add action dimensions agent by agent FOR THE SAME TYPE (22+22+22/20/22+22)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_comm': num_in_comm})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_types': agent_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'is_cuda': is_cuda}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

class hierDDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64, critic_lr=0.01, policy_lr=0.01, discrete_action=True, device="cpu", constrained=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = hierMLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, out_fn=identity_map).to(device)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False).to(device)
        self.target_policy = hierMLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim, out_fn=identity_map).to(device)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False).to(device)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        if not discrete_action:
            self.exploration = OUNoise(8, scale=0.1, mu=0, theta=0.15, sigma=15)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.num_out_pol = num_out_pol
        self.device = device

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, episode_i, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        candidate_actions_size = 2 * self.num_out_pol
        candidate_actions = obs[-candidate_actions_size:].view(-1, 2)
        if episode_i % 2 == 1:
            subpolicy_coeffs = self.policy(obs).view(-1, 1) # temperature_softmax(self.policy(obs), temperature=1.0).view(-1, 1)      
        else:
            subpolicy_coeffs = self.policy(obs).view(-1, 1) # temperature_softmax(self.policy(obs), temperature=1.0).view(-1, 1)     
        # subpolicy_coeffs = torch.Tensor([[0], [0], [1]])
        if explore:
            subpolicy_coeffs = subpolicy_coeffs + Variable(Tensor(self.exploration.noise()).view(4,2).to(self.device), requires_grad=False)
        action = torch.sum((subpolicy_coeffs * candidate_actions), dim=0)
        # print("subpolicy_coeffs=", subpolicy_coeffs)
        # print("candidate_actions=", candidate_actions)
        # print("action=", action)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            # if explore:
                # action += Variable(Tensor(self.exploration.noise()).to(self.device),
                #                    requires_grad=False)
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

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_high_pol, num_in_low_pol, subpolicy_num_out_pol, para_num_out_pol, num_in_high_critic, num_in_low_critic, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=(64,64), device="cuda"):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.high_level_policy = HighLevelPolicy(state_shape=(num_in_high_pol,),
                        subpolicy_shape=(subpolicy_num_out_pol, ),
                        hidden_units=hidden_dim,
                        hidden_activation=nn.ReLU()).to(device)
        self.low_level_policy = LowLevelPolicy(
                        state_shape=(num_in_low_pol,),
                        para_shape=(para_num_out_pol,),
                        hidden_units=hidden_dim,
                        hidden_activation=nn.ReLU() # original: nn.Tanh()
                        ).to(device)
        self.target_high_policy = copy.deepcopy(self.high_level_policy).to(device)
        self.target_low_policy = copy.deepcopy(self.low_level_policy).to(device)

        self.low_critic = MLPNetwork(num_in_low_critic, 1, hidden_dim[0], nonlin=torch.nn.functional.relu).to(device)
        self.target_low_critic = copy.deepcopy(self.low_critic).to(device)
        self.high_critic = MLPNetwork(num_in_high_critic, 1, hidden_dim[0], nonlin=torch.nn.functional.relu).to(device)
        self.target_high_critic = copy.deepcopy(self.high_critic).to(device)


        self.niter = 0
        self.gamma = gamma
        self.tau = tau
                                        
        # hard_update(self.target_policy, self.policy)
        # hard_update(self.target_critic, self.critic)
        self.high_policy_optimizer = torch.optim.Adam(self.high_level_policy.parameters(), lr=lr)
        self.low_policy_optimizer = torch.optim.Adam(self.low_level_policy.parameters(), lr=lr) # original: lr
        self.high_critic_optimizer = torch.optim.Adam(self.high_critic.parameters(), lr=lr)
        self.low_critic_optimizer = torch.optim.Adam(self.low_critic.parameters(), lr=lr)
                                                            
        self.exploration = OUNoise(para_num_out_pol)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step_high(self, obs, target=False, explore=False):
        if target:
            subpolicy = self.target_high_policy(obs)
        else:
            subpolicy = self.high_level_policy(obs)
        return subpolicy

    def step_low(self, obs, target=False, explore=False):
        if target:
            paras = self.target_low_policy(obs)
        else:
            paras = self.low_level_policy(obs)
        if explore:
            paras += Variable(Tensor(self.exploration.noise()), requires_grad=False)
        return paras

    def step(self, obs, target=False, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if target:
            subpolicy, subpolicy_idx = self.target_high_policy(obs)
            paras = self.target_low_policies[subpolicy_idx](obs)
        else:
            subpolicy, subpolicy_idx = self.high_level_policy(obs)
            paras = self.low_level_policies[subpolicy_idx](obs)
        # subpolicy, paras = self.policy(obs)
        if explore:
            paras += Variable(Tensor(self.exploration.noise()), requires_grad=False)
        return subpolicy, subpolicy_idx, paras

    def get_params(self):
        return {'high_policy': self.high_level_policy.state_dict(),
                'low_policy': self.low_level_policy.state_dict(),
                'high_critic': self.high_critic.state_dict(),
                'low_critic': self.low_critic.state_dict(),
                'target_high_policy': self.target_high_policy.state_dict(),
                'target_low_policies': self.target_low_policy.state_dict(),
                # 'target_critic': self.target_critic.state_dict(),
                # 'policy_optimizer': self.policy_optimizer.state_dict(),
                # 'critic_optimizer': self.critic_optimizer.state_dict()
                }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        # self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        # self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    if isinstance(target, list):
        for t, s in zip(target, source):
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(param.data)
    else:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    return

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))]).squeeze()
    else:
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    # uniform_: Fills self tensor with numbers sampled from the continuous uniform distribution
    # https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html (Language: EN)
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
        y = gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = onehot_from_logits(y)
            y = (y_hard - y).detach() + y
            y = y.squeeze()
    else:
        y = gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = onehot_from_logits(y)
            y = (y_hard - y).detach() + y
    return y

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    if isinstance(target, list):
        for t, s in zip(target, source):
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    else:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)