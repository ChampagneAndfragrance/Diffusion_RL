import enum
import os
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
import copy

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, buffer_dims, is_cuda):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        # INFO: Only obs and act
        if len(buffer_dims) == 2:
            self.save_detection = False
            obs_dims = buffer_dims[0]
            ac_dims = buffer_dims[1]
        # INFO: Obs, act, detection_input and detection_output
        elif len(buffer_dims) == 4:
            self.save_detection = True
            obs_dims = buffer_dims[0]
            ac_dims = buffer_dims[1]
            detection_in_dims = buffer_dims[2]
            detection_out_dims = buffer_dims[3]
        else:
            pass
        
        # self.agents_groups_dims = num_agents * num_gaussians
        self.max_steps = max_steps # 1*(10^6)
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.filtering_buffs = []
        self.next_filtering_buffs = []
        self.loc_buffs = []
        # self.agents_groups_buffs = []
        self.detection_in_buffs = []
        self.detection_out_buffs = []
        self.device = 'cuda' if is_cuda else 'cpu'

        # obs_dims = [26(predator), 26(predator), 26(predator), 24(prey), 24(capturer), 24(capturer)] in pcp environment
        # ac_dims = [5(predator), 5(predator), 5(predator), 5(prey), 6(capturer), 6(capturer)]
        for odim, adim in zip(obs_dims, ac_dims): # create obs, action, reward, next obs, done buffers for each agent
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))
            # self.agents_groups_buffs.append(np.array([[] for _ in range(max_steps])))

        if self.save_detection == True:
            for d_in_dim, d_out_dim in zip(detection_in_dims, detection_out_dims):
                self.detection_in_buffs.append(np.zeros((max_steps, *d_in_dim)))
                self.detection_out_buffs.append(np.zeros((max_steps, d_out_dim)))
            

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, detection_in=None, detection_out=None):
        nentries = 1  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                if detection_in is not None:
                    self.detection_in_buffs[agent_i] = np.roll(self.detection_in_buffs[agent_i], rollover, axis=0)
                    self.detection_out_buffs[agent_i] = np.roll(self.detection_out_buffs[agent_i], rollover, axis=0)
                
            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i] # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones
            if detection_in is not None:
                self.detection_in_buffs[agent_i][self.curr_i:self.curr_i + nentries] = detection_in
                self.detection_out_buffs[agent_i][self.curr_i:self.curr_i + nentries] = detection_out

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def push_filter(self, observations, actions, rewards, next_observations, dones, filter_save, next_filter_save, prisoner_save, agents_groups=None):
        nentries = 1  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                self.filtering_buffs[agent_i] = np.roll(self.filtering_buffs[agent_i], rollover, axis=0)
                self.next_filtering_buffs[agent_i] = np.roll(self.next_filtering_buffs[agent_i], rollover, axis=0)
                self.loc_buffs[agent_i] = np.roll(self.loc_buffs[agent_i], rollover, axis=0)
                # self.agents_groups_buffs[agent_i] = np.roll(self.agents_groups_buffs[agent_i], rollover, axis=0)
                
            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i] # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones
            self.filtering_buffs[agent_i][self.curr_i:self.curr_i + nentries] = filter_save
            self.next_filtering_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_filter_save
            self.loc_buffs[agent_i][self.curr_i:self.curr_i + nentries] = prisoner_save

            # self.agents_groups_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agents_groups

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is true by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def sample_threat(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is true by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.detection_in_buffs[i][inds-1]) for i in range(self.num_agents)],
                [cast(self.detection_in_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds


    def sample_filter(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is true by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.filtering_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_filtering_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.loc_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def trace_back_sample(self, N, mode="filter"):
        trace_back_num = int(np.minimum(N, self.filled_i))
        inds = np.arange(self.curr_i - trace_back_num, self.curr_i) % self.max_steps
        if mode == "filter":
            return ([(self.filtering_buffs[i][inds]) for i in range(self.num_agents)],
                    [(self.loc_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def seq_sample(self, N, seq_len, to_gpu=False, mode="detection"):
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        batch_num = int(np.minimum(N, self.filled_i))
        start_inds = np.random.choice(np.arange(self.filled_i), size=batch_num, replace=False)
        end_inds = start_inds + seq_len
        dones_sample = [torch.vstack([torch.any(cast(self.done_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps])==1).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
        detection_in_sample = [cast(self.detection_in_buffs[i][start_inds]) for i in range(self.num_agents)]
        detection_out_sample = [torch.vstack([cast(self.detection_out_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps]).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
        # obs_sample = [torch.vstack([cast(self.obs_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps]).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
        # detection_sample = [torch.vstack([cast(self.detection_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps]).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
        return (detection_in_sample, detection_out_sample, dones_sample)

    # def recent_seq_sample(self, N, seq_len, to_gpu=False, mode="detection"):
    #     if to_gpu:
    #         cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
    #     else:
    #         cast = lambda x: Variable(Tensor(x), requires_grad=False)
    #     batch_num = int(np.minimum(N, self.filled_i))
    #     start_inds = np.random.choice(np.arange(self.filled_i), size=batch_num, replace=False)
    #     start_inds = np.arange()
    #     end_inds = start_inds + seq_len
    #     obs_sample = [torch.vstack([cast(self.obs_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps]).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
    #     detection_sample = [torch.vstack([cast(self.detection_buffs[ag_i][np.arange(start_inds[i], end_inds[i]) % self.max_steps]).unsqueeze(0) for i in range(N)]) for ag_i in range(self.num_agents)]
    #     return (obs_sample, detection_sample)

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

    def get_std_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].std() for i in range(self.num_agents)]


class IndividualReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, obs_ac_filter_loc_dim, is_cuda):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        if len(obs_ac_filter_loc_dim) == 2:
            self.save_filter_loc = False
            obs_dim = obs_ac_filter_loc_dim[0]
            ac_dim = obs_ac_filter_loc_dim[1]
        elif len(obs_ac_filter_loc_dim) == 4:
            self.save_filter_loc = True
            obs_dim = obs_ac_filter_loc_dim[0]
            ac_dim = obs_ac_filter_loc_dim[1]
            filter_dim = obs_ac_filter_loc_dim[2]
            loc_dim = obs_ac_filter_loc_dim[3]
        else:
            pass
        
        # self.agents_groups_dims = num_agents * num_gaussians
        self.max_steps = max_steps # 1*(10^6)
        self.obs_buffs = np.zeros((max_steps, obs_dim))
        self.ac_buffs = np.zeros((max_steps, ac_dim))
        self.rew_buffs = np.zeros(max_steps)
        self.next_obs_buffs = np.zeros((max_steps, obs_dim))
        self.done_buffs = np.zeros(max_steps)
        self.filtering_buffs = np.zeros((max_steps, *filter_dim))
        self.next_filtering_buffs = np.zeros((max_steps, *filter_dim))
        self.loc_buffs = np.zeros((max_steps, loc_dim))
        # self.agents_groups_buffs = []
        self.device = 'cuda' if is_cuda else 'cpu'

        # obs_dims = [26(predator), 26(predator), 26(predator), 24(prey), 24(capturer), 24(capturer)] in pcp environment
        # ac_dims = [5(predator), 5(predator), 5(predator), 5(prey), 6(capturer), 6(capturer)]
        
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = 1  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                
            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i] # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def push_filter(self, observations, actions, rewards, next_observations, dones, filter_save, next_filter_save, prisoner_save, agents_groups=None):
        nentries = 1  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs_buffs = np.roll(self.obs_buffs, rollover, axis=0) # np.roll: rolling end to start by the unit of rollover
            self.ac_buffs = np.roll(self.ac_buffs, rollover, axis=0)
            self.rew_buffs = np.roll(self.rew_buffs, rollover)
            self.next_obs_buffs = np.roll(self.next_obs_buffs, rollover, axis=0)
            self.done_buffs = np.roll(self.done_buffs, rollover)
            self.filtering_buffs = np.roll(self.filtering_buffs, rollover, axis=0)
            self.next_filtering_buffs = np.roll(self.next_filtering_buffs, rollover, axis=0)
            self.loc_buffs = np.roll(self.loc_buffs, rollover, axis=0)
            # self.agents_groups_buffs[agent_i] = np.roll(self.agents_groups_buffs[agent_i], rollover, axis=0)
                
            self.curr_i = 0
            self.filled_i = self.max_steps


        self.obs_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(observations) # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
        # actions are already batched by agent, so they are indexed differently
        self.ac_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(actions)
        self.rew_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(rewards)
        self.next_obs_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(next_observations)
        self.done_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(dones)
        self.filtering_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(filter_save)
        self.next_filtering_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(next_filter_save)
        self.loc_buffs[self.curr_i:self.curr_i + nentries] = copy.deepcopy(prisoner_save)


            # self.agents_groups_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agents_groups

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is true by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds


    def sample_filter(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is true by default
            ret_rews = cast((self.rew_buffs[inds] -
                              self.rew_buffs[:self.filled_i].mean()) /
                             self.rew_buffs[:self.filled_i].std())
                         # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = cast(self.rew_buffs[inds])
        return (cast(self.obs_buffs[inds]),
                cast(self.ac_buffs[inds]),
                ret_rews,
                cast(self.next_obs_buffs[inds]),
                cast(self.done_buffs[inds]),
                cast(self.filtering_buffs[inds]),
                cast(self.next_filtering_buffs[inds]),
                cast(self.loc_buffs[inds])) # return 5 lists whose position is defined by inds

    def trace_back_sample(self, N, to_gpu=False, norm_rews=False):
        trace_back_num = int(np.minimum(N, self.filled_i))
        inds = np.arange(self.curr_i - trace_back_num, self.curr_i) % self.max_steps
        return ([(self.filtering_buffs[i][inds]) for i in range(self.num_agents)],
                [(self.loc_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return self.rew_buffs[inds].mean()

    def get_std_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return self.rew_buffs[inds].std()


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, reward_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, *reward_shape), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = torch.from_numpy(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def individual_append(self, item_names, items, pointer_move_flag):
        for (name, item) in zip(item_names, items):
            getattr(self, name)[self._p].copy_(torch.from_numpy(item) if (not torch.is_tensor(item)) else item.detach())
        if pointer_move_flag:
            self._p = (self._p + 1) % self.buffer_size
            self._n = min(self._n + 1, self.buffer_size)      
        else:
            pass
              
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)

class AdditionalBuffer(Buffer):
    """ Used to add filtering module into the buffer """
    def __init__(self, buffer_size, state_shape, filtering_input_shape, prisoner_loc_shape, action_shape, reward_shape, device):
        super().__init__(buffer_size, state_shape, action_shape, reward_shape, device)

        self.filtering_input = torch.empty(
            (buffer_size, *filtering_input_shape), dtype=torch.float, device=device)
        self.prisoner_loc = torch.empty(
            (buffer_size, *prisoner_loc_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, filtering_input, prisoner_loc, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = torch.from_numpy(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.filtering_input[self._p].copy_(torch.tensor(filtering_input))
        self.prisoner_loc[self._p].copy_(torch.tensor(prisoner_loc) / 2428)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)
    
    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.filtering_input[idxes],
            self.prisoner_loc[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
            'filtering_input': self.filtering_input.clone().cpu(),
            'prisoner_loc': self.prisoner_loc.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )


class RolloutBufferHier:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.zeros(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.zeros(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.zeros(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.zeros(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = [torch.zeros(1, 1) for _ in range(self.total_size)]
        # self.log_pis = torch.zeros(
        #     (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.zeros(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = log_pi
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

        return self._p - 1

    def append_rew(self, reward):
        self.rewards[self._p] = float(reward)
        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def accumulate_reward(self, start_idx, gamma):
        acc_rew = 0
        for acc_idx, rew_idx in enumerate(range(start_idx, self._n)):
            acc_rew = acc_rew + self.rewards[rew_idx] * np.power(gamma, acc_idx)
        return acc_rew

    def get_property(self, idx, name):
        property_val = getattr(self, name)[idx]
        return property_val


    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
