import enum
import os
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

import enum
import os
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

class SequenceReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_ac_dims, is_cuda, sequence_length=16, filter_dim = (331,), loc_dim = 2, max_timesteps=4320):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        assert len(obs_ac_dims) == 2
        self.save_filter_loc = False
        obs_dims = obs_ac_dims[0]
        ac_dims = obs_ac_dims[1]
        self.obs_dim = obs_dims[0]
        self.max_steps = max_steps # 1*(10^6)
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.device = 'cuda' if is_cuda else 'cpu'

        self.sequence_length = sequence_length
        self.max_timesteps = max_timesteps
        self.filter_dim = filter_dim

        # obs_dims = [26(predator), 26(predator), 26(predator), 24(prey), 24(capturer), 24(capturer)] in pcp environment
        # ac_dims = [5(predator), 5(predator), 5(predator), 5(prey), 6(capturer), 6(capturer)]
        for odim, adim in zip(obs_dims, ac_dims): # create obs, action, reward, next obs, done buffers for each agent
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        # for gnns
        self.agent_buffs = np.zeros((max_steps, ) + filter_dim)
        self.next_agent_buffs = np.zeros((max_steps, ) + filter_dim)
        self.hideout_buffs = np.zeros((max_steps, 2)) # currently just assume 1 known hideout
        self.timestep_buffs = np.zeros(max_steps)
        self.total_num_agents = np.zeros(max_steps)


        self.loc_buffs = np.zeros((max_steps, loc_dim))


        # if self.save_filter_loc == True:
        #     for filter_dim, loc_dim in zip(filter_dims, loc_dims):
        #         self.filtering_buffs.append(np.zeros((max_steps, *filter_dim)))
        #         self.loc_buffs.append(np.zeros((max_steps, loc_dim)))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, gnn_obs, next_gnn_obs, prisoner_location):
        nentries = 1  # handle multiple parallel environments (num of environments)
        agent_obs, hideout_obs, timestep_obs, total_num_agents = gnn_obs
        next_agent_obs, _, _, _ = next_gnn_obs 
        
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            # for the sequences, need to rollover the entire episode

            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                
                self.agent_buffs[agent_i] = np.roll(self.agent_buffs[agent_i], rollover, axis=0)
                self.next_agent_buffs[agent_i] = np.roll(self.next_agent_buffs[agent_i], rollover, axis=0)
                self.hideout_buffs[agent_i] = np.roll(self.hideout_buffs[agent_i], rollover, axis=0)
                self.timestep_buffs[agent_i] = np.roll(self.timestep_buffs[agent_i], rollover, axis=0)
                self.total_num_agents[agent_i] = np.roll(self.total_num_agents[agent_i], rollover, axis=0)

                self.loc_buffs[agent_i] = np.roll(self.loc_buffs[agent_i], rollover, axis=0)
                
            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i] # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones
        
        self.agent_buffs[self.curr_i:self.curr_i + nentries] = agent_obs
        self.next_agent_buffs[self.curr_i:self.curr_i + nentries] = next_agent_obs
        self.hideout_buffs[self.curr_i:self.curr_i + nentries] = hideout_obs
        self.timestep_buffs[self.curr_i:self.curr_i + nentries] = timestep_obs
        self.total_num_agents[self.curr_i:self.curr_i + nentries] = total_num_agents
        self.loc_buffs[self.curr_i:self.curr_i + nentries] = prisoner_location

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.sequence_length, self.filled_i), size=N,
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

        filtering_input = sample_sequence_from_array(self.agent_buffs, self.timestep_buffs, inds, self.sequence_length, self.filter_dim, self.max_timesteps)
        next_filtering_input = sample_sequence_from_array(self.next_agent_buffs, self.timestep_buffs, inds, self.sequence_length, self.filter_dim, self.max_timesteps)
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                (cast(filtering_input), cast(self.hideout_buffs[inds]), cast(self.timestep_buffs[inds]).unsqueeze(1), cast(self.total_num_agents[inds]).unsqueeze(1)), # gnn obs
                (cast(next_filtering_input), cast(self.hideout_buffs[inds]), cast(self.timestep_buffs[inds]).unsqueeze(1), cast(self.total_num_agents[inds]).unsqueeze(1)),
                cast(self.loc_buffs[inds]))
                # return 5 lists whose position is defined by inds

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

    def push_oar(self, observations, actions, rewards, next_observations, dones):
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

    def sample_oar_sequence_from_array(self, N, sequence_length, to_gpu=False, norm_rews=False):
        """
        Sample a sequence from the buffer of a given length
        Pad with 0's if the sequence is at the start
        """
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        inds = np.random.choice(np.arange(self.sequence_length, self.filled_i), size=N,
                            replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if norm_rews: # it is true by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        dones = [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]


        # agent_batch_reward = []
        agent_batch_obs = []
        # agent_batch_ac = []
        agent_batch_nobs = []
        # agent_batch_dones = []

        for agent_idx in range(self.num_agents):
            # rew_batch = []
            obs_batch = []
            # ac_batch = []
            nobs_batch = []
            # done_batch = []
            for index in inds:
                timestep = round(self.timestep_buffs[index] * self.max_timesteps)

                if timestep >= sequence_length - 1:
                    obs = self.obs_buffs[agent_idx][index-sequence_length+1:index+1]
                    # ac = self.ac_buffs[agent_idx][index-sequence_length+1:index+1]
                    # ret_rews = self.rew_buffs[agent_idx][index-sequence_length+1:index+1]
                    next_obs = self.next_obs_buffs[agent_idx][index-sequence_length+1:index+1]
                    # dones = self.done_buffs[agent_idx][index-sequence_length+1:index+1]

                else:
                    shape = (sequence_length-timestep-1, ) + (self.obs_dim, )
                    empty_sequences = np.zeros(shape)
                    
                    obs = self.obs_buffs[agent_idx][index-timestep:index+1]
                    obs = np.concatenate((empty_sequences, obs), axis=0)
                    # ac = self.ac_buffs[agent_idx][index-sequence_length+1:index+1]
                    # ac = np.concatenate((empty_sequences, ac), axis=0)
                    # ret_rews = self.rew_buffs[agent_idx][index-sequence_length+1:index+1]
                    # ret_rews = np.concatenate((empty_sequences, ret_rews), axis=0)
                    next_obs = self.next_obs_buffs[agent_idx][index-timestep:index+1]
                    next_obs = np.concatenate((empty_sequences, next_obs), axis=0)
                    # dones = self.next_obs_buffs[agent_idx][index-sequence_length+1:index+1]
                    # dones = np.concatenate((empty_sequences, dones), axis=0)
                obs = obs.reshape(obs.shape[0]*obs.shape[1])
                # ac = ac.reshape()
                next_obs = next_obs.reshape(next_obs.shape[0]*next_obs.shape[1])
                
                obs_batch.append(obs)
                # ac_batch.append(ac)
                # rew_batch.append(ret_rews)
                nobs_batch.append(next_obs)
                # done_batch.append(dones)
                desired_shape = (sequence_length, ) + (self.obs_dim, )
                assert nobs_batch[0].shape[0] == desired_shape[0]*desired_shape[1], "Wrong shape: %s, %s" % (nobs_batch[0].shape, desired_shape)

            # agent_batch_reward.append(cast(np.stack(rew_batch)))
            agent_batch_obs.append(cast(np.stack(obs_batch)))
            # agent_batch_ac.append(cast(np.stack(ac_batch)))
            agent_batch_nobs.append(cast(np.stack(nobs_batch)))
            # agent_batch_dones.append(cast(np.stack(done_batch)))

        return agent_batch_obs, [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)], ret_rews, agent_batch_nobs, dones


def sample_sequence_from_array(arr, timesteps, indices, sequence_length, filter_dim, max_timesteps):
    """
    Sample a sequence from the buffer of a given length
    Pad with 0's if the sequence is at the start
    """
    batch = []
    for index in indices:
        timestep = round(timesteps[index] * max_timesteps)

        if timestep >= sequence_length - 1:
            sequence = arr[index-sequence_length+1:index+1]

        else:
            shape = (sequence_length-timestep-1, ) + filter_dim
            empty_sequences = np.zeros(shape)
            sequence = arr[index-timestep:index+1]
            sequence = np.concatenate((empty_sequences, sequence), axis=0)

        batch.append(sequence)
        desired_shape = (sequence_length, ) + filter_dim
        assert sequence.shape == desired_shape, "Wrong shape: %s, %s" % (sequence.shape, desired_shape)
    return np.stack(batch)

