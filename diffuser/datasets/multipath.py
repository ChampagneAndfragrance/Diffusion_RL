import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy
import random
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
from diffuser.datasets.prisoner import pad_collate_detections, pad_collate_detections_repeat

global_device_name = "cpu"
global_device = torch.device("cpu")

class StateOnlyDataset(torch.utils.data.Dataset):
    """ Single stream dataset where we cannot tell which agent is which in the detections"""
    def __init__(self, 
                 folder_path, 
                 horizon,
                 dataset_type = "sponsor",
                 include_start_detection = False,
                 condition_path = True,
                 max_trajectory_length = 4320,
                 ):
        print("Loading dataset from: ", folder_path)

        # assert global_lstm_include_start # this variable is a remnant from past dataset

        self.condition_path = condition_path

        print("Condition Path: ", self.condition_path)

        self.dataset_type = dataset_type
        self.observation_dim = 2
        self.horizon = horizon
        self.max_trajectory_length = max_trajectory_length

        self.dones = []
        self.agent_locs = []
        self.process_first_graph = True

        self._load_data(folder_path)

        # These mark the end of each episode
        self.include_start_detection = include_start_detection
        self.indices = np.arange(self.file_num)


    def _load_data(self, folder_path):
        self.file_num = 0
        np_files = []
        fps = get_lowest_root_folders(folder_path)
        for fp in fps:
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                # print(np_file)
                # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
                np_files.append(np_file)
                self.file_num = self.file_num + 1

        self.set_normalization_factors()
        for np_file in np_files:
            self._load_file(np_file)

        # print("Path Lengths: ")
        # print(max(self.path_lengths), min(self.path_lengths))
        
        # normalize hideout locations
        if self.dataset_type == "prisoner_globe":
            for i in range(len(self.hideout_locs)):
                # INFO: find the target hideout for each file
                target_hideout_loc = self.find_target_hideout(self.hideout_locs[i][0].flatten(), i)
                self.target_hideout_locs[i] = self.normalize(target_hideout_loc)
                self.hideout_locs[i] = self.normalize(self.hideout_locs[i])

    def find_target_hideout(self, hideout_loc, path_ind):
        # INFO: find the hideout the prisoner is reaching
        red_path_terminal_loc = self.unnormalize(self.red_locs[path_ind][-1,:2])
        hideout_num = len(hideout_loc) // 2
        hideout_reached_id = 0
        hideout_reached_dist = np.inf
        for hideout_id in range(hideout_num):
            candidate_hideout_loc = hideout_loc[2*hideout_id:2*hideout_id+2]
            candidate_terminal_error = np.linalg.norm(red_path_terminal_loc - candidate_hideout_loc)
            if candidate_terminal_error < hideout_reached_dist:
                hideout_reached_dist = candidate_terminal_error
                hideout_reached_id = hideout_id
            else:
                pass
        hideout_loc = hideout_loc[2*hideout_reached_id:2*hideout_reached_id+2]
        return hideout_loc

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def normalize(self, arr):
        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)

        last_dim = obs.shape[-1]
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = obs[..., evens]
        obs[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., odds]
        obs[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x = obs[..., 0]
        # obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y = obs[..., 1]
        # obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x_1 = obs[..., 2]
        # obs[..., 2] = ((x_1 + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y_1 = obs[..., 3]
        # obs[..., 3] = ((y_1 + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs
    
    def unnormalize_single_dim(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs
    
    def select_random_rows(self, array, n):
        b, m = array.shape

        if n >= b:
            return array

        indices = np.arange(b)
        np.random.shuffle(indices)

        selected_indices = indices[:n]
        remaining_indices = indices[n:]

        selected_rows = np.full((n, m), -np.inf)
        selected_rows[:len(selected_indices)] = array[selected_indices]

        result = np.copy(array)
        result[remaining_indices] = -np.inf

        return result

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        red_locs = np.float32(file["red_locations"])[0]
        # INFO: what's this line doing?
        if red_locs.shape[-1] == 4:
            print(file)
        hideout_locs = np.float32(file["hideout_observations"])[0]
        path_length = len(red_locs)
        # INFO: path length should be smaller than the limit
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        # INFO: normalize red locations
        normalized_red_loc = self.normalize(red_locs)

        if self.process_first_graph:
            self.process_first_graph = False
            self.timesteps = [timesteps]
            self.red_locs = [normalized_red_loc]
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs = [hideout_locs]
                self.target_hideout_locs = [hideout_locs]
        else:
            self.timesteps.append(timesteps)
            self.red_locs.append(normalized_red_loc)
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs.append(hideout_locs)
                self.target_hideout_locs.append(hideout_locs)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        # assert len(global_cond_idx) != 0
            # return torch.tensor([[-1, -1, -1, -1, -1]])
        if len(global_cond_idx) == 0:
            return -1 * torch.ones((1, 213)) # 229 for 5s1h, 213 for 1s1h
        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)


        return torch.tensor(global_cond).float()

    def get_conditions(self, idx):
        '''
            condition on current observation for planning
        '''

        # INFO: get current path red loc
        red_loc = self.red_locs[idx]

        if self.condition_path:
            # always include the start of the path
            if self.include_start_detection:
                idxs = np.array([[0], [-1]])
                detects = np.array([red_loc[0], self.target_hideout_locs[idx]])
            else:
                idxs = np.array([])
                detects = np.array([])
        else:
            idxs = np.array([])
            detects = np.array([])

        return(idxs, detects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # INFO: which path file we are chosing?
        path_ind = self.indices[idx]

        prisoner_locs = self.red_locs[path_ind]

        local_cond = self.get_conditions(path_ind)

        hideout_loc = self.hideout_locs[path_ind]

        global_cond = hideout_loc[0].flatten()

        prisoner_at_start = np.array(self.red_locs[path_ind][0,:2])

        batch = (prisoner_locs, global_cond, local_cond, prisoner_at_start)
        return batch

    def collate_fn(self, batch):
        (prisoner_locs, global_cond, local_cond, prisoner_at_start) = zip(*batch)

        path = torch.tensor(np.stack(prisoner_locs, axis=0))

        global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # Pass this to condition our models rather than pass them separately
        global_dict = {"hideouts": global_cond.to(global_device_name), "red_start": torch.Tensor(prisoner_at_start).to(global_device_name)}

        return path, global_dict, local_cond
    
    def collate_fn_repeat(self, batch, num_samples):
        (global_cond, local_cond, prisoner_at_start) = zip(*batch)

        global_cond = torch.tensor(np.stack(global_cond, axis=0))

        global_cond = global_cond.repeat((num_samples, 1))
        local_cond = list(local_cond) * num_samples

        # INFO: This is for red traj only
        global_dict = {"hideouts": global_cond.to(global_device_name), 
            "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}

        return global_dict, local_cond

class NAgentsSingleDataset(torch.utils.data.Dataset):
    """ Single stream dataset where we cannot tell which agent is which in the detections"""
    def __init__(self, 
                 folder_path, 
                 horizon,
                 normalizer,
                 preprocess_fns,
                 use_padding,
                 max_path_length,
                 dataset_type = "sponsor",
                 include_start_detection = False,
                 global_lstm_include_start = False,
                 condition_path = True,
                 max_detection_num = 32,
                 max_trajectory_length = 4320,
                 num_detections = 16,

                 ):
        print("Loading dataset from: ", folder_path)

        self.global_lstm_include_start = global_lstm_include_start
        # assert global_lstm_include_start # this variable is a remnant from past dataset

        self.condition_path = condition_path

        print("Global LSTM Include Start: ", self.global_lstm_include_start)
        print("Condition Path: ", self.condition_path)

        self.dataset_type = dataset_type
        self.use_padding = use_padding
        self.observation_dim = 2
        self.horizon = horizon
        self.max_detection_num = max_detection_num
        self.max_trajectory_length = max_trajectory_length
        self.num_detections = num_detections

        self.dones = []
        self.agent_locs = []
        self.process_first_graph = True

        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]
        self.max_path_length = max_path_length
        self.include_start_detection = include_start_detection
        self.indices = self.make_indices(self.path_lengths, horizon)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def _load_data(self, folder_path):

        np_files = []
        fps = get_lowest_root_folders(folder_path)
        for fp in fps:
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                # print(np_file)
                # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
                np_files.append(np_file)

        self.set_normalization_factors()
        for np_file in np_files:
            self._load_file(np_file)

        print("Path Lengths: ")
        print(max(self.path_lengths), min(self.path_lengths))
        
        self.process_detections()

        # after processing detections, we can pad
        if self.use_padding:
            for i in range(len(self.agent_locs)):
                # need to add padding to the end of the red_locs
                # self.agent_locs[i] = np.pad(self.agent_locs[i], ((0, self.horizon), (0, 0)), 'constant', constant_values=self.agent_locs[i][-1])
                self.agent_locs[i] = np.pad(self.agent_locs[i], ((0, self.horizon), (0, 0)), 'edge')
        
        # normalize hideout locations
        if self.dataset_type == "prisoner_globe":
            for i in range(len(self.hideout_locs)):
                self.hideout_locs[i] = self.normalize(self.hideout_locs[i])

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def normalize(self, arr):
        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)

        last_dim = obs.shape[-1]
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = obs[..., evens]
        obs[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., odds]
        obs[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x = obs[..., 0]
        # obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y = obs[..., 1]
        # obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x_1 = obs[..., 2]
        # obs[..., 2] = ((x_1 + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y_1 = obs[..., 3]
        # obs[..., 3] = ((y_1 + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs
    
    def unnormalize_single_dim(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs

    def process_detections(self):
        self.detected_dics = []
        for detected_locs in self.detected_locations:
            indices = []
            detects = []
            for i in range(len(detected_locs)):
                loc = detected_locs[i]
                if np.any(loc[1::4]):
                    # need to convert from 0-1 to -1 to 1
                    loc[2::4] = loc[2::4] * 2 - 1
                    loc[3::4] = loc[3::4] * 2 - 1

                    indices.append(i)
                    detects.append(loc)
            detects = np.stack(detects, axis=0)
            indices = np.stack(indices, axis=0)
            self.detected_dics.append((indices, detects))
    
    def select_random_rows(self, array, n):
        b, m = array.shape

        if n >= b:
            return array

        indices = np.arange(b)
        np.random.shuffle(indices)

        selected_indices = indices[:n]
        remaining_indices = indices[n:]

        selected_rows = np.full((n, m), -np.inf)
        selected_rows[:len(selected_indices)] = array[selected_indices]

        result = np.copy(array)
        result[remaining_indices] = -np.inf

        return result

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        detected_locations = file["detected_locations"]
        red_locs = np.float32(file["red_locations"])
        blue_locs = np.float32(file["blue_locations"])
        agent_locs = np.concatenate((np.expand_dims(red_locs, axis=1), blue_locs), axis=1)

        if red_locs.shape[-1] == 4:
            print(file)
        hideout_locs = np.float32(file["hideout_observations"])[0]
        # timesteps = np.arange(agent_locs.shape[0]) / self.horizon
        path_length = len(agent_locs)
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")


        agents = []
        for i in range(agent_locs.shape[1]):
            agent = self.normalize(agent_locs[:, i, :])
            # agent_b = self.normalize(blue_locs[:, 1, :])
            agents.append(agent)

        # r_locs_normalized = np.concatenate((agent_a, agent_b), axis=1)
        locs_normalized = np.concatenate(agents, axis=1)

        # agent_random = [self.select_random_rows(agent, self.num_detections) for agent in agents]
        # agent_a = self.select_random_rows(agent_a, self.num_detections)
        # agent_b = self.select_random_rows(agent_b, self.num_detections)

        # detected_locations = np.concatenate((agent_a, agent_b), axis=1)
        # detected_locations = np.concatenate(agent_random, axis=1)

        if self.global_lstm_include_start:
            detected_locations[0] = copy.deepcopy(locs_normalized[0])

        if self.process_first_graph:
            self.process_first_graph = False
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.agent_locs = [locs_normalized]
            self.detected_locations = [detected_locations]
            self.path_lengths = [path_length]
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs = [hideout_locs]
        else:
            self.agent_locs.append(locs_normalized)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.detected_locations.append(detected_locations)
            self.path_lengths.append(path_length)
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs.append(hideout_locs)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        # assert len(global_cond_idx) != 0
            # return torch.tensor([[-1, -1, -1, -1, -1]])
        if len(global_cond_idx) == 0:
            return -1 * torch.ones((1, 213)) # 229 for 5s1h, 213 for 1s1h
        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)


        return torch.tensor(global_cond).float()

    def get_conditions(self, idx, start, end, trajectories):
        '''
            condition on current observation for planning
        '''
        detected_dic = self.detected_dics[idx]
        # subtract off the start and don't take anything past the end

        start_idx_find = np.where(detected_dic[0] >= start)[0]
        end_idx_find = np.where(detected_dic[0] < end)[0]
        # These are global conditions where the global_cond_idx is the 
        # integer index within the trajectory of where the detection occured

        # Take the detections before the start of the trajectory
        before_start_detects = np.where(detected_dic[0] <= start)[0]
        if len(before_start_detects) == 0:
            global_cond_idx = np.array([])
            global_cond = np.array([])
        else:
            global_cond_idx = detected_dic[0][:before_start_detects[-1]+1]
            global_cond = detected_dic[1][:before_start_detects[-1]+1]

        detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, start)

        if self.condition_path:
            if len(start_idx_find) == 0 or len(end_idx_find) == 0 or start_idx_find[0] > end_idx_find[-1]:
                # always include the start of the path
                if self.include_start_detection:
                    idxs = np.array([0])
                    detects = np.array([trajectories[0]])
                else:
                    idxs = np.array([])
                    detects = np.array([])
            else:
                start_idx = start_idx_find[0]
                end_idx = end_idx_find[-1]

                idxs = detected_dic[0][start_idx:end_idx+1] - start
                detects = detected_dic[1][start_idx:end_idx+1]

                if idxs[0] != 0 and self.include_start_detection:
                    idxs = np.concatenate((np.array([0]), idxs))
                    detects = np.concatenate((np.array([trajectories[0]]), detects))
        else:
            idxs = np.array([])
            detects = np.array([])

        return detection_lstm, (idxs, detects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        trajectories = self.agent_locs[path_ind][start:end]
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)

        hideout_loc = self.hideout_locs[path_ind]
        hideout_loc = self.find_target_hideout(hideout_loc, path_ind)
        global_cond = hideout_loc

        prisoner_at_start = np.concatenate((np.array([0]), np.array(trajectories[0,:2])))

        batch = (trajectories, global_cond, all_detections, conditions, prisoner_at_start)
        return batch
    
    def find_target_hideout(self, hideout_loc, path_ind):
        # INFO: find the hideout the prisoner is reaching
        red_path_terminal_loc = self.unnormalize(self.agent_locs[path_ind][-1,:2])
        hideout_num = len(hideout_loc) // 2
        hideout_reached_id = 0
        hideout_reached_dist = np.inf
        for hideout_id in range(hideout_num):
            candidate_hideout_loc = self.hideout_locs[path_ind][2*hideout_id:2*hideout_id+2] * 2428
            candidate_terminal_error = np.linalg.norm(red_path_terminal_loc - candidate_hideout_loc)
            if candidate_terminal_error < hideout_reached_dist:
                hideout_reached_dist = candidate_terminal_error
                hideout_reached_id = hideout_id
            else:
                pass
        hideout_loc = hideout_loc[2*hideout_reached_id:2*hideout_reached_id+2]
        return hideout_loc

    def collate_fn(self):
        return pad_collate_detections
    
    def collate_fn_repeat(self):
        return pad_collate_detections_repeat


class NAgentsIncrementalDataset(torch.utils.data.Dataset):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env
        self.set_normalization_factors()
        self.hideout_locs = (np.array(self.env.hideout_locations).astype(float) / self.env.dim_x)
        self.episode_t = 0
        self.max_detection_num = 32
        self.agent_locs = []
        self.red_observations = []
        self.red_detect_blue = []
        self.red_detect_blue_t = []
        self.gt_blue_states = []
        self.gt_blue_states_t = []

    def reset(self):
        self.red_observations = []
        self.red_detect_blue = []

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def push(self):
        self.push_red_obs()
        self.push_locations()
        self.push_gt_blue_state(freq=1)
        self.episode_t = self.episode_t + 1

    def push_gt_blue_state(self, freq):
        self.gt_blue_states.append(self.env.construct_gt_blue_state(freq))
        if self.episode_t % freq == 0:
            self.gt_blue_states_t.append(self.episode_t)


    def push_red_obs(self):
        red_obs = self.env.get_fugitive_observation()
        self.red_observations.append(red_obs)
        self.update_detections(red_obs)

    def push_locations(self):
        prisoner_location = self.env.get_prisoner_location()
        search_party_locations, helicopter_locations = self.env.get_blue_locations()
        blue_location = helicopter_locations + search_party_locations
        # agent_locs = np.array([prisoner_location]+blue_location)
        agent_locs = np.array([prisoner_location]+blue_location)
        self.agent_locs.append(self.normalize(agent_locs.astype(float)))

    def update_detections(self, red_obs):
        red_obs = copy.deepcopy(red_obs)
        wrapped_red_observation = self.env.obs_names(red_obs)
        # INFO: include all detections
        final_camera_idx = self.env.num_known_cameras + self.env.num_unknown_cameras - 1
        red_detections_of_blue = wrapped_red_observation.get_section_include_terminals(key_start="helicopter_detect_0", key_end="camera_detect_%d"%final_camera_idx)
        if np.any(red_detections_of_blue[1::4]):
            # need to convert from 0-1 to -1 to 1
            red_detections_of_blue[2::4] = red_detections_of_blue[2::4] * 2 - 1
            red_detections_of_blue[3::4] = red_detections_of_blue[3::4] * 2 - 1
            self.red_detect_blue.append(red_detections_of_blue)
            self.red_detect_blue_t.append(self.episode_t)
        else:
            pass

    def get_conditions(self, trajectories, max_detection_num=32):
        '''
            condition on current observation for planning
        '''

        if len(self.red_detect_blue) == 0:
            global_cond_idx = np.array([])
            global_cond = np.array([])
        else:
            global_cond_idx = self.red_detect_blue_t[-max_detection_num:]
            global_cond = self.red_detect_blue[-max_detection_num:]

        # INFO: This detection is for hsc
        # detection_lstm = self.convert_global_for_lstm(global_cond_idx=np.array(self.red_detect_blue_t), global_cond=self.red_detect_blue, start=self.episode_t-1)
        # INFO: This detection is for hs only
        # detection_lstm = self.convert_global_for_lstm(global_cond_idx=np.array(self.gt_blue_states_t), global_cond=self.gt_blue_states, start=self.episode_t-1)
        num_agents = len(self.gt_blue_states[0])
        detection_lstm = []
        for i in range(num_agents):
            detects = np.stack([x[i] for x in self.gt_blue_states])
            for_lstm = self.convert_global_for_lstm(global_cond_idx=np.array(self.gt_blue_states_t), global_cond=detects, start=self.episode_t-1)
            detection_lstm.append(for_lstm)
        idxs = np.array([[0]])
        detects = np.array([trajectories[0]])

        return detection_lstm, (idxs, detects)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        # assert len(global_cond_idx) != 0
            # return torch.tensor([[-1, -1, -1, -1, -1]])
        if len(global_cond_idx) == 0:
            return -1 * torch.ones((1, 13)) # 229 for 5s1h, 213 for 1s1h, 13 for hs only
        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.env.max_timesteps
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        return torch.tensor(global_cond).float()

    def __len__(self):
        return self.episode_t

    def __getitem__(self, idx):
        # path_ind, start, end = self.indices[idx]
        trajectories = self.agent_locs[-1]
        all_detections, conditions = self.get_conditions(trajectories)

        random_idx = np.random.randint(len(self.hideout_locs)//2)
        hideout_loc = self.hideout_locs
        global_cond = hideout_loc

        prisoner_at_start = np.concatenate((np.array([0]), np.array(trajectories[0,:2])))

        batch = (trajectories, global_cond, all_detections, conditions, prisoner_at_start)
        return batch

    def collate_fn(self):
        return pad_collate_detections
    
    def collate_fn_repeat(self):
        return pad_collate_detections_multiHideout

    def sel_collate_fn(self):
        return pad_collate_detections_selHideout

    def normalize(self, arr):
        arr = np.array(arr).astype(float)

        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

class RedBlueIncrementalDataset(NAgentsIncrementalDataset):
    def __init__(self, env):
        super().__init__(env)


    def push_gt_blue_state(self, freq):
        self.gt_blue_states.append(self.env.construct_gt_blue_state(freq))
        if self.episode_t % freq == 0:
            self.gt_blue_states_t.append(self.episode_t)

    def collate_fn(self):
        return pad_collate_detections_red_blue
    
    def collate_fn_repeat(self):
        return pad_collate_detections_repeat_red_blue
    
    def sel_collate_fn(self):
        return pad_collate_detections_selHideout_red_blue

    def get_conditions(self, trajectories, max_detection_num=32):
        '''
            condition on current observation for planning
        '''

        if len(self.red_detect_blue) == 0:
            global_cond_idx = np.array([])
            global_cond = np.array([])
        else:
            global_cond_idx = self.red_detect_blue_t[-max_detection_num:]
            global_cond = self.red_detect_blue[-max_detection_num:]

        num_agents = len(self.gt_blue_states[0])
        detection_lstm = []
        for i in range(num_agents):
            detects = np.stack([x[i] for x in self.gt_blue_states])
            for_lstm = self.convert_global_for_lstm(global_cond_idx=np.array(self.gt_blue_states_t), global_cond=detects, start=self.episode_t-1)
            detection_lstm.append(for_lstm)

        # detection_lstm = self.convert_global_for_lstm(global_cond_idx=np.array(self.gt_blue_states_t), global_cond=self.gt_blue_states, start=self.episode_t-1)
        idxs = np.array([[0]])
        detects = np.array([trajectories[0]])

        return detection_lstm, (idxs, detects)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        # assert len(global_cond_idx) != 0
            # return torch.tensor([[-1, -1, -1, -1, -1]])
        if len(global_cond_idx) == 0:
            return -1 * torch.ones((1, 3)) # 229 for 5s1h, 213 for 1s1h, 13 for hs only
        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.env.max_timesteps
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        return torch.tensor(global_cond).float()


def pad_collate_detections_red_blue(batch):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    # all detections is [batch x n_agents x tensors]

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    global_dict = {"hideouts": global_cond, "red_start": prisoner_at_start}
    for i, d in enumerate(detects):
        x_lens = [len(x) for x in d]
        xx_pad = pad_sequence(d, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions

def pad_collate_detections_repeat_red_blue(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    global_cond = global_cond.repeat((num_samples, 1, 1))
    conditions = list(conditions) * num_samples

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    red_start = (prisoner_at_start).repeat((num_samples,1))

    global_dict = {"hideouts": global_cond, "red_start": red_start}
    for i, d in enumerate(detects):
        d_mult = d * num_samples
        x_lens = [len(x) for x in d_mult]
        xx_pad = pad_sequence(d_mult, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions

def pad_collate_detections_selHideout_red_blue(batch, num_samples_each_hideout):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    data = data.repeat((num_samples_each_hideout, 1, 1))

    global_cond = torch.tensor(global_cond[0])
    hideout_num = global_cond.shape[0]
    samples_num = num_samples_each_hideout * hideout_num
    global_cond = torch.cat([global_cond[i].repeat(num_samples_each_hideout, 1) for i in range(hideout_num)], dim=0)
    

    # all_detections = list(all_detections) * samples_num
    conditions = list(conditions) * samples_num
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(samples_num)]

    x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=samples_num, dim=0)}
    # INFO: This is for red traj only

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    # global_dict = {"hideouts": global_cond.to(global_device_name), 
    #     # "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=samples_num, dim=0)
    #     "red_start": prisoner_at_start.to(global_device_name).repeat((samples_num, 1))
    #     }

    red_start = prisoner_at_start.repeat((samples_num,1))
    global_dict = {"hideouts": global_cond, "red_start": red_start}
    for i, d in enumerate(detects):
        d_mult = d * samples_num
        x_lens = [len(x) for x in d_mult]
        xx_pad = pad_sequence(d_mult, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions


class RedIncrementalDataset(NAgentsIncrementalDataset):
    def __init__(self, env) -> None:
        super().__init__(env)

    def push_locations(self):
        prisoner_location = self.env.get_prisoner_location()
        agent_locs = np.array([prisoner_location])
        self.agent_locs.append(self.normalize(agent_locs.astype(float)))        

    def __getitem__(self, idx):
        # path_ind, start, end = self.indices[idx]
        trajectories = self.agent_locs[-1]
        all_detections, conditions = self.get_conditions(trajectories)

        random_idx = np.random.randint(len(self.hideout_locs)//2)
        hideout_loc = self.hideout_locs
        global_cond = hideout_loc

        prisoner_at_start = np.concatenate((np.array([0]), np.array(trajectories[0,:2])))

        batch = (trajectories, global_cond, all_detections, conditions, prisoner_at_start)
        return batch

    def sel_collate_fn(self):
        return pad_collate_detections_selHideout

def pad_collate_detections(batch):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), "detections": detections.to(global_device_name), "red_start": torch.Tensor(prisoner_at_start).to(global_device_name)}

    return data, global_dict, conditions

def pad_collate_detections_multiHideout(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(global_cond[0])

    data = data.repeat((num_samples, 1, 1))

    hideout_ind_sel = torch.randint(low=0, high=global_cond.shape[0], size=(num_samples,))
    global_cond = global_cond[hideout_ind_sel,:]
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(num_samples)]

    # x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), 
        # "unpacked": torch.cat(all_detections, axis=0).to(global_device_name), 
            "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}
    # global_dict = {"hideouts": global_cond, "detections": detections}

    return data, global_dict, conditions

def pad_collate_detections_selHideout(batch, num_samples_each_hideout):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    data = data.repeat((num_samples_each_hideout, 1, 1))

    global_cond = torch.tensor(global_cond[0])
    hideout_num = global_cond.shape[0]
    samples_num = num_samples_each_hideout * hideout_num
    global_cond = torch.cat([global_cond[i].repeat(num_samples_each_hideout, 1) for i in range(hideout_num)], dim=0)
    

    all_detections = list(all_detections) * samples_num
    conditions = list(conditions) * samples_num
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(samples_num)]

    # x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=samples_num, dim=0)}
    # INFO: This is for red traj only

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    global_dict = {"hideouts": global_cond.to(global_device_name), 
        # "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=samples_num, dim=0)
        "red_start": prisoner_at_start.to(global_device_name).repeat((samples_num, 1))
        }

    return data, global_dict, conditions

def pad_collate_detections_repeat(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    global_cond = global_cond.repeat((num_samples, 1))
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=num_samples, dim=0)}
    # INFO: This is for red traj only
    global_dict = {"hideouts": global_cond.to(global_device_name), 
        "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}

    return data, global_dict, conditions

def get_lowest_root_folders(root_folder):
    lowest_folders = []
    
    # Get all items in the root folder
    items = os.listdir(root_folder)
    
    # Check if each item is a directory
    for item in items:
        item_path = os.path.join(root_folder, item)
        
        if os.path.isdir(item_path):
            # Recursively call the function for subfolders
            subfolders = get_lowest_root_folders(item_path)
            
            if not subfolders:
                # If there are no subfolders, add the current folder to the lowest_folders list
                lowest_folders.append(item_path)         
            lowest_folders.extend(subfolders)
    if len(lowest_folders) == 0:
        return [root_folder]
    return lowest_folders

class NAgentsRewardDataset(torch.utils.data.Dataset):
    def __init__(self, traj_max_num=10192) -> None:
        super().__init__()
        self.traj_max_num = traj_max_num
        self.traj_num = 0
        self.traj_pt_counter = 0
        self.seq_len = 240
        self.max_steps = 350
        self.set_normalization_factors()
        self.agent_locations = []
        self.red_rewards = []
        self.hideout_loc = []
        self.dones = []
        self.traj_lens = []

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def normalize(self, arr):
        arr = np.array(arr).astype(float)

        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr
    
    def push(self, prisoner_loc, blue_locs, red_rew, done, red_hideout):
        normalized_prisonerLoc = self.normalize([prisoner_loc])
        normalized_blueLocs = self.normalize(blue_locs)
        agent_locs = np.concatenate((normalized_prisonerLoc, normalized_blueLocs), axis=0).reshape(-1)
        if self.traj_num >= self.traj_max_num:
            # INFO: pop out the first traj
            del self.agent_locations[:self.traj_lens[0]]
            del self.red_rewards[:self.traj_lens[0]]
            del self.dones[:self.traj_lens[0]]
            del self.hideout_loc[:self.traj_lens[0]]
            self.traj_lens.pop(0)
            self.traj_num = self.traj_num - 1
        if self.traj_num < self.traj_max_num:
            self.agent_locations.append(agent_locs)
            self.red_rewards.append(red_rew)
            self.dones.append(done)
            self.hideout_loc.append(red_hideout)
            self.traj_pt_counter = self.traj_pt_counter + 1
            if done:
                self.traj_num = self.traj_num + 1
                self.traj_lens.append(self.traj_pt_counter)
                self.traj_pt_counter = 0

    def __len__(self):
        return len(self.agent_locations)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        curr_batch_loc = self.agent_locations[start_idx:end_idx]
        curr_hideout_loc = self.hideout_loc[start_idx:end_idx]
        curr_batch_rew = self.red_rewards[start_idx:end_idx]
        if end_idx < len(self):
            traj_end_localized_step = np.where(self.dones[start_idx:end_idx])[0]
            # traj_start_localized_step = np.where(self.dones[start_idx-self.max_steps:start_idx])[-1]
            if len(traj_end_localized_step) != 0:
                step_loc = np.stack(curr_batch_loc[0:traj_end_localized_step[0]+1], axis=0)
                step_rew = np.stack(curr_batch_rew[0:traj_end_localized_step[0]+1], axis=0)
                hideout = curr_hideout_loc[traj_end_localized_step[0]]
                step_loc = np.pad(step_loc, ((0, self.seq_len-(traj_end_localized_step[0]+1)), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, self.seq_len-(traj_end_localized_step[0]+1)), (0, 0)), 'constant')
            else:
                step_loc = np.stack(curr_batch_loc, axis=0)
                step_rew = np.stack(curr_batch_rew, axis=0)
                hideout = curr_hideout_loc[-1]
            # # INFO: 
            # for i in range(self.max_steps):
            #     if self.dones[start_idx-i] and i != 0:
            #         break
        else:
            if len(curr_batch_loc) != 0:
                step_loc = np.stack(curr_batch_loc, axis=0)
                step_rew = np.stack(curr_batch_rew, axis=0)
                hideout = curr_hideout_loc[-1]
                step_loc = np.pad(step_loc, ((0, end_idx-len(self)), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, end_idx-len(self)), (0, 0)), 'constant')
            else:
                step_loc = np.stack([self.agent_locations[-1]], axis=0)
                step_rew = np.stack([self.red_rewards[-1]], axis=0)
                hideout = self.hideout_loc[-1]
                step_loc = np.pad(step_loc, ((0, self.seq_len-1), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, self.seq_len-1), (0, 0)), 'constant')

        condition = (np.array([]), np.array([]))
        prisoner_at_start = np.concatenate((np.array([0]), np.array(step_loc[0,:2])))
        return step_loc, step_rew, hideout, condition, torch.Tensor(prisoner_at_start)

    def collate_loc_reward(self):
        return pad_loc_reward

    def collate_loc(self):
        return pad_loc

def pad_loc_reward(batch, gamma, period):
    step_loc, step_rew, _, _, _ = zip(*batch)

    batches_seqLen_agentLocations = torch.Tensor(np.stack(step_loc, axis=0)).to(global_device_name)
    red_rews = torch.Tensor(np.stack(step_rew, axis=0)).squeeze()
    seq_len = batches_seqLen_agentLocations.shape[1]
    discount_factors = torch.Tensor([gamma**i for i in range(seq_len)])
    batches_seqLen_redRews = torch.sum(red_rews*discount_factors, axis=-1, keepdim=True).to(global_device_name)

    return batches_seqLen_agentLocations[:,::period,:2], batches_seqLen_redRews

def pad_loc(batch):
    step_loc, step_rew, hideout, condition, prisoner_at_start = zip(*batch)
    batches_seqLen_agentLocations = torch.Tensor(np.stack(step_loc, axis=0)).to(global_device_name)
    hideout = torch.stack(hideout, dim=0).to(global_device_name)
    prisoner_at_start = torch.stack(prisoner_at_start, dim=0).to(global_device_name)
    # INFO: construct the global condition
    global_dict = {"hideouts": hideout, "red_start": prisoner_at_start}
    return batches_seqLen_agentLocations[:,:,:2], global_dict, condition

def update_raw_traj(raw_red_downsampled_traj, detected_blue_states, red_vel, perception_max_thresh=0.1, perception_min_thresh=0.05):
    raw_red_downsampled_traj = copy.deepcopy(raw_red_downsampled_traj)
    repulse_vec = torch.zeros_like(raw_red_downsampled_traj).to(global_device_name)
    for detects in detected_blue_states:
        detect_loc = torch.Tensor([detects[0]]).to(global_device_name)
        detect_vel = torch.Tensor(detects[1]).to(global_device_name)

        blue_to_pt = raw_red_downsampled_traj - detect_loc
        dist_from_blue_to_pt = torch.norm(blue_to_pt, dim=-1, keepdim=True)
        dist_from_blue_to_pt[dist_from_blue_to_pt>perception_max_thresh] = 1e6
        dist_from_blue_to_pt[dist_from_blue_to_pt<perception_min_thresh] = 0.05

        # INFO: vertical to relative vel
        relative_vel = detect_vel - red_vel
        if relative_vel[0] == 0 and relative_vel[1] == 0:
            repulse_direction_vec = blue_to_pt
            repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
            repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt
        else:
            repulse_direction_vec = blue_to_pt - torch.inner(blue_to_pt, relative_vel).unsqueeze(-1) / torch.norm(relative_vel) @ (relative_vel / torch.norm(relative_vel)).unsqueeze(0)
            repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
            repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt        

        # INFO: vertical to abs vel
        # if detect_vel[0] == 0 and detect_vel[1] == 0:
        #     repulse_direction_vec = blue_to_pt
        #     repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
        #     repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt
        # else:
        #     repulse_direction_vec = blue_to_pt - torch.inner(blue_to_pt, detect_vel).unsqueeze(-1) / torch.norm(detect_vel) @ (detect_vel / torch.norm(detect_vel)).unsqueeze(0)
        #     repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
        #     repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt

        raw_red_downsampled_traj = raw_red_downsampled_traj + repulse_vec
    return raw_red_downsampled_traj


if __name__ == "__main__":

    # data_path = "/home/sean/october_datasets/multiagent/rectilinear"
    # data_path = "/home/sean/PrisonerEscape/datasets/multiagent/rectilinear"
    # data_path = "/home/sean/october_datasets/multiagent/sinusoidal"
    data_path = "/home/wu/Research/Diffuser/data/prisoner_datasets/october_datasets/gnn_map_0_run_600_AStar_only_dr"

    # data_path = "/home/sean/PrisonerEscape/datasets/multiagent/AStar"
    dataset = NAgentsSingleDataset(data_path,                  
                 horizon = 60,
                 normalizer = None,
                 global_lstm_include_start=False,
                 condition_path = False)
    
    # print(dataset[0])

    def cycle(dl):
        while True:generate_path_samples
    train_batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn())

    for i in dataloader:
        pass

    # for i in range(len(dataset)):
    #     print(dataset[i])