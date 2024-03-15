import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer

class PrisonerDatasetCondition(torch.utils.data.Dataset):
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
                 end_pad = 60):
        """ This dataset is uses the pad_packed_sequence function to ensure the entire detection history is located in the LSTM """
        print("Loading dataset from: ", folder_path)

        self.global_lstm_include_start = global_lstm_include_start
        self.condition_path = condition_path

        print("Global LSTM Include Start: ", self.global_lstm_include_start)
        print("Condition Path: ", self.condition_path)

        self.dataset_type = dataset_type
        self.use_padding = use_padding
        self.observation_dim = 2
        self.horizon = horizon
        self.max_detection_num = max_detection_num
        self.max_trajectory_length = max_trajectory_length
        self.end_pad = end_pad

        self.dones = []
        self.red_locs = []
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
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # print(np_file)
            # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_file(np_file)

        print("Path Lengths: ")
        print(max(self.path_lengths), min(self.path_lengths))

        self.set_normalization_factors()
        for i in range(len(self.red_locs)):
            self.red_locs[i] = self.normalize(self.red_locs[i])
        self.process_detections()

        # after processing detections, we can pad
        if self.use_padding:
            for i in range(len(self.red_locs)):
                # need to add padding to the end of the red_locs
                self.red_locs[i] = np.pad(self.red_locs[i], ((0, self.horizon), (0, 0)), 'edge')
        
        # normalize hideout locations
        if self.dataset_type == "prisoner_globe":
            for i in range(len(self.hideout_locs)):
                self.hideout_locs[i] = self.normalize(self.hideout_locs[i])

    def set_normalization_factors(self):
        if self.dataset_type == "sponsor" or self.dataset_type == "prisoner_globe":
            all_red_locs = np.concatenate(self.red_locs, axis=0)

            self.min_x = min(all_red_locs[:, 0])
            self.max_x = max(all_red_locs[:, 0])
            self.min_y = min(all_red_locs[:, 1])
            self.max_y = max(all_red_locs[:, 1])
        else:
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
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return obs

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
            # prisoner dataset
            detected_locations = file["detected_locations"]
            red_locs = np.float32(file["red_locations"])
            hideout_locs = np.float32(file["hideout_observations"])[0]
        else:
            # sponsor dataset
            detected_locations = file["all_detections_of_fugitive"]
            red_locs = np.float32(file["red_locations"])
        
        path_length = len(red_locs)
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        if self.global_lstm_include_start:
            detected_locations[0] = copy.deepcopy(file["red_locations"][0]) / 2428

        if self.process_first_graph:
            self.process_first_graph = False
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = [red_locs]
            self.detected_locations = [detected_locations]
            self.path_lengths = [path_length]
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs = [hideout_locs]
        else:
            self.red_locs.append(red_locs)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.detected_locations.append(detected_locations)
            self.path_lengths.append(path_length)
            if self.dataset_type == "prisoner" or self.dataset_type == "prisoner_globe":
                self.hideout_locs.append(hideout_locs)

    def process_detections(self):
        self.detected_dics = []
        for detected_locs in self.detected_locations:
            indices = []
            detects = []
            for i in range(len(detected_locs)):
                loc = detected_locs[i]
                if loc[0] != -1:
                    if self.dataset_type == "sponsor" or self.dataset_type == "prisoner_globe":
                        # sponsor dataset needs normalization
                        loc[0] = (loc[0] - self.min_x) / (self.max_x - self.min_x)
                        loc[1] = (loc[1] - self.min_y) / (self.max_y - self.min_y)
                    elif self.dataset_type == "prisoner":
                        # need to convert from 0-1 to -1 to 1
                        loc[0] = loc[0] * 2 - 1
                        loc[1] = loc[1] * 2 - 1
                    indices.append(i)
                    detects.append(loc)
            detects = np.stack(detects, axis=0)
            indices = np.stack(indices, axis=0)
            self.detected_dics.append((indices, detects))

    def _preprocess_detections(self, detected_locs, timestamps):
        """ Given a numpy array of [T x 2] where if there is a detection, the value is (x, y) and if there is not, the value is (-1, -1)
        
        For each row in the array, return all previous detections before that row
        Also need to add the time difference between each step so we return a [dt, x, y] for each detection
        """
        processed_detections = []
        detected_locs = self.coordinate_transform(detected_locs)
        detected_locs = np.concatenate((detected_locs, timestamps), axis=1)
        for i in range(detected_locs.shape[0]):
            curr_detections = copy.deepcopy(detected_locs[:i+1])
            curr_detections = curr_detections[curr_detections[:, 0] != -1]
            curr_detections[:, 2] = detected_locs[i, 2] - curr_detections[:, 2]
            processed_detections.append(curr_detections)
        return processed_detections

    def get_conditions(self, idx, start, end, trajectories):
        '''
            condition on current observation for planning
        '''
        detected_dic = self.detected_dics[idx]
        # subtract off the start and don't take anything past the end

        # self.end_pad is used to ensure that we have no detections in this region.
        # This is so we can call this part the prediction region
        start_idx_find = np.where(detected_dic[0] >= start)[0]
        end_idx_find = np.where(detected_dic[0] < end - self.end_pad)[0]

        # These are global conditions where the global_cond_idx is the 
        # integer index within the trajectory of where the detection occured

        # Take the detections before the start of the trajectory
        before_start_detects = np.where(detected_dic[0] < end - self.end_pad)[0]
        if len(before_start_detects) == 0:
            global_cond_idx = np.array([])
            global_cond = np.array([])
        else:
            global_cond_idx = detected_dic[0][:before_start_detects[-1]]
            global_cond = detected_dic[1][:before_start_detects[-1]]

        detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, end - self.end_pad)

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

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        if len(global_cond_idx) == 0:
            return torch.tensor([[-1, -1, -1]])

        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        return torch.tensor(global_cond)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        trajectories = self.red_locs[path_ind][start:end]
        # conditions = self.get_conditions(trajectories)
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)

        timestep = start / self.max_trajectory_length

        hideout_loc = self.hideout_locs[path_ind]
        global_cond = np.concatenate((hideout_loc, np.array([timestep], dtype=float)))

        batch = (trajectories, global_cond, all_detections, conditions)
        return batch

def pad_collate_detections(batch):
    (data, global_cond, all_detections, conditions) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond, "detections": detections}

    return data, global_dict, conditions

def pad_collate_global(batch):
    (data, global_cond, conditions) = zip(*batch)
    
    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    return data, global_cond, conditions

def pad_collate(batch):
    (data, conditions) = zip(*batch)
    
    data = torch.tensor(np.stack(data, axis=0))

    return data, conditions

if __name__ == "__main__":

    # data_path = "/data/prisoner_datasets/sponsor_datasets/processed_sponsor/train"
    # data_path = "/data/prisoner_datasets/october_datasets/4_detect/train"
    # data_path = "/home/sean/october_datasets/4_detect/train"

    # data_path = "/home/sean/october_datasets/3_detect/test"
    data_path = "/home/sean/october_datasets/7_detect/train"

    dataset = PrisonerDatasetCondition(data_path,                  
                 horizon = 120,
                 normalizer = None,
                 preprocess_fns = None,
                 use_padding = True,
                 max_path_length = 40000,
                 dataset_type = "prisoner",
                 include_start_detection=False,
                 global_lstm_include_start=True,
                 condition_path = False,
                 end_pad = 60)

    
    # print(dataset.path_lengths)
    gt_path = dataset[0][0]
    print(gt_path.shape)