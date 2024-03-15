import torch
import numpy as np
from torch.utils.data import Dataset
import copy
import os

class TrajGraderDataset(Dataset):
    def __init__(self, file_root):
        # file_paths should be a list of paths to NPZ files
        self.map_dim = 2428
        self.goal_rew = 50
        self.file_root = file_root
        self.process_first_graph = True

        self._load_data(file_root)
        

    def __len__(self):
        return self.file_num

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

        for np_file in np_files:
            self._load_file(np_file)

    def normalize(self, arr, mode="map"):
        arr = copy.deepcopy(arr)
        if mode == "map":
            arr = arr / self.map_dim
        elif mode == "diffusion":
            arr = arr / self.map_dim * 2 - 1
        elif mode == "reward":
            arr = arr / self.goal_rew 
        else:
            raise NotImplementedError
        return arr

    def _load_file(self, file):

        diffusion_path = file["diffusion_path"]
        blue_init_loc = file["blue_init_loc"]
        reward = file["reward"]
        red_detection_of_blue = file["redLocations_when_detectBlue"]
        prisoner_locations = file["prisoner_locs"]

        # INFO: normalize red locations
        normalized_diffusion_path = self.normalize(diffusion_path, mode="map")
        normalized_blue_init = self.normalize(blue_init_loc, mode="map")
        normalized_rew = self.normalize(reward, mode="reward")
        normalized_red_detection_of_blue = self.normalize(red_detection_of_blue, mode="map")
        normalized_prisoner_locations = self.normalize(prisoner_locations, mode="map")

        if self.process_first_graph:
            self.process_first_graph = False
            self.waypoints = [normalized_diffusion_path]
            self.blue_init = [normalized_blue_init]
            self.reward = [normalized_rew]
            self.red_detection_of_blue = [normalized_red_detection_of_blue]
            self.prisoner_locations = [normalized_prisoner_locations]
        else:
            self.waypoints.append(normalized_diffusion_path)
            self.blue_init.append(normalized_blue_init)
            self.reward.append(normalized_rew)
            self.red_detection_of_blue.append(normalized_red_detection_of_blue)
            self.prisoner_locations.append(normalized_prisoner_locations)

    def __getitem__(self, idx):
        waypoints = self.waypoints[idx].flatten()
        blue_init = self.blue_init[idx].flatten()
        detection = self.red_detection_of_blue[idx]
        reward = self.reward[idx]
        loc = self.prisoner_locations[idx]
        # input = torch.Tensor(np.concatenate((waypoints, blue_init)))
        input = torch.Tensor(waypoints)
        detection = torch.Tensor(detection)
        loc = torch.Tensor(loc)
        output = torch.Tensor(reward)
        return input, detection, loc, output


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