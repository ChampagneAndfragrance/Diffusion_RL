import copy
import math
from types import SimpleNamespace

import gc
import os
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import copy

from PIL import Image
from dataclasses import dataclass
from gym import spaces
from tqdm import tqdm
from enum import Enum, auto
from numpy import genfromtxt
from collections import namedtuple

from simulator.forest_coverage.generate_square_map import generate_square_map
from simulator.forest_coverage.autoencoder import ConvAutoencoder
from simulator.abstract_object import AbstractObject, DetectionObject
from simulator.camera import Camera
from simulator.fugitive import Fugitive
from simulator.helicopter import Helicopter
from simulator.hideout import Hideout
from simulator.search_party import SearchParty
from simulator.terrain import Terrain
from simulator.utils import create_camera_net
# from blue_policies.blue_heuristic import BlueHeuristic

from simulator.observation_spaces import create_observation_space_ground_truth, create_observation_space_fugitive, create_observation_space_blue_team, create_observation_space_prediction
from simulator.observation_spaces import transform_blue_detection_of_fugitive, transform_predicted_detection_of_fugitive, create_partial_observation_space_blue_team
from simulator.observation_spaces import create_observation_space_blue_team_original
from simulator.forest_coverage.autoencoder import produce_terrain_embedding

from blue_bc.utils import sort_filtering_output, get_localized_trgt_gaussian_locations, localize_filtering_mu

class ObservationType(Enum):
    Fugitive = auto()
    FugitiveGoal = auto()
    Blue = auto()
    GroundTruth = auto()
    Prediction = auto()


@dataclass
class RewardScheme:
    time: float = 0 # -0.2 # -0.1
    detected = 3.0
    known_detected: float = 1.0 # 0.5 # 1.0 # 2.0, 2.5
    known_undetected: float = 0.01 # -0.5
    unknown_detected: float = 2.0
    unknown_undetected: float = -0.5
    timeout: float = 3.0


presets = RewardScheme.presets = SimpleNamespace()
presets.default = RewardScheme()
presets.none = RewardScheme(0., 0., 0., 0., 0., 0.)
presets.any_hideout = RewardScheme(known_detected=1., known_undetected=1., unknown_detected=1., unknown_undetected=1.,
                                   timeout=-1.)
presets.time_only = copy.copy(presets.none)
presets.time_only.time = presets.default.time
presets.timeout_only = copy.copy(presets.none)
presets.timeout_only.timeout = -3.
del presets  # access as RewardScheme.presets


# from simulator.utils import overlay_transparent

class PrisonerBothEnv(gym.Env):
    """
    PrisonerEnv simulates the prisoner behavior in a grid world.
    The considered factors include
        - prisoner
        - terrain (woods, dense forest, high mountains)
        - hideouts (targets)
        - max-time (72 hours)
        - cameras
        - helicopters
        - search parties

    *Detection is encoded by a three tuple [b, x, y] where b in binary. If b=1 (detected), [x, y] will have the detected location in world coordinates. If b=0 (not detected), [x, y] will be [-1, -1].

    State space
        - Terrain
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, hideouts, search parties, fugitive]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Fugitive's detection of [helicopters, helicopter dropped cameras, search parties]

    Observation space (fugitive)
        - Time
        - Locations of [known cameras, hideouts]
        - Self location, speed, heading
        - Detection of [helicopters, helicopter dropped cameras, search parties]

    Action space
        - 2 dimensional: speed [1,15] x direction [-pi, pi]

    Observation space (good guys')
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, search parties, known hideouts]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Terrain

    Coordinate system:
        - By default, cartesian Coordinate:
        y
        ^
        |
        |
        |
        |
        |----------->x

    Limitations:
        - Food and towns are not implemented yet
        - Sprint time maximum is not implemented yet. However, sprinting does still have drawbacks (easier to be detected)
        - No rain/fog
        - Fixed terrain, fixed hideout locations
        - Detection does not utilize an error ellipse. However, detection still has the range-based PoD.
        - Helicopter dropped cameras are not implemented yet.

    Details:
        - Good guys mean the side of search parties (opposite of the fugitive)
        - Each timestep represents 1 min, and the episode horizon is T=4320 (72 hours)
        - Each grid represents 21 meters
        - Our grid size is 2428x2428 as of now, representing a 50.988 kms x 50.988 kms field
        - We have continuous speed profile from 1 grid/timestep to 15 grids/timestep (1.26km/h to 18.9km/h)
        - Right now we have by default:
            - 2 search parties
            - 1 helicopter
            - 5 known hideouts
            - 5 unknown hideouts
            - 5 known cameras, 5 unknown cameras, with another 5 known cameras on known hideouts (to encode the fact that we can always detect the fugitive when he/she goes to known hideouts)
    """

    def __init__(self,
                 terrain=None,
                 terrain_map=None,
                 num_towns=0,
                 num_search_parties=2,
                 num_helicopters=1,
                 helicopter_battery_life=360,
                 helicopter_recharge_time=360,
                 spawn_mode='normal',
                 spawn_range=15.,
                 blue_spawn_mode = 'random',
                 blue_spawn_range = 250.,
                 max_timesteps=4320,
                 hideout_radius=50.,
                 reward_scheme=None,
                 random_hideout_locations=False,
                 num_known_hideouts=1,
                 num_unknown_hideouts=2,
                 known_hideout_locations=[[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624],
                                          [234, 171], [2398, 434], [633, 2136], [1590, 2]],
                 unknown_hideout_locations=[[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883],
                                            [1077, 1445], [602, 372], [80, 2274], [279, 477]],
                 random_cameras=False,
                 min_distance_from_hideout_to_start=1000,
                 num_random_unknown_cameras=25,
                 num_random_known_cameras=25,
                 camera_net_bool=False,
                 camera_net_path=None,
                 mountain_locations=[(400, 300), (1600, 1800)],
                 camera_range_factor=1,
                 camera_file_path="simulator/camera_locations/original.txt",
                 observation_step_type="Fugitive",  # Fugitive, Blue, GroundTruth
                 observation_terrain_feature=True,
                 include_camera_at_start=False,
                 include_start_location_blue_obs=False,
                 stopping_condition=False,
                 step_reset=True,
                 debug=False,
                 store_last_k_fugitive_detections=False,
                 search_party_speed=6.5,
                 helicopter_speed = 127,
                 fugitive_speed_limit = 7.5,
                 include_fugitive_location_in_blue_obs=False,
                 gnn_agent_last_detect="no", 
                 comm_dim=32,
                    # if True, append global last detections to blue gnn agent observations (for marl)
                    # if False, append agent specific last detections to blue gnn agent observations (for marl)
                 detection_factor = 4.0,
                 reward_setting = "high",
                 ):
        """
        PrisonerEnv simulates the prisoner behavior in a grid world.
        :param terrain: If given, the terrain is used from this object
        :param terrain_map_file: This is the file that contains the terrain map, only used if terrain is None
            If none, the default map is used.
            Currently all the maps are stored in "/star-data/prisoner-maps/"
                We load in the map from .npy file, we use csv_generator.py to convert .nc to .npy
            If directory, cycle through all the files upon reset
            If single .npy file, use that file
        :param num_towns:
        :param num_search_parties:
        :param num_helicopters:
        :param random_hideout_locations: If True, hideouts are placed randomly with num_known_hideouts and num_unknown_hideouts
            If False, hideouts are selected from known_hideout_locations and unknown_hideout_locations based on the num_known_hideouts and num_unknown_hideouts
        :param num_known_hideouts: number of hideouts known to good guys
        :param num_unknown_hideouts: hideouts unknown to the good guys
        :param: known_hideout_locations: locations of known hideouts when random_hideout_locations=False
        :param: unknown_hideout_locations: locations of unknown hideouts when random_hideout_locations=False
        :param helicopter_battery_life: how many minutes the helicopter can last in the game
        :param helicopter_recharge_time: how many minutes the helicopter need to recharge itself
        :param spawn_mode: how the prisoner location is initialized on reset. Can be:
            'normal': the prisoner is spawned in the northeast corner
            'uniform': the prisoner spawns at a uniformly sampled random location
            'uniform_hideout_dist': spawn the prisoner at min_distance_from_hideout_to_start from the hideouts
                        This assumes the hideouts are chosen first
            'hideout': the prisoner spawns within `spawn_range` of the hideout
        :param spawn_range: how far from the edge of the hideout the prisoner spawns in 'hideout' mode, or how far from the corner the prisoner spawn in 'corner' mode
        :param max_timesteps: time horizon for each rollout. Default is 4320 (minutes = 72 hours)
        :param hideout_radius: minimum distance from a hideout to be considered "on" the hideout
        :param reward_scheme: a RewardScheme object definining reward scales for different events. If omitted, a default will be used. A custom one can be constructed. Several presets are available under RewardScheme.presets.
        :param known_hideout_locations: list of tuples of known hideout locations
        :param unknown_hideout_locations: list of tuples of unknown hideout locations
        :param random_cameras: boolean of whether to use random camera placements or fixed camera placements
        :param num_random_unknown_cameras: number of random unknown cameras
        :param num_random_known_cameras: number of random known cameras
        :param camera_file_path: path to the file containing the camera locations for the unknown cameras. This it for us to test the Filtering algorithm
        :param camera_net_bool: boolean of whether to use the camera net around the fugitive or not
        :param camera_net_path: if None, place camera net by generating, if path, use the path
        :observation_step_type: What observation is returned in the "step" and "reset" functions
            'Fugitive': Returns fugitive observations
            'Blue': Returns observations from the BlueTeam (aka blue team's vision of the fugitive)
            'GroundTruth': Returns information of all agents in the environment
            'Prediction': Returns fugitive observations but without the unknown hideouts
        :observation_terrain_feature: boolean of whether to include the terrain feature in the observation
        :stopping_condition: boolean of whether to stop the game when the fugitive produces 0 speed
        :step_reset: boolean of whether to reset the game after the episode is over or just wait at the final location no matter what action is given to it
            This is to make the multi-step prediction rollouts to work properly.
            Default is True 
        :param include_start_location_blue_obs: boolean of whether to include the start location of the prisoner in the blue team observation
            Default is True
        :param store_last_k_fugitive_detections: Whether or not to store the last k(=8) detections of the fugitive

        """

        self.stopping_condition = stopping_condition
        self.terrain_list = []
        self.DEBUG = debug
        self.store_last_k_fugitive_detections = store_last_k_fugitive_detections
        self._cached_terrain_image = None
        forest_color_scale = 1
        self.reward_setting = reward_setting
        # If no terrain is provided, we read from map file
        if terrain is None:
            if terrain_map is None:
                # use original map with size 2428x2428
                dim_x = 2428
                dim_y = 2428
                percent_dense = 0.30
                size_of_dense_forest = int(dim_x * percent_dense)
                forest_density_array = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=dim_x,
                                                           dim_y=dim_y)
                forest_density_list = [forest_density_array]
                self.terrain_list = [Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale=forest_color_scale,
                                             forest_density_array=forest_density_array,
                                             mountain_locations=mountain_locations)]
            else:
                # if directory, cycle through all the files
                if os.path.isdir(terrain_map):
                    forest_density_list = []
                    for f in os.listdir(terrain_map):
                        if f.endswith(".npy"):
                            forest_density_array = np.load(os.path.join(terrain_map, f))
                            forest_density_list.append(forest_density_array)
                            dim_x, dim_y = forest_density_array.shape
                            self.terrain_list.append(
                                Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale=forest_color_scale,
                                        forest_density_array=forest_density_array,
                                        mountain_locations=mountain_locations))
                else:
                    forest_density_array = np.load(terrain_map)
                    forest_density_list = [forest_density_array]
                    dim_x, dim_y = forest_density_array.shape
                    self.terrain_list = [Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale=forest_color_scale,
                                                 forest_density_array=forest_density_array,
                                                 mountain_locations=mountain_locations)]
        else:
            # Getting terrain from terrain object
            # Assume we are just using a single terrain object
            # TODO: make this robust when we are switching terrains
            forest_density_list = [terrain.forest_density_array]
            self.terrain_list = [terrain]

        if observation_terrain_feature:
            # we save these to add to the observations
            model = ConvAutoencoder()
            model.load_state_dict(torch.load('simulator/forest_coverage/autoencoder_state_dict.pt'))
            self._cached_terrain_embeddings = [produce_terrain_embedding(model, terrain_np) for terrain_np in
                                               forest_density_list]
            terrain_embedding_size = self._cached_terrain_embeddings[0].shape[0]
        else:
            # empty list
            self._cached_terrain_embeddings = [np.array([])] * len(forest_density_list)
            terrain_embedding_size = 0

        # initialize terrain for this run
        self.set_terrain_paramaters()
        self.prisoner = Fugitive(self.terrain, [2400, 2400], fugitive_speed_limit=fugitive_speed_limit)  # the actual spawning will happen in set_up_world

        # Read in the cameras from file
        if random_cameras: 
            self.num_random_unknown_cameras = num_random_unknown_cameras
            self.num_random_known_cameras = num_random_known_cameras 
        else:
            self.camera_file_path = camera_file_path
            self.known_camera_locations, self.unknown_camera_locations = self.read_camera_file(camera_file_path)

        self.include_camera_at_start = include_camera_at_start
        
        self.dim_x = self.terrain.dim_x
        self.dim_y = self.terrain.dim_y

        # self.num_unknown_cameras = self.num_unknown_cameras
        self.num_towns = num_towns
        self.num_search_parties = num_search_parties
        self.num_helicopters = num_helicopters
        self.random_hideout_locations = random_hideout_locations

        self.num_known_hideouts = num_known_hideouts
        self.num_unknown_hideouts = num_unknown_hideouts
        # self.num_known_cameras = self.num_known_cameras + num_known_hideouts # add camera for each known hideout

        self.helicopter_battery_life = helicopter_battery_life
        self.helicopter_recharge_time = helicopter_recharge_time
        self.spawn_mode = spawn_mode
        self.spawn_range = spawn_range
        self.blue_spawn_mode = blue_spawn_mode
        self.blue_spawn_range = blue_spawn_range
        self.hideout_radius = hideout_radius
        self.reward_scheme = reward_scheme or RewardScheme()  # accept a custom or use the default
        self.known_hideout_locations = known_hideout_locations
        self.unknown_hideout_locations = unknown_hideout_locations
        self.random_cameras = random_cameras
        self.camera_file_path = camera_file_path
        self.camera_range_factor = camera_range_factor
        self.current_prisoner_speed = 3  # initialize prisoner speed, used to render detection range
        self.current_prisoner_velocity = np.zeros(2)
        self.search_party_speed = search_party_speed
        self.helicopter_speed = helicopter_speed
        self.fugitive_speed_limit = fugitive_speed_limit
        self.step_reset = step_reset
        self.camera_net_bool = camera_net_bool
        self.camera_net_path = camera_net_path
        self.include_start_location_blue_obs = include_start_location_blue_obs
        self.min_distance_from_hideout_to_start = min_distance_from_hideout_to_start
        self.include_fugitive_location_in_blue_obs = include_fugitive_location_in_blue_obs
        
        self.detection_factor = detection_factor
        DetectionObject.detection_factor = detection_factor
        self.search_party_speed = search_party_speed
        self.helicopter_speed = helicopter_speed


        self.max_timesteps = max_timesteps  # 72 hours = 4320 minutes
        self._max_episode_steps = max_timesteps  # 72 hours = 4320 minutes
        self.gnn_agent_last_detect = gnn_agent_last_detect

        assert isinstance(self.reward_scheme, (type(None), str, RewardScheme))
        if isinstance(self.reward_scheme, str):
            self.reward_scheme = getattr(RewardScheme.presets, self.reward_scheme)

        self.action_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([15, np.pi]))

        # initialization of variables
        self.camera_list = []
        self.helicopters_list = []
        self.hideout_list = []
        self.search_parties_list = []
        self.town_list = []
        self.comm_dim = comm_dim
        self.comm = [np.zeros(self.comm_dim) for i in range(self.num_helicopters + self.num_search_parties)]
        self.timesteps = 0
        self.near_goal = False
        self.done = False
        self.is_detected = False
        self.is_cam_detected = False
        self.last_detected_timestep = 0
        self.prisoner_detected_loc_history = [0, 0]
        self.prisoner_detected_loc_history2 = [-1, -1, -1, -1]
        self.nonlocalized_trgt_gaussians = None
        self.maddpg_agents = None
        

        # initialize objects
        self.set_up_world()

        self.modified_blue_observation_space, self.modified_blue_obs_names = create_observation_space_blue_team(
                                        num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size,
                                        include_start_location_blue_obs=include_start_location_blue_obs,
                                        include_fugitive_location_in_blue_obs=include_fugitive_location_in_blue_obs)

        self.blue_observation_space, self.blue_obs_names = create_observation_space_blue_team_original(
                                        num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size,
                                        include_start_location_blue_obs=include_start_location_blue_obs,
                                        include_fugitive_location_in_blue_obs=include_fugitive_location_in_blue_obs)

        self.blue_partial_observation_space, self.blue_partial_obs_names = create_partial_observation_space_blue_team(num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size,
                                        include_start_location_blue_obs=include_start_location_blue_obs)
        self.fugitive_observation_space, self.fugitive_obs_names = create_observation_space_fugitive(num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts, 
                                        num_unknown_hideouts=self.num_unknown_hideouts, 
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size)

        self.gt_observation_space, self.gt_obs_names = create_observation_space_ground_truth(num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts, 
                                        num_unknown_hideouts=self.num_unknown_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size)

        self.prediction_observation_space, self.prediction_obs_names = create_observation_space_prediction(num_known_cameras=self.num_known_cameras,
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters,
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size)
        self.blue_action_space_shape = (2 * (self.num_search_parties + self.num_helicopters),)

        self.prisoner_location_history = [self.prisoner.location.copy()]

        # load image assets
        self.known_camera_pic = Image.open("simulator/assets/camera_blue.png")
        self.unknown_camera_pic = Image.open("simulator/assets/camera_red.png")
        self.known_hideout_pic = Image.open("simulator/assets/star.png")
        self.unknown_hideout1_pic = Image.open("simulator/assets/star_blue.png")
        self.unknown_hideout2_pic = Image.open("simulator/assets/star_blue.png")
        self.town_pic = Image.open("simulator/assets/town.png")
        self.search_party_pic = Image.open("simulator/assets/searching.png")
        self.helicopter_pic = Image.open("simulator/assets/helicopter.png")
        self.prisoner_pic = Image.open("simulator/assets/prisoner.png")
        self.detected_prisoner_pic = Image.open("simulator/assets/detected_prisoner.png")

        self.known_camera_pic_cv = cv2.imread("simulator/assets/camera_blue.png", cv2.IMREAD_UNCHANGED)
        self.unknown_camera_pic_cv = cv2.imread("simulator/assets/camera_red.png", cv2.IMREAD_UNCHANGED)
        self.known_hideout_pic_cv = cv2.imread("simulator/assets/star.png", cv2.IMREAD_UNCHANGED)
        self.unknown_hideout1_pic_cv = cv2.imread("simulator/assets/star_blue.png", cv2.IMREAD_UNCHANGED)
        self.unknown_hideout2_pic_cv = cv2.imread("simulator/assets/star_red.png", cv2.IMREAD_UNCHANGED)
        self.town_pic_cv = cv2.imread("simulator/assets/town.png", cv2.IMREAD_UNCHANGED)
        self.search_party_pic_cv = cv2.imread("simulator/assets/searching.png", cv2.IMREAD_UNCHANGED)
        self.helicopter_pic_cv = cv2.imread("simulator/assets/helicopter.png", cv2.IMREAD_UNCHANGED)
        self.helicopter_no_pic_cv = cv2.imread("simulator/assets/helicopter_no.png", cv2.IMREAD_UNCHANGED)
        self.prisoner_pic_cv = cv2.imread("simulator/assets/prisoner.png", cv2.IMREAD_UNCHANGED)
        self.detected_prisoner_pic_cv = cv2.imread("simulator/assets/detected_prisoner.png", cv2.IMREAD_UNCHANGED)

        self.default_asset_size = 52
        """Zixuan newly added property starts"""
        self.costmap = np.maximum(self.terrain.world_representation[0,:,:] * 10.0, self.terrain.world_representation[1,:,:])
        """Zixuan newly added property ends"""
        # Store (t,x,y, vel) for last k detections. Only updated if store_last_k_fugitive_detections is True
        init_detection = [0] + (np.array(self.prisoner_start_location) / 2428).tolist()
        init_detection_vel = init_detection + [0, 0]
        self.last_k_blue_obs_only_red_loc = [[0, 0] for _ in range(15)] + [(np.array(self.prisoner_start_location) / 2428).tolist()]

        if not self.include_start_location_blue_obs:
            self.last_k_fugitive_detections_vel = [[-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0],
                                           [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0]]  
            self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                            [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        else:
            self.last_k_fugitive_detections_vel = [[-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0],
                                            [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], init_detection_vel, init_detection_vel]
            self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                            [-1, -1, -1], [-1, -1, -1], init_detection, init_detection]    

        # INFO: Initialize the last k red detection of blue
        self.last_k_fugitive_detect_blue_posVel = np.zeros((8, self.num_helicopters+self.num_search_parties, 5))

        # INFO: Initialize the last k [red_state, blue_relatives]
        self.last_k_red_blue_states = np.zeros((16, 12))

        # INFO: Initialize the lask k blue detection of red
        self.lask_k_blue_detection_of_red = np.zeros((16, 1))

        # This is for MADDPG shared parameters
        # map from agent id to agent type
        # We currently hard code order of search party observations and helicopter observations
        self.type_dict = {}
        for i in range(self.num_search_parties):
            self.type_dict[i] = 0
        for  i in range(self.num_helicopters):
            self.type_dict[i+num_search_parties] = 1
        self.total_agents = self.num_search_parties + self.num_helicopters
        self.filter_model = None

        # INFO: piecewise RL
        self.waypoints = None
        self.waypt_idx = 1

        
        # self.predicted_locations_from_last_two_detections = [[-1, -1]]
        # self._cached_terrain_image = self.terrain.visualize(just_matrix=True)
        # self.render(show=True)

    def read_camera_file(self, camera_file_path):
        """Generate a lists of camera objects from file

        Args:
            camera_file_path (str): path to camera file

        Raises:
            ValueError: If Camera file does not have a u or k at beginning of each line

        Returns:
            (list, list): Returns known camera locations and unknown camera locations
        """
        unknown_camera_locations = []
        known_camera_locations = []
        camera_file = open(camera_file_path, "r").readlines()
        for line in camera_file:
            line = line.strip().split(",")
            if line[0] == 'u':
                unknown_camera_locations.append([int(line[1]), int(line[2])])
            elif line[0] == 'k':
                known_camera_locations.append([int(line[1]), int(line[2])])
            else:
                raise ValueError(
                    "Camera file format is incorrect, each line must start with 'u' or 'k' to denote unknown or known")
        return known_camera_locations, unknown_camera_locations

    def set_terrain_paramaters(self):
        """ Sets self.terrain_embedding, self.terrain, and self._cached_terrain_image"""
        # Choose a random terrain from the list
        terrain_index = random.randint(0, len(self.terrain_list) - 1)
        self.terrain = self.terrain_list[terrain_index]
        # self._cached_terrain_image = self._cached_terrain_images[terrain_index]
        self._terrain_embedding = self._cached_terrain_embeddings[terrain_index]

    def place_random_hideouts(self):
        for known_hid in range(self.num_known_hideouts):
            if known_hid == 0:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                while np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location) < self.min_distance_from_hideout_to_start:
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                if self.DEBUG:
                    print('prisoner location: ', self.prisoner.location)
                    print('hideout location: ', location)
                    print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=True))
            else:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                # make sure hideout is far from each other and far from start location
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                while np.linalg.norm(np.array([location[0], location[
                    1]]) - self.prisoner.location) <= self.min_distance_from_hideout_to_start \
                        or (dists < self.min_distance_between_hideouts).any():
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                    s = [tuple(i.location) for i in self.hideout_list]
                    dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                if self.DEBUG:
                    print('prisoner location: ', self.prisoner.location)
                    print('hideout location: ', location)
                    print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=True))

        for unknown_hid in range(self.num_unknown_hideouts):
            if len(self.hideout_list) >=1:
                # make sure hideout is far from each other and far from start location
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                while np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location) <= self.min_distance_from_hideout_to_start\
                        or (dists < self.min_distance_between_hideouts).any():
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                    s = [tuple(i.location) for i in self.hideout_list]
                    dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                    if self.DEBUG:
                        print('prisoner location: ', self.prisoner.location)
                        print('hideout location: ', location)
                        print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=False))
            else:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=False))

    def place_fixed_hideouts(self):
        # specify hideouts' locations. These are passed in from the input args
        # We select a number of hideouts from num_known_hideouts and num_unknown_hideouts

        assert self.num_known_hideouts <= len(self.known_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.known_hideout_locations)}) greater than number of known hideouts {self.num_known_hideouts}"
        assert self.num_unknown_hideouts <= len(self.unknown_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.unknown_hideout_locations)}) greater than number of known hideouts {self.num_unknown_hideouts}"

        known_hideouts = random.sample(self.known_hideout_locations, self.num_known_hideouts)
        unknown_hideouts = random.sample(self.unknown_hideout_locations, self.num_unknown_hideouts)

        self.hideout_list = []
        self.unknown_hideout_list = []
        self.known_hideout_list = []
        for hideout_location in known_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=True))
            self.known_hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=True))

        for hideout_location in unknown_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=False))
            self.unknown_hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=False))

    def set_up_world(self):
        """
        This function places all the objects,
        Right now,
            - cameras are initialized randomly
            - helicopter is initialized randomly
            - hideouts are initialized always at [20, 80], [100, 20]
            - search parties are initialized randomly
            - prisoner is initialized by different self.spawn_mode
        """
        self.camera_list = []
        self.helicopters_list = []
        self.hideout_list = []
        self.search_parties_list = []
        self.town_list = []
        self.hideout_list = []
        self.comm = [np.zeros(self.comm_dim) for i in range(self.num_helicopters + self.num_search_parties)]
        self.min_distance_between_hideouts = 300

        # randomized
        if not self.random_hideout_locations:
            self.place_fixed_hideouts()
        else:
            raise NotImplementedError
            # Random hideouts have not been implemented with spawn mode as uniform hideout dist
            # Random hideouts need to have prisoner location initialized first
            self.place_random_hideouts()

        if self.spawn_mode == 'normal':
            prisoner_location = [2400, 2400] # original [2400, 2400]
        elif self.spawn_mode == 'uniform':
            # in_mountain = True
            mountain_range = 150
            near_mountain = 0
            while near_mountain < mountain_range:
                # We do not want to place the fugitive on a mountain!
                prisoner_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                # We do not want to place the fugitive within a distance of the mountain!
                m_dists = np.array([np.linalg.norm(np.array(prisoner_location) - np.array([m[1], m[0]])) for m in self.terrain.mountain_locations])
                near_mountain = min(m_dists)
        elif self.spawn_mode == 'uniform_hideout_dist':
            # Spawn uniformly on the map but with a distance of at least min_distance_from_hideout_to_start
            mountain_range = 150
            near_mountain = 0
            min_dist = 0
            while near_mountain < mountain_range or min_dist < self.min_distance_from_hideout_to_start:
                prisoner_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array([math.sqrt((prisoner_location[0] - s0) ** 2 + (prisoner_location[1] - s1) ** 2) for s0, s1 in s])
                min_dist = min(dists)
                
                # We do not want to place the fugitive within a distance of the mountain!
                m_dists = np.array([np.linalg.norm(np.array(prisoner_location) - np.array([m[1], m[0]])) for m in self.terrain.mountain_locations])
                near_mountain = min(m_dists)
        elif self.spawn_mode == 'hideout':
            in_mountain = True
            in_map = False
            while in_mountain or not in_map:
                # We do not want to place the fugitive on a mountain or outside the map!
                hideout_id = np.random.choice(range(len(self.unknown_hideout_list)))
                hideout_loc = np.array(self.unknown_hideout_list[hideout_id].location)
                angle = np.random.random() * 2 * math.pi - math.pi
                radius = self.hideout_radius + np.random.random() * self.spawn_range
                vector = np.array([math.cos(angle), math.sin(angle)]) * radius
                prisoner_location = (hideout_loc + vector).astype(int).tolist()
                in_mountain = self.terrain.world_representation[0, np.minimum(prisoner_location[0], self.dim_x-1), np.minimum(prisoner_location[1], self.dim_y-1)] == 1
                in_map = prisoner_location[0] in range(0, self.dim_x) and \
                         prisoner_location[1] in range(0, self.dim_y)
        elif self.spawn_mode == 'corner':
            # generate the fugitive randomly near the top right corner
            prisoner_location = AbstractObject.generate_random_locations_with_range([self.dim_x-self.spawn_range, self.dim_x],                                                                                    [self.dim_y-self.spawn_range, self.dim_y])
        else:
            raise ValueError('Unknown spawn mode "%s"' % self.spawn_mode)
        self.prisoner = Fugitive(self.terrain, prisoner_location, fugitive_speed_limit=self.fugitive_speed_limit)
        self.prisoner_start_location = prisoner_location

        # specify cameras' initial locations
        if(self.random_cameras):
            # randomized 
            known_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in range(self.num_random_known_cameras)]
            unknown_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in range(self.num_random_unknown_cameras)]
        else:
            known_camera_locations = self.known_camera_locations[:]
            unknown_camera_locations = copy.deepcopy(self.unknown_camera_locations)
        
        if self.camera_net_bool:
            if self.camera_net_path is None:
                cam_locs = create_camera_net(prisoner_location, dist_x=360, dist_y=360, spacing=30, include_camera_at_start=self.include_camera_at_start)
                unknown_camera_locations.extend(cam_locs.tolist())
            else:
                known_net, unknown_net = self.read_camera_file(self.camera_net_path)
                known_camera_locations.extend(known_net)
                unknown_camera_locations.extend(unknown_net)
        elif self.include_camera_at_start:
            unknown_camera_locations.append(prisoner_location)

        # append cameras at known hideouts
        # for i in self.hideout_list:
        #     if i.known_to_good_guys:
        #         known_camera_locations.append(i.location)

        # for gnn observations
        self._known_hideouts = np.zeros((self.num_known_hideouts, 2))
        i = 0
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                self._known_hideouts[i, :] = [hideout.location[0] / self.dim_x, hideout.location[1] / self.dim_y]
                i += 1
        self._known_hideouts = self._known_hideouts.flatten()

        # initialize these variables for observation spaces
        self.num_known_cameras = len(known_camera_locations)
        self.num_unknown_cameras = len(unknown_camera_locations)

        for counter in range(self.num_known_cameras):
            camera_location = known_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=True, 
                                           detection_object_type_coefficient=self.camera_range_factor))

        for counter in range(self.num_unknown_cameras):
            camera_location = unknown_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=False,
                                           detection_object_type_coefficient=self.camera_range_factor))

        # specify helicopters' initial locations
        # generate helicopter lists
        for _ in range(self.num_helicopters):
            if self.reward_setting != "collision":
                helicopter_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                self.init_heli_loc = [helicopter_location]
            else:
                helicopter_location = AbstractObject.generate_random_locations_with_range([prisoner_location[0]-self.blue_spawn_range, 
                                                                                           prisoner_location[0]+self.blue_spawn_range],
                                                                                          [prisoner_location[1]-self.blue_spawn_range, 
                                                                                           prisoner_location[1]+self.blue_spawn_range])
            self.helicopters_list.append(
                Helicopter(self.terrain, helicopter_location, speed=self.helicopter_speed, detection_object_type_coefficient=0.75))  # 100mph=127 grids/timestep

        # uncomment if you want to specify search parties' initial locations
        # search_party_initial_locations = [[1800, 2400], [1400, 1500]]

        # random
        # search_party_initial_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y),
        #                                   AbstractObject.generate_random_locations(self.dim_x, self.dim_y)]
        search_party_initial_locations = []
        for _ in range(self.num_search_parties):
            if self.blue_spawn_mode == "random":
                search_party_initial_locations.append(AbstractObject.generate_random_locations(self.dim_x, self.dim_y))
            elif self.blue_spawn_mode == "near_red":
                search_party_initial_locations.append(AbstractObject.generate_random_locations_with_range([prisoner_location[0]-self.blue_spawn_range, 
                                                                                                          prisoner_location[0]+self.blue_spawn_range],
                                                                                                          [prisoner_location[1]-self.blue_spawn_range, 
                                                                                                          prisoner_location[1]+self.blue_spawn_range]))
            else:
                raise NotImplementedError
                
        self.init_sp_loc = search_party_initial_locations

        # generate search party lists
        for counter in range(self.num_search_parties):
            search_party_location = search_party_initial_locations[
                counter]  # AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
            self.search_parties_list.append(SearchParty(self.terrain, search_party_location, speed=self.search_party_speed, detection_object_type_coefficient=0.75))  # speed=4

        self.map_div_num = 15
        self.local_range_edge_len = 450
        self.grid_edge_len = int(self.local_range_edge_len/self.map_div_num)
        self.agents_locs_map = np.zeros((self.map_div_num, self.map_div_num))
        self.probmap = (np.zeros((self.map_div_num, self.map_div_num)), np.array([0, 0]))
        self.reach_waypoint = False
        # Currently not modeling town
        # # generate town lists
        # for i in range(self.num_towns):
        #     # this is just trying to get to similar set-up as the figure in the pdf
        #     if i == 0:
        #         town_location = AbstractObject.generate_random_locations_with_range((300, 350), (600, 700))
        #     elif i == 1:
        #         town_location = AbstractObject.generate_random_locations_with_range((400, 450), (700, 800))
        #     elif i == 2:
        #         town_location = AbstractObject.generate_random_locations_with_range((330, 430), (70, 130))
        #     elif i == 3:
        #         town_location = AbstractObject.generate_random_locations_with_range((500, 600), (250, 350))
        #     else:
        #         town_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
        #     self.town_list.append(Town(self.terrain, town_location))

    @property
    def all_hideout_list(self):
        all_hideout_list = []
        for hideout_location in self.known_hideout_locations:
            all_hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=True))
        for hideout_location in self.unknown_hideout_locations:
            all_hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=False))
        return all_hideout_list

    @property
    def hideout_locations(self):
        return [hideout.location for hideout in self.hideout_list]

    @property
    def unknown_hideout_locations_on_map(self):
        return [np.array(hideout.location) for hideout in self.hideout_list if not hideout.known_to_good_guys]

    @property
    def known_hideout_locations_on_map(self):
        return [np.array(hideout.location) for hideout in self.hideout_list if hideout.known_to_good_guys]

    @property
    def closest_unknown_hideout_location(self):
        prisoner_location = np.array(self.get_prisoner_location())
        closest_loc = self.unknown_hideout_locations_on_map[0]
        dist_min = np.linalg.norm(closest_loc - prisoner_location)
        for unknown_hideout_location in self.unknown_hideout_locations_on_map:
            dist_temp = np.linalg.norm(unknown_hideout_location - prisoner_location)
            if dist_temp < dist_min:
                dist_min = dist_temp
                closest_loc = unknown_hideout_location
        return closest_loc

    @property
    def closest_known_hideout_location(self):
        prisoner_location = np.array(self.get_prisoner_location())
        closest_loc = self.known_hideout_locations_on_map[0]
        dist_min = np.linalg.norm(closest_loc - prisoner_location)
        for known_hideout_location in self.known_hideout_locations_on_map:
            dist_temp = np.linalg.norm(known_hideout_location - prisoner_location)
            if dist_temp < dist_min:
                dist_min = dist_temp
                closest_loc = known_hideout_location
        return closest_loc



    def get_state(self):
        """
        Compile a dictionary to represent environment's current state (only including things that will change in .step())
        :return: a dictionary with prisoner_location, search_party_locations, helicopter_locations, timestep, done, prisoner_location_history, is_detected
        """
        prisoner_location = self.prisoner.location.copy()
        search_party_locations = []
        for search_party in self.search_parties_list:
            search_party_locations.append(search_party.location.copy())
        helicopter_locations = []
        for helicopter in self.helicopters_list:
            helicopter_locations.append(helicopter.location.copy())
        timestep = self.timesteps
        done = self.done
        prisoner_location_history = self.prisoner_location_history.copy()
        is_detected = self.is_detected

        prediction_observation = self._prediction_observation.copy()
        fugitive_observation = self._fugitive_observation.copy()
        ground_truth_observation = self._ground_truth_observation.copy()
        blue_observation = self._blue_observation.copy()

        # print(self.search_parties_list[0].location)
        return {
            "prisoner_location": prisoner_location,
            "search_party_locations": search_party_locations,
            "helicopter_locations": helicopter_locations,
            "timestep": timestep,
            "done": done,
            "prisoner_location_history": prisoner_location_history,
            "is_detected": is_detected,
            # "blue_heuristic": copy.deepcopy(self.blue_heuristic),
            "prediction_observation": prediction_observation,
            "fugitive_observation": fugitive_observation,
            "ground_truth_observation": ground_truth_observation,
            "blue_observation": blue_observation,
            "done": self.done
        }

    def set_state(self, state_dict):
        """
        Set the state of the env by state_dict. Paired with `get_state`
        :param state_dict: a state dict returned by `get_state`
        """
        self.prisoner.location = state_dict["prisoner_location"].copy()
        for i, search_party in enumerate(self.search_parties_list):
            search_party.location = state_dict["search_party_locations"][i].copy()
        for i, helicopter in enumerate(self.helicopters_list):
            helicopter.location = state_dict["helicopter_locations"][i].copy()
        self.timesteps = state_dict["timestep"]
        self.done = state_dict["done"]
        self.prisoner_location_history = state_dict["prisoner_location_history"].copy()
        self.is_detected = state_dict["is_detected"]
        # self.blue_heuristic = state_dict["blue_heuristic"]

        # self.search_parties_list = self.blue_heuristic.search_parties
        # self.helicopters_list = self.blue_heuristic.helicopters

        # set previous observations
        self._prediction_observation = state_dict["prediction_observation"].copy()
        self._fugitive_observation = state_dict["fugitive_observation"].copy()
        self._ground_truth_observation = state_dict["ground_truth_observation"].copy()
        self._blue_observation = state_dict["blue_observation"].copy()
        self.done = state_dict["done"]
        gc.collect()
        # self.blue_heuristic.step(self.prisoner.location)

    def update_filtering_reward(self, filtering_loss):
        self.filter_reward = -filtering_loss - 6

    def update_comm(self, comm):
        self.comm = comm

    def step_both(self, red_action: np.ndarray, blue_action: np.ndarray, localized_trgt_gaussians=None):
        """
        The environment moves one timestep forward with the action chosen by the agent.
        :param red_action: an speed and direction vector for the red agent
        :param blue_action: currently a triple of [dx, dy, speed] where dx and dy is the vector
            pointing to where the agent should go
            this vector should have a norm of 1
            we can potentially take np.arctan2(dy, dx) to match action space of fugitive


        :return: observation, reward, done (boolean), info (dict)
        """
        # print("Before step", self.search_parties_list[0].location)
        if self.done:
            if self.step_reset:
                raise RuntimeError("Episode is done")
            else:
                observation = np.zeros(self.observation_space.shape)
                total_reward = 0
                return observation, total_reward, self.done, {}
        # assert self.action_space.contains(red_action), f"Actions should be in the action space, but got {red_action}"

        self.timesteps += 1
        old_prisoner_location = self.prisoner.location.copy()

        # move red agent
        direction = np.array([np.cos(red_action[1]), np.sin(red_action[1])])

        fugitive_speed = red_action[0]
        self.current_prisoner_speed = fugitive_speed
        self.current_prisoner_velocity = direction * fugitive_speed * (self.max_timesteps / self.dim_x)

        # prisoner_location = np.array(self.prisoner.location, dtype=np.float)
        # new_location = np.round(prisoner_location + direction * fugitive_speed)
        # new_location[0] = np.clip(new_location[0], 0, self.dim_x - 1)
        # new_location[1] = np.clip(new_location[1], 0, self.dim_y - 1)
        # new_location = new_location.astype(np.int)

        # # bump back from mountain
        # if self.terrain.world_representation[0, new_location[0], new_location[1]] == 1:
        #     new_location = np.array(old_prisoner_location)

        # # finish moving the prisoner
        # self.prisoner.location = new_location.tolist()
        # self.prisoner_location_history.append(self.prisoner.location.copy())

        self.prisoner.path_v3(direction=direction, speed=fugitive_speed)
        self.prisoner_location_history.append(self.prisoner.location.copy())

        # move blue agents
        for i, search_party in enumerate(self.search_parties_list):
            # getattr(search_party, command)(*args, **kwargs)
            direction = blue_action[i][0:2]
            speed = blue_action[i][2]
            search_party.path_v3(direction=direction, speed=speed)
        if self.is_helicopter_operating():
            for j, helicopter in enumerate(self.helicopters_list):
                # getattr(helicopter, command)(*args, **kwargs)
                direction = blue_action[i + j + 1][0:2]
                speed = blue_action[i + j + 1][2]
                helicopter.path_v3(direction=direction, speed=speed)

        if self.stopping_condition:
            # add stop condition if our speed is between 0 and 1
            if (0 <= red_action[0] < 1):
                self.done = True
        else:
            # stop if we are near hideout
            if self.near_hideout():
                if self.near_goal:
                    self.done = True
                else:
                    self.near_goal = True

        # game ends?
        if self.timesteps >= self.max_timesteps:
            self.done = True

        if self.waypoints is not None:
            if self.near_waypoint():
                self.reach_waypoint = True
            else:
                self.reach_waypoint = False

        
        # self.generate_localized_filter_output()
        # Construct observation from these
        parties_detection_of_fugitive, self._gnn_agent_obs = self._determine_blue_detection_of_red(fugitive_speed)
        # _, self._gnn_agent_obs = self._determine_passive_blue_detection_of_red(fugitive_speed)

        self.is_detected = self.is_fugitive_detected(parties_detection_of_fugitive)
        self.is_cam_detected = self.is_fugitive_detected_by_cam(parties_detection_of_fugitive, len(self.camera_list))
        self.is_hs_detected = self.is_fugitive_detected_by_hs(parties_detection_of_fugitive, len(self.camera_list))
        if self.is_detected and self.store_last_k_fugitive_detections:
            self.last_k_fugitive_detections.pop(0)  # Remove old detection
            self.last_k_fugitive_detections.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y])  # Append latest detection
            self.last_k_fugitive_detections_vel.pop(0)  # Remove old detection
            self.last_k_fugitive_detections_vel.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y, self.current_prisoner_velocity[0], self.current_prisoner_velocity[1]])  # Append latest detection            
        self.last_k_blue_obs_only_red_loc.pop(0)
        if self.is_detected:
            self.last_k_blue_obs_only_red_loc.append([self.prisoner.location[0] / self.dim_x, self.prisoner.location[1] / self.dim_y])
        else:
            self.last_k_blue_obs_only_red_loc.append([0, 0])
        # self.generate_nonlocalized_filter_output()

        self.gnn_sequence_array[0:-1] = self.gnn_sequence_array[1:]
        self.gnn_sequence_array[-1] = self._gnn_agent_obs
        active_parties_detection_of_fugitive = self._determine_active_blue_detection_of_red(fugitive_speed)
        fugitive_detection_of_parties = self._determine_red_detection_of_blue(fugitive_speed)
        self.is_detecting_hs = self.is_fugitive_detecting_hs(fugitive_detection_of_parties)
        # INFO: Generate the last k red detection of blue -- dim: [k, blue_agent_num, feature_num]
        for blue_idx, d_flag in enumerate(self.is_detecting_hs):
            if d_flag:
                time_feature = self.last_k_fugitive_detect_blue_posVel[:,blue_idx,:]
                shifted_time_time_feature = np.roll(time_feature, shift=-1, axis=0) # first to last
                shifted_time_time_feature[-1,:] = np.concatenate(([self.timesteps/self.max_timesteps], self.get_hs_locVels()[blue_idx]/self.dim_x))
                self.last_k_fugitive_detect_blue_posVel[:,blue_idx,:] = shifted_time_time_feature 

        self.last_k_red_blue_states = np.roll(self.last_k_red_blue_states, shift=-1, axis=0) # first to last
        self.last_k_red_blue_states[-1] = self.get_red_blue_state()

        self.lask_k_blue_detection_of_red = np.roll(self.lask_k_blue_detection_of_red, shift=-1, axis=0) # first to last
        self.lask_k_blue_detection_of_red[-1] = self.is_hs_detected

        self._detected_blue_states = self._construct_detected_blue_state(fugitive_detection_of_parties)
        self._fugitive_observation = self._construct_fugitive_observation(red_action, self._detected_blue_states[-1], self.terrain) # fugitive_detection_of_parties
        self._prediction_observation = self._construct_prediction_observation(red_action, fugitive_detection_of_parties, self._terrain_embedding)
        self._ground_truth_observation = self._construct_ground_truth(red_action, fugitive_detection_of_parties, parties_detection_of_fugitive, self._terrain_embedding)
        """Add prisoner location history"""
        # if self.prisoner_detected_loc_history != [-1, -1]:
        #     print("self.prisoner_detected_loc_history is: ", self.prisoner_detected_loc_history)
        # else:
        #     print("NO PRISONER DETECTED IN THIS EPISODE")
        """One prisoner detected history to observation"""
        parties_detection_of_fugitive_one_hot_og = transform_blue_detection_of_fugitive(parties_detection_of_fugitive, self.prisoner_detected_loc_history)
        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot_og, self._terrain_embedding, self.include_start_location_blue_obs)

        """Two prisoner detected history to observation"""
        parties_detection_of_fugitive_one_hot = transform_predicted_detection_of_fugitive(parties_detection_of_fugitive, self.predicted_locations_from_last_two_detections)
        # self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot, self._terrain_embedding, self.include_start_location_blue_obs)

        self._partial_blue_observation = self.__construct_partial_blue_observation(self.prisoner_detected_loc_history2)
        """Ego-centric observation"""
        blue_detect_idx = self.which_blue_detect(active_parties_detection_of_fugitive)
        self._modified_blue_observation = self._construct_each_blue_observation(self.predicted_locations_from_last_two_detections, blue_detect_idx, self._terrain_embedding, \
                    include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        
        self._modified_blue_observation_last_detections = self._construct_each_blue_observation_last_detections(self.get_last_two_detections_vel_time(), blue_detect_idx, self._terrain_embedding, \
                    include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)

        self._modified_blue_observation_no_detections = self._construct_each_blue_observation_no_detections(blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        # self._modified_blue_observation_no_detections_with_gaussians = self._construct_each_blue_observation_no_detections_with_gaussians(blue_detect_idx, self._terrain_embedding, \
        #     include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        self._modified_blue_observation_no_detections_with_actions = self._construct_each_blue_observation_no_detections_with_high_actions(blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        
        all_blue_detect_idx = self.which_blue_detect(parties_detection_of_fugitive)
        # self._modified_blue_observation_map = self._construct_each_blue_observation_probmap(all_blue_detect_idx)
    
        self.blue_obs_sequence_array[0:-1] = self.blue_obs_sequence_array[1:] # for storing past sequence of gnn observations
        self.blue_obs_sequence_array[-1] = np.array(self._modified_blue_observation_no_detections)

        # calculate reward
        if self.reward_setting == "rl":
            # INFO: rl reward
            reward = self.get_rl_reward()
        elif self.reward_setting == "piece":
            # INFO: piecewise reward
            reward = self.get_piece_reward()
        else:
            reward = 0

        if self.reach_waypoint:
            self.waypt_idx = self.waypt_idx + 1

        return self._fugitive_observation, self._blue_observation, reward, self.done, blue_detect_idx, self.is_hs_detected

    def set_dist_coeff(self, current_ep, dist_coeff_ep, minimum_coeff=0.0):
        # self.dist_coeff = 0
        self.dist_coeff = np.maximum(-1 / dist_coeff_ep * current_ep + 1, minimum_coeff)

    def step_partial_blue_obs(self, red_action: np.ndarray, blue_action: np.ndarray):
        """
        The environment moves one timestep forward with the action chosen by the agent.
        :param red_action: an speed and direction vector for the red agent
        :param blue_action: currently a triple of [dx, dy, speed] where dx and dy is the vector
            pointing to where the agent should go
            this vector should have a norm of 1
            we can potentially take np.arctan2(dy, dx) to match action space of fugitive
            

        :return: observation, reward, done (boolean), info (dict)
        """
        # print("Before step", self.search_parties_list[0].location)
        if self.done:
            if self.step_reset:
                raise RuntimeError("Episode is done")
            else:
                observation = np.zeros(self.observation_space.shape)
                total_reward = 0
                return observation, total_reward, self.done, {}
        assert self.action_space.contains(red_action), f"Actions should be in the action space, but got {red_action}"

        self.timesteps += 1
        old_prisoner_location = self.prisoner.location.copy()

        # move red agent
        direction = np.array([np.cos(red_action[1]), np.sin(red_action[1])])

        speed = red_action[0]
        
        self.current_prisoner_speed = speed
        self.current_prisoner_velocity = direction * speed * (self.max_timesteps / self.dim_x)

        prisoner_location = np.array(self.prisoner.location, dtype=np.float)
        new_location = np.round(prisoner_location + direction * speed)
        new_location[0] = np.clip(new_location[0], 0, self.dim_x - 1)
        new_location[1] = np.clip(new_location[1], 0, self.dim_y - 1)
        new_location = new_location.astype(np.int)

        # bump back from mountain
        if self.terrain.world_representation[0, new_location[0], new_location[1]] == 1:
            new_location = np.array(old_prisoner_location)

        # finish moving the prisoner
        self.prisoner.location = new_location.tolist()
        self.prisoner_location_history.append(self.prisoner.location.copy())

        parties_detection_of_fugitive, self._gnn_agent_obs = self._determine_blue_detection_of_red(speed)
        # _, self._gnn_agent_obs = self._determine_passive_blue_detection_of_red(speed)

        # move blue agents
        for i, search_party in enumerate(self.search_parties_list):
            # getattr(search_party, command)(*args, **kwargs)
            direction = blue_action[i][0:2]
            speed = blue_action[i][2]
            search_party.path_v3(direction=direction, speed=speed)
        if self.is_helicopter_operating():
            for j, helicopter in enumerate(self.helicopters_list):
                # getattr(helicopter, command)(*args, **kwargs)
                direction = blue_action[i+j+1][0:2]
                speed = blue_action[i+j+1][2]
                helicopter.path_v3(direction=direction, speed=speed)

        if self.stopping_condition: 
            # add stop condition if our speed is between 0 and 1
            if (0 <= red_action[0] < 1):
                self.done = True
        else:
            # stop if we are near hideout
            if self.near_hideout():
                self.done = True

        # game ends?
        if self.timesteps >= self.max_timesteps:
            self.done = True

        # Construct observation from these
        fugitive_detection_of_parties = self._determine_red_detection_of_blue(speed)
        self._detected_blue_states = self._construct_detected_blue_state(fugitive_detection_of_parties)
        self._fugitive_observation = self._construct_fugitive_observation(red_action, self._detected_blue_states[-1], self.terrain) # fugitive_detection_of_parties
        self._prediction_observation = self._construct_prediction_observation(red_action, fugitive_detection_of_parties, self._terrain_embedding)
        self._ground_truth_observation = self._construct_ground_truth(red_action, fugitive_detection_of_parties, parties_detection_of_fugitive, self._terrain_embedding)
        """Add prisoner location history"""
        # if self.prisoner_detected_loc_history != [-1, -1]:
        #     print("self.prisoner_detected_loc_history is: ", self.prisoner_detected_loc_history)
        # else:
        #     print("NO PRISONER DETECTED IN THIS EPISODE")
        parties_detection_of_fugitive_one_hot = transform_blue_detection_of_fugitive(parties_detection_of_fugitive, self.prisoner_detected_loc_history)
        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot, self._terrain_embedding, self.include_start_location_blue_obs)

        self._partial_blue_observation = self.__construct_partial_blue_observation(self.prisoner_detected_loc_history)
        # print("parties_detection_of_fugitive_one_hot = ", parties_detection_of_fugitive_one_hot)
        # print("The prisoner showed up at: ", self._blue_observation[-2:])

        # construct observation
        # if self.observation_type == ObservationType.Fugitive:
        #     observation = self._fugitive_observation
        # elif self.observation_type == ObservationType.GroundTruth:
        #     observation =  self._ground_truth_observation
        # elif self.observation_type == ObservationType.Blue:
        #     observation = self._blue_observation
        # elif self.observation_type == ObservationType.Prediction:
        #     observation = self._prediction_observation
        # else:
        #     raise ValueError("self.observation_type incorrect")

        # calculate reward
        self.is_detected = self.is_fugitive_detected(parties_detection_of_fugitive)
        self.is_cam_detected = self.is_fugitive_detected_by_cam(parties_detection_of_fugitive, len(self.camera_list))
        if self.is_detected and self.store_last_k_fugitive_detections:
            self.last_k_fugitive_detections.pop(0)  # Remove old detection
            self.last_k_fugitive_detections.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y])  # Append latest detection
            self.last_k_fugitive_detections_vel.pop(0)  # Remove old detection
            self.last_k_fugitive_detections_vel.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y, self.current_prisoner_velocity[0], self.current_prisoner_velocity[1]]) 
        total_reward = self.get_reward()

        # print("After step", self.search_parties_list[0].location)
        
        return self._fugitive_observation, self._blue_observation, self._partial_blue_observation, total_reward, self.done, {}


    def step(self, red_action, blue_action):
        red_obs, blue_obs, total_reward, done, empty = self.step_both(red_action, blue_action)
        return red_obs, total_reward, done, empty

    @property
    def hideout_locations(self):
        return [hideout.location for hideout in self.hideout_list]

    def is_helicopter_operating(self):
        """
        Determines whether the helicopter is operating right now
        :return: Boolean indicating whether the helicopter is operating
        """
        timestep = self.timesteps % (self.helicopter_recharge_time + self.helicopter_battery_life)
        if timestep < self.helicopter_battery_life:
            return True
        else:
            return False

    @property
    def spawn_point(self):
        return self.prisoner_location_history[0].copy()

    @staticmethod
    def is_fugitive_detected(parties_detection_of_fugitive):
        for e, i in enumerate(parties_detection_of_fugitive):
            if e % 3 == 0:
                if i == 1:
                    return True
        return False

    @staticmethod
    def is_fugitive_detected_by_cam(parties_detection_of_fugitive, camera_num):
        for e, i in enumerate(parties_detection_of_fugitive[0:3*camera_num]):
            if e % 3 == 0:
                if i == 1:
                    return True
        return False

    @staticmethod
    def is_fugitive_detected_by_hs(parties_detection_of_fugitive, camera_num):
        for e, i in enumerate(parties_detection_of_fugitive[3*camera_num:]):
            if e % 3 == 0:
                if i == 1:
                    return True
        return False
    
    @staticmethod
    def is_fugitive_detecting_hs(fugitive_detection_of_parties):
        detect_indicator = fugitive_detection_of_parties[:12][1::6]
        detect_indicator = (np.array(detect_indicator) == 1)
        return detect_indicator

    @staticmethod
    def is_fugitive_active_detected(parties_detection_of_fugitive):
        is_detected = False
        detect_num = 0
        for e, i in enumerate(parties_detection_of_fugitive):
            if e % 3 == 0:
                if i == 1:
                    is_detected = True
                    detect_num = detect_num + 1
        return is_detected, detect_num

    @staticmethod
    def which_blue_detect(parties_detection_of_fugitive):
        idx = 0
        blue_detect_idx = []
        for e, i in enumerate(parties_detection_of_fugitive):
            if e % 3 == 0:
                if i == 1:
                    blue_detect_idx.append(idx)
                idx = idx + 1
        return blue_detect_idx

    def get_rl_reward(self):
        hideout_dist_reward = []
        reward = self.reward_scheme.time
        # INFO: Prisoner receives a large reward when it reaches any hideout
        hideout = self.near_hideout(hideout_idx=None)
        if hideout is not None:
            reward = reward + 50.0
        # INFO: Prisoner receives a negative reward when it is detected by h&s
        if self.is_hs_detected:
            reward = reward - self.reward_scheme.known_detected
        # INFO: Prisoner receives a negative reward when it collides with mountain
        mountain = self.near_mountain()
        if mountain is not None:
            reward = reward - 1.0
        # INFO: Time out! Mission failed.
        if self.timesteps == self.max_timesteps:
            reward = reward - 50
        return reward

    def get_eval_score(self):
        hideout_dist_reward = []
        reward = 0
        # reward = self.reward_scheme.time
        # INFO: Prisoner receives a large reward when it reaches any hideout
        hideout = self.near_hideout(hideout_idx=None)
        if hideout is not None:
            reward = reward + 50.0
        # INFO: Prisoner receives a negative reward when it is detected by h&s
        if self.is_hs_detected:
            reward = reward - self.reward_scheme.known_detected
        return reward

    def get_piece_reward(self):
        reward = self.reward_scheme.time
        # INFO: Prisoner receives a large reward when it reaches any waypoint
        if not self.reach_waypoint:
            reward = reward - 2.5 * self.dist_coeff * np.linalg.norm(np.asarray(self.waypoints[self.waypt_idx]) - np.asarray(self.prisoner.location)) / self.dim_x
        else:
            reward = reward + 13 
            # self.waypt_idx = self.waypt_idx + 1
        # # INFO: Prisoner receives a negative reward when it is detected by h&s
        if self.is_hs_detected:
            reward = reward - self.reward_scheme.known_detected
        # # INFO: Prisoner receives positive reward according to their distance to h&s
        # for adv in self.helicopters_list+self.search_parties_list:
        #     reward += 0.0001 * np.sqrt(np.sum(np.square(np.array(self.prisoner.location) - np.array(adv.location))))
        # INFO: Prisoner receives a negative reward when it collides with mountain
        # mountain = self.near_mountain()
        # if mountain is not None:
        #     reward = reward - 1.0
        return reward
    
    """ This setting is useful in off-policy training"""
    def get_blue_reward(self, blue_detect_idx):
        blue_agent_num = len(self.search_parties_list) + len(self.helicopters_list)
        share_weight = 0 # 1/blue_agent_num
        idx = 0
        detect_reward_list = []
        dist_reward_list = []
        # TODO recode this so combinations of scenarios are possible per timestep
        # if self.timesteps == self.max_timesteps:
        #     return self.reward_scheme.timeout  # running out of time is bad for fugitive but good for blue team!
        # if active_detected:
        #     reward = reward + self.reward_scheme.detected * detect_num
        # reward = reward + self.reward_scheme.time
        
        for _, search_party in enumerate(self.search_parties_list):
            detect_reward = 0
            dist_reward = 0
            for detected_id in blue_detect_idx:
                if idx == detected_id:
                    detect_reward = detect_reward + self.reward_scheme.detected
                else:
                    detect_reward = detect_reward + self.reward_scheme.detected * share_weight
                    # detect_reward = detect_reward + self.reward_scheme.detected / share_weight
            dist = np.linalg.norm(np.array(search_party.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1

        for _, helicopter in enumerate(self.helicopters_list):
            detect_reward = 0
            dist_reward = 0
            for detected_id in blue_detect_idx:
                if idx == detected_id:
                    detect_reward = detect_reward + self.reward_scheme.detected
                else:
                    detect_reward = detect_reward + self.reward_scheme.detected * share_weight
                    # detect_reward = detect_reward + self.reward_scheme.detected / share_weight
            dist = np.linalg.norm(np.array(helicopter.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1

        return np.array(detect_reward_list), np.array(dist_reward_list)

    """ This setting is useful in off-policy training"""
    def get_only_reward(self, blue_detect_idx):
        blue_agent_num = len(self.search_parties_list) + len(self.helicopters_list)
        share_weight = blue_agent_num
        idx = 0
        detect_reward_list = []
        dist_reward_list = []
        # TODO recode this so combinations of scenarios are possible per timestep
        # if self.timesteps == self.max_timesteps:
        #     return self.reward_scheme.timeout  # running out of time is bad for fugitive but good for blue team!
        # if active_detected:
        #     reward = reward + self.reward_scheme.detected * detect_num
        # reward = reward + self.reward_scheme.time
        search_party_group_detect_flag = False
        for _, search_party in enumerate(self.search_parties_list):
            detect_reward = 0
            dist_reward = 0
            for detected_id in blue_detect_idx:
                if idx == detected_id & search_party_group_detect_flag == False:
                    search_party_group_detect_flag = True
                    detect_reward = detect_reward + self.reward_scheme.detected
                # else:
                #     detect_reward = detect_reward + self.reward_scheme.detected / share_weight
                    # detect_reward = detect_reward + self.reward_scheme.detected / share_weight
            dist = np.linalg.norm(np.array(search_party.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1

        helicopter_group_detect_flag = False
        for _, helicopter in enumerate(self.helicopters_list):
            detect_reward = 0
            dist_reward = 0
            for detected_id in blue_detect_idx:
                if idx == detected_id & helicopter_group_detect_flag == False:
                    helicopter_group_detect_flag == True
                    detect_reward = detect_reward + self.reward_scheme.detected
                # else:
                #     detect_reward = detect_reward + self.reward_scheme.detected / share_weight
                    # detect_reward = detect_reward + self.reward_scheme.detected / share_weight
            dist = np.linalg.norm(np.array(helicopter.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1

        return np.array(detect_reward_list), np.array(dist_reward_list)

    """ This setting is useful in off-policy training"""
    def get_shared_reward(self, blue_detect_idx):
        blue_agent_num = len(self.search_parties_list) + len(self.helicopters_list)
        share_weight = blue_agent_num
        # idx = 0
        detect_reward_list = []
        dist_reward_list = []
        # TODO recode this so combinations of scenarios are possible per timestep
        # if self.timesteps == self.max_timesteps:
        #     return self.reward_scheme.timeout  # running out of time is bad for fugitive but good for blue team!
        # if active_detected:
        #     reward = reward + self.reward_scheme.detected * detect_num
        # reward = reward + self.reward_scheme.time
        # search_party_group_detect_flag = False
        for _, search_party in enumerate(self.search_parties_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.arange(5), blue_detect_idx)):
            if len(blue_detect_idx) != 0:
                detect_reward = detect_reward + self.reward_scheme.detected
            dist = np.linalg.norm(np.array(search_party.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -2.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            # idx = idx + 1

        for _, helicopter in enumerate(self.helicopters_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.array([5]), blue_detect_idx)):
            if len(blue_detect_idx) != 0:
                detect_reward = detect_reward + self.reward_scheme.detected

            dist = np.linalg.norm(np.array(helicopter.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -2.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            # idx = idx + 1
        return np.array(detect_reward_list), np.array(dist_reward_list)

    """ This setting is useful in off-policy training"""
    def get_gaussian_shared_reward(self, localized_trgt_gaussians, range=0.05):
        dists = np.linalg.norm(localized_trgt_gaussians, axis=1)
        blue_reach_gaussian = any(dists < range)
        blue_agent_num = len(self.search_parties_list) + len(self.helicopters_list)
        share_weight = blue_agent_num
        idx = 0
        detect_reward_list = []
        dist_reward_list = []
        # TODO recode this so combinations of scenarios are possible per timestep
        # if self.timesteps == self.max_timesteps:
        #     return self.reward_scheme.timeout  # running out of time is bad for fugitive but good for blue team!
        # if active_detected:
        #     reward = reward + self.reward_scheme.detected * detect_num
        # reward = reward + self.reward_scheme.time
        # search_party_group_detect_flag = False
        for search_party_idx, search_party in enumerate(self.search_parties_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.arange(5), blue_detect_idx)):
            if blue_reach_gaussian:
                detect_reward = detect_reward + self.reward_scheme.detected
            dist = dists[idx]
            dist_reward = -1.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1

        for helicopter_idx, helicopter in enumerate(self.helicopters_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.array([5]), blue_detect_idx)):
            if blue_reach_gaussian:
                detect_reward = detect_reward + self.reward_scheme.detected

            dist = dists[idx]
            dist_reward = -1.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            idx = idx + 1
        return np.array(detect_reward_list), np.array(dist_reward_list)

    def get_group_reward(self, blue_detect_idx):
        blue_agent_num = len(self.search_parties_list) + len(self.helicopters_list)
        share_weight = blue_agent_num
        # idx = 0
        detect_reward_list = []
        dist_reward_list = []
        # TODO recode this so combinations of scenarios are possible per timestep
        # if self.timesteps == self.max_timesteps:
        #     return self.reward_scheme.timeout  # running out of time is bad for fugitive but good for blue team!
        # if active_detected:
        #     reward = reward + self.reward_scheme.detected * detect_num
        # reward = reward + self.reward_scheme.time
        # search_party_group_detect_flag = False
        for _, search_party in enumerate(self.search_parties_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.arange(5), blue_detect_idx)):
            if len(blue_detect_idx) != 0:
                detect_reward = detect_reward + self.reward_scheme.detected
            dist = np.linalg.norm(np.array(search_party.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            # idx = idx + 1

        for _, helicopter in enumerate(self.helicopters_list):
            detect_reward = 0
            dist_reward = 0
            # if any(np.isin(np.array([5]), blue_detect_idx)):
            if len(blue_detect_idx) != 0:
                detect_reward = detect_reward + self.reward_scheme.detected

            dist = np.linalg.norm(np.array(helicopter.location)/2428.0 - np.array(self.prisoner.location)/2428.0)
            dist_reward = -0.5 * dist # original: -0.5
            detect_reward_list.append(detect_reward)
            # detect_reward_list.append(detect_reward + self.filter_reward)
            dist_reward_list.append(dist_reward)
            # idx = idx + 1
        return np.ones_like(np.array(detect_reward_list))*np.array(detect_reward_list).mean(), np.ones_like(np.array(dist_reward_list))*np.array(dist_reward_list).mean()


    def near_hideout(self, hideout_idx=None):
        """If the prisoner is within range of a hideout, return it. Otherwise, return None."""
        if hideout_idx is None:
            for hideout in self.hideout_list:
                if ((np.asarray(hideout.location) - np.asarray(
                        self.prisoner.location)) ** 2).sum() ** .5 <= self.hideout_radius + 11:
                    # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                    return hideout
        else:
            hideout = self.hideout_list[hideout_idx]
            if ((np.asarray(hideout.location) - np.asarray(self.prisoner.location)) ** 2).sum() ** .5 <= self.hideout_radius + 11:
                # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                return hideout           
        return None

    def near_waypoint(self):
        """If the prisoner is within range of nextwaypoint, return true. Otherwise, return false."""
        if ((np.asarray(self.waypoints[self.waypt_idx]) - np.asarray(self.prisoner.location)) ** 2).sum() ** .5 <= 10:
            return True
        else:
            return False
    
    def near_mountain(self):
        """If the prisoner is within range of a hideout, return it. Otherwise, return None."""
        for mountain_loc in self.terrain.mountain_locations:
            mountain_loc = np.array((mountain_loc[1], mountain_loc[0]))
            if ((np.asarray(mountain_loc) - np.asarray(self.prisoner.location)) ** 2).sum() ** .5 <= 150:
                # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                return mountain_loc
        return None
    
    def near_border(self):
        left_margin = np.abs(self.prisoner.location[0] - 0)
        right_margin = np.abs(self.prisoner.location[0] - self.dim_x)
        up_margin = np.abs(self.prisoner.location[1] - self.dim_y)
        down_margin = np.abs(self.prisoner.location[1] - 0)
        if left_margin < 10 or right_margin < 10 or up_margin < 10 or down_margin < 10:
            return True
        else:
            return False

    def _determine_red_detection_of_blue(self, speed):
        
        def prisoner_state_vec():
            normalized_prisoner_loc = np.array(self.prisoner.location) / self.dim_x
            return np.concatenate(([0], normalized_prisoner_loc))
            
        # INFO: this version is with velocity of the dynamic object
        fugitive_detection_of_parties = []
        SPRINT_SPEED_THRESHOLD = 70
        for helicopter in self.helicopters_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, 0, -1, -1, -1, -1])
            else:
                detect_vec = self.prisoner.detect(helicopter.location, helicopter, self.timesteps / self.max_timesteps, self.prisoner.location)
                if detect_vec[-1] == -1:
                    fugitive_detection_of_parties.extend([0] + self.prisoner.detect(helicopter.location, helicopter, self.timesteps / self.max_timesteps, self.prisoner.location) + [-1, -1])
                else:
                    fugitive_detection_of_parties.extend([0] + self.prisoner.detect(helicopter.location, helicopter, self.timesteps / self.max_timesteps, self.prisoner.location) + (helicopter.step_dist_xy/self.dim_x).tolist())
        for search_party in self.search_parties_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([1, 0, -1, -1, -1, -1])
            else:
                detect_vec = self.prisoner.detect(search_party.location, search_party, self.timesteps / self.max_timesteps, self.prisoner.location)
                if detect_vec[-1] == -1:
                    fugitive_detection_of_parties.extend([1] + self.prisoner.detect(search_party.location, search_party, self.timesteps / self.max_timesteps, self.prisoner.location) + [-1, -1])
                else:
                    fugitive_detection_of_parties.extend([1] + self.prisoner.detect(search_party.location, search_party, self.timesteps / self.max_timesteps, self.prisoner.location) + (search_party.step_dist_xy/self.dim_x).tolist())
        for camera in self.camera_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([-1, 0, -1, -1])
            else:
                fugitive_detection_of_parties.extend([-1] + self.prisoner.detect(camera.location, camera, self.timesteps / self.max_timesteps, self.prisoner.location))
        return fugitive_detection_of_parties


    def construct_gnn_obs(self, detected, agent, last_two_time_stamped_detection):
        """Construct the observation for a single agent for the GNN."""
        obs = []
        detected_bool = detected[0]
        detected_location = detected[1:]
        # obs.append(detected_bool)
        obs.append(agent.location[0] / self.dim_x)
        obs.append(agent.location[1] / self.dim_y)

        # obs.extend(detected_location)
        obs.append(self.timesteps / self.max_timesteps)

        # Add last two time_stamped_detection 
        if self.gnn_agent_last_detect == "global":
            obs.extend(last_two_time_stamped_detection)
        elif self.gnn_agent_last_detect == "local":
            obs.extend(np.array(agent.last_two_detections).flatten())
        elif self.gnn_agent_last_detect == "no":
            pass
        else:
            raise NotImplementedError

        if type(agent) == Camera:
            obs.extend([1, 0, 0])
        elif type(agent) == Helicopter:
            obs.extend([0, 1, 0])
        elif type(agent) == SearchParty:
            obs.extend([0, 0, 1])
        else:
            raise ValueError("Unknown agent type")
        
        return obs

    def _determine_blue_detection_of_red(self, speed):
        """ Also build the GNN obs here to make it faster and consistent """ 

        last_two_time_stamped_detection = np.concatenate([self.last_k_fugitive_detections[-2], self.last_k_fugitive_detections[-1]])

        parties_detection_of_fugitive = []
        gnn_obs = []
        for camera in self.camera_list:
            detected = camera.detect(self.prisoner.location, speed, self.timesteps / self.max_timesteps)
            gnn_obs.append(self.construct_gnn_obs(detected, camera, last_two_time_stamped_detection))
            parties_detection_of_fugitive.extend(detected)
        for helicopter in self.helicopters_list:
            detected = helicopter.detect(self.prisoner.location, speed, self.timesteps / self.max_timesteps)
            gnn_obs.append(self.construct_gnn_obs(detected, helicopter, last_two_time_stamped_detection))
            parties_detection_of_fugitive.extend(detected)
        for search_party in self.search_parties_list:
            detected = search_party.detect(self.prisoner.location, speed, self.timesteps / self.max_timesteps)
            gnn_obs.append(self.construct_gnn_obs(detected, search_party, last_two_time_stamped_detection))
            parties_detection_of_fugitive.extend(detected)

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps
            self.prisoner_detected_loc_history = self.prisoner.location
            """Hold/update the prisoner location history when detected"""
            if self.prisoner_detected_loc_history2 == [-1, -1, -1, -1]:
                self.prisoner_detected_loc_history2[0:2] = self.prisoner.location
            else:
                self.prisoner_detected_loc_history2[2:4] = self.prisoner_detected_loc_history2[0:2]
                self.prisoner_detected_loc_history2[0:2] = self.prisoner.location

        gnn_obs = np.array(gnn_obs)
        return parties_detection_of_fugitive, gnn_obs # gnn_obs shape: [agent_num (camera, blue agent, etc.), feature_num]

    def _determine_passive_blue_detection_of_red(self, speed):
        """ Also build the GNN obs here to make it faster and consistent """ 

        last_two_time_stamped_detection = np.concatenate([self.last_k_fugitive_detections[-2], self.last_k_fugitive_detections[-1]])

        parties_detection_of_fugitive = []
        gnn_obs = []
        for camera in self.camera_list:
            detected = camera.detect(self.prisoner.location, speed, self.timesteps / self.max_timesteps)
            gnn_obs.append(self.construct_gnn_obs(detected, camera, last_two_time_stamped_detection))
            parties_detection_of_fugitive.extend(detected)

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps
            self.prisoner_detected_loc_history = self.prisoner.location
            """Hold/update the prisoner location history when detected"""
            if self.prisoner_detected_loc_history2 == [-1, -1, -1, -1]:
                self.prisoner_detected_loc_history2[0:2] = self.prisoner.location
            else:
                self.prisoner_detected_loc_history2[2:4] = self.prisoner_detected_loc_history2[0:2]
                self.prisoner_detected_loc_history2[0:2] = self.prisoner.location

        gnn_obs = np.array(gnn_obs)
        return parties_detection_of_fugitive, gnn_obs # gnn_obs shape: [agent_num (camera, blue agent, etc.), feature_num]

    def _determine_active_blue_detection_of_red(self, speed):
        parties_detection_of_fugitive = []
        for search_party in self.search_parties_list:
            parties_detection_of_fugitive.extend(search_party.detect(self.prisoner.location, speed, self.timesteps/self.max_timesteps))
        for helicopter in self.helicopters_list:
            parties_detection_of_fugitive.extend(helicopter.detect(self.prisoner.location, speed, self.timesteps/self.max_timesteps))
        return parties_detection_of_fugitive


    def _determine_detection(self, speed):
        fugitive_detection_of_parties = []
        SPRINT_SPEED_THRESHOLD = 8
        for helicopter in self.helicopters_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(helicopter.location, helicopter))
        for search_party in self.search_parties_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(search_party.location, search_party))
        parties_detection_of_fugitive = []
        for camera in self.camera_list:
            parties_detection_of_fugitive.extend(camera.detect(self.prisoner.location, speed))
        for helicopter in self.helicopters_list:
            parties_detection_of_fugitive.extend(helicopter.detect(self.prisoner.location, speed))
        for search_party in self.search_parties_list:
            parties_detection_of_fugitive.extend(search_party.detect(self.prisoner.location, speed))

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps

        return fugitive_detection_of_parties, parties_detection_of_fugitive

    def _construct_fugitive_observation(self, action, fugitive_detection_of_parties, terrain):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param action: the action taken by the fugitive at this timestep
        :param fugitive_detection_of_parties: a list encoding fugitive's detection of all parties
        :param terrain: the terrain feature of the environment
        :return: the observation from the prisoner perspective
        """

        # NOTE: When editing, make sure this corresponds to names/orders in obs_names, in constructor
        # Future: Assign these using obs_names instead? may be slower...
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                observation.append(camera.location[0] / self.dim_x)
                observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            observation.append(float(hideout.known_to_good_guys))
            observation.append((hideout.location[0]-self.prisoner.location[0]) / self.dim_x)
            observation.append((hideout.location[1]-self.prisoner.location[1]) / self.dim_y)
        for mountain_loc in self.terrain.mountain_locations:
            observation.append((mountain_loc[1]-self.prisoner.location[0]) / self.dim_x)
            observation.append((mountain_loc[0]-self.prisoner.location[1]) / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        # FIXME: Maybe too much noise here, delete the red detection of cameras?
        # observation.extend(fugitive_detection_of_parties)
        # observation.extend(self.predicted_relative_blue_locations_from_last_two_detections)
        observation.extend(np.concatenate(self.get_relative_hs_locVels())/self.dim_x)
        if self.waypoints is not None:
            observation.extend((np.array(self.waypoints[self.waypt_idx]) - np.array(self.prisoner.location)) / self.dim_x)
        observation = np.array(observation)
        observation = np.concatenate((observation, np.array([terrain.detection_coefficient_given_location(self.prisoner.location)])))
        return observation # shape = (120,), 120 = 1(timestep) + 44(known camera num) * 2 + 3(hideout num) * 3(known_to_good_guys+hideout loc) + 2(prisoner location) + 2(prisoner action) + 18(fugitive_detection_of_parties) + 0(terrain)

    def construct_diffusion_conditions(self, cond_on_hideout_num):

        batch_size = sum(cond_on_hideout_num)

        def normalize(arr):
            arr = copy.deepcopy(arr).astype(float)
            x = arr[..., 0]
            arr[..., 0] = ((x - 0) / (self.dim_x - 0)) * 2 - 1

            y = arr[..., 1]
            arr[..., 1] = ((y - 0) / (self.dim_y - 0)) * 2 - 1
            return arr

        normalized_hideouts = normalize(np.array(self.hideout_locations))
        global_cond_hideouts = torch.Tensor(normalized_hideouts).view(1, -1).repeat(batch_size, 1)
        normalized_red_start = normalize(np.array(self.prisoner.location))
        global_cond_red_start = torch.Tensor(normalized_red_start).view(1, -1).repeat(batch_size, 1)
        
        global_cond = {'hideouts': global_cond_hideouts, 'red_start': global_cond_red_start}

        local_cond0 = (np.array([[0], [-1]]), np.array([normalized_red_start, normalized_hideouts[0]]))
        local_cond1 = (np.array([[0], [-1]]), np.array([normalized_red_start, normalized_hideouts[1]]))
        local_cond2 = (np.array([[0], [-1]]), np.array([normalized_red_start, normalized_hideouts[2]]))

        local_cond = [local_cond0 for _ in range(cond_on_hideout_num[0])] + [local_cond1 for _ in range(cond_on_hideout_num[1])] + [local_cond2 for _ in range(cond_on_hideout_num[2])]
        # local_cond = [(np.array([]), np.array([])) for _ in range(batch_size)]
        return global_cond, local_cond
        


    def construct_gt_blue_state(self, freq):
        see_blue_flag = ((self.timesteps % freq) == 0)
        fugitive_detection_of_parties = []
        for helicopter in self.helicopters_list:
            if not see_blue_flag:
                # fugitive_detection_of_parties.extend([0, 0, -1, -1])
                fugitive_detection_of_parties.append(np.array([-1, -1]))
            else:
                # fugitive_detection_of_parties.extend([0, 1] + self.normalize_to_pm(helicopter.location).tolist() + (helicopter.step_dist_xy/self.dim_x).tolist()) # [type, if detected, loc_x, loc_y]
                fugitive_detection_of_parties.append(self.normalize_to_pm(helicopter.location))
        for search_party in self.search_parties_list:
            if not see_blue_flag:
                # fugitive_detection_of_parties.extend([1, 0, -1, -1])
                fugitive_detection_of_parties.append(np.array([-1, -1]))
            else:
                # fugitive_detection_of_parties.extend([1, 1] + self.normalize_to_pm(search_party.location).tolist() + (search_party.step_dist_xy/self.dim_x).tolist())
                fugitive_detection_of_parties.append(self.normalize_to_pm(search_party.location))
        return fugitive_detection_of_parties

    def _construct_detected_blue_state(self, fugitive_detection_of_parties, max_len=35):

        def fit_to_maxLen(blue_location_velocity):
            if len(blue_location_velocity) != 0:
                blue_state_vec = np.concatenate(([np.hstack(lv) for lv in blue_location_velocity]))
            else:
                blue_state_vec = np.array([])
            if blue_state_vec.shape[0] >= max_len:
                blue_state_vec = blue_state_vec[:max_len]
            else:
                blue_state_vec = np.pad(blue_state_vec, pad_width=(0,max_len-blue_state_vec.shape[0]), constant_values=-1)
            return blue_state_vec

        blue_location_velocity = []
        red_detect_blue_id = []
        current_element_pointer = 0
        current_blue_pointer = 0
        for helicopter in self.helicopters_list:
            detect_flag = fugitive_detection_of_parties[current_element_pointer+1]
            location = fugitive_detection_of_parties[current_element_pointer+2:current_element_pointer+4]
            velocity = fugitive_detection_of_parties[current_element_pointer+4:current_element_pointer+6]
            if detect_flag:
                blue_location_velocity.append([location, velocity, [helicopter.detection_ranges[0]/self.dim_x]])
                red_detect_blue_id.append(current_blue_pointer)
            else:
                blue_location_velocity.append([[-1, -1], [-1, -1], [-1]])
            current_element_pointer = current_element_pointer + 6
            current_blue_pointer = current_blue_pointer + 1
        for search_party in self.search_parties_list:
            detect_flag = fugitive_detection_of_parties[current_element_pointer+1]
            location = fugitive_detection_of_parties[current_element_pointer+2:current_element_pointer+4]
            velocity = fugitive_detection_of_parties[current_element_pointer+4:current_element_pointer+6]
            if detect_flag:
                blue_location_velocity.append([location, velocity, [search_party.detection_ranges[0]/self.dim_x]])
                red_detect_blue_id.append(current_blue_pointer)
            else:
                blue_location_velocity.append([[-1, -1], [-1, -1], [-1]])
            current_element_pointer = current_element_pointer + 6
            current_blue_pointer = current_blue_pointer + 1
        for cam_i, camera in enumerate(self.camera_list):
            if cam_i < self.num_unknown_cameras:
                detect_flag = fugitive_detection_of_parties[current_element_pointer+1]
                location = fugitive_detection_of_parties[current_element_pointer+2:current_element_pointer+4]
                velocity = [0, 0]
                if detect_flag:
                    blue_location_velocity.append([location, velocity, [camera.detection_ranges[0]/self.dim_x]])
                    red_detect_blue_id.append(current_blue_pointer)
            current_element_pointer = current_element_pointer + 4
            current_blue_pointer = current_blue_pointer + 1
        return red_detect_blue_id, blue_location_velocity, fit_to_maxLen(blue_location_velocity).tolist()

    def normalize_to_pm(self, arr):
        arr = np.array(arr).astype(float)

        x = arr[..., 0]
        arr[..., 0] = ((x - 0) / (self.dim_x - 0)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - 0) / (self.dim_y - 0)) * 2 - 1
        return arr

    def _construct_prediction_observation(self, action, fugitive_detection_of_parties, terrain):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param action: the action taken by the fugitive at this timestep
        :param fugitive_detection_of_parties: a list encoding fugitive's detection of all parties
        :param terrain: the terrain feature of the environment
        :return: the observation from the prisoner perspective
        """
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                observation.append(camera.location[0] / self.dim_x)
                observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                observation.append(hideout.location[0] / self.dim_x)
                observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        observation.extend(fugitive_detection_of_parties)
        observation = np.array(observation)
        observation = np.concatenate((observation, terrain))

        return observation 

    def _construct_ground_truth(self, action, fugitive_detection_of_parties, parties_detection_of_fugitive, terrain):
        """
        Construct ground truth state.
        """
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            observation.append(camera.location[0] / self.dim_x)
            observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            observation.append(float(hideout.known_to_good_guys))
            observation.append(hideout.location[0] / self.dim_x)
            observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)

        # We include here the observations of each of the parties (don't need the locations given from these)
        for i in range(0, len(fugitive_detection_of_parties), 4):
            observation.append(fugitive_detection_of_parties[i])

        for i in range(0, len(parties_detection_of_fugitive), 3):
            observation.append(parties_detection_of_fugitive[i])

        observation = np.array(observation)
        observation = np.concatenate((observation, terrain))
        return observation

    def _construct_blue_observation(self, parties_detection_of_fugitive_one_hot, terrain, include_start_location_blue_obs=False, include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """

        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            observation.append(camera.location[0] / self.dim_x)
            observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                observation.append(hideout.location[0] / self.dim_x)
                observation.append(hideout.location[1] / self.dim_y)
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)

        observation.extend(parties_detection_of_fugitive_one_hot)

        if include_start_location_blue_obs:
            observation.append(self.prisoner_start_location[0]/ self.dim_x)
            observation.append(self.prisoner_start_location[1] / self.dim_y)

        if include_fugitive_location_in_blue_obs:
            observation.append(self.prisoner.location[0] / self.dim_x)
            observation.append(self.prisoner.location[1] / self.dim_y)

        observation = np.array(observation)
        observation = np.concatenate((observation, terrain))

        return observation

    def get_linear_estimation_error(self):
        normalized_prisoner_gt_location = np.array([self.prisoner.location[0] / self.dim_x, self.prisoner.location[1] / self.dim_y])
        return np.linalg.norm(self.predicted_locations_from_last_two_detections - normalized_prisoner_gt_location)

    def _construct_each_blue_observation(self, predicted_locations_from_last_two_detections, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            for camera in self.camera_list:
                observation.append(camera.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(camera.location[1] / self.dim_y - normalized_ag_loc_y)
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                        observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                        observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)
            observation.append(predicted_locations_from_last_two_detections[0] - normalized_ag_loc_x)
            observation.append(predicted_locations_from_last_two_detections[1] - normalized_ag_loc_y)
            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations

    def _construct_each_blue_observation_probmap(self, blue_detect_idx):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        moving_blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        all_blue_agents = [agent for agent in (self.camera_list + self.search_parties_list + self.helicopters_list)]
        observations = []

        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        maddpg_blue_agents = self.maddpg_agents


        if maddpg_blue_agents == None:
            group_indices = np.array([7 for _ in range(len(blue_agents))]).astype("int")
        else:
            group_indices = np.array([int(np.nonzero(maddpg_blue_agents.agents[i].high_level_action)[0]) for i in range(len(blue_agents))]).astype("int")

        self.probmap = self.update_probmap(all_blue_agents, blue_detect_idx, group_indices)
        ag_vel_map = self.construct_vel_map(moving_blue_agents, group_indices)
        terrain_maps = self.construct_terrain_map()
        ag_loc_maps = self.construct_agent_map(blue_agents, group_indices)
        
        for blue_idx, ag in enumerate(blue_agents):
            group_idx = group_indices[blue_idx]
            prob_map = self.probmap[0][group_idx].reshape(self.map_div_num,self.map_div_num)
            terrain_map = terrain_maps[group_idx]
            detect_map = self.construct_detect_map(blue_detect_idx)
            observations.append(np.stack(((prob_map), ag_loc_maps[blue_idx], detect_map, ag_vel_map[0], ag_vel_map[1], self.normalize_to_zero_one(terrain_map))).reshape(-1))
        # print("prob_map_max: ", self.normalize_prob_map(prob_map).max())
        # print("prob_map_min: ", self.normalize_prob_map(prob_map).min())
        return observations

    def normalize_prob_map(self, prob_map):
        return (prob_map - np.mean(prob_map)) / (np.std(prob_map))

    def normalize_to_zero_one(self, vector):
        vector = (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
        return vector

    def construct_detect_map(self, blue_detect_idx):
        map = np.zeros((self.map_div_num, self.map_div_num))
        if len(blue_detect_idx) != 0:
            group_idx=7
            x_coord_grid, y_coord_grid = self.loc_to_grid(self.prisoner, group_idx)
            if self.grid_coord_in_range(x_coord_grid, y_coord_grid):
                map[x_coord_grid, y_coord_grid] = 1
        return map
    
    def get_region(self, x, y):
        grid = self.terrain.world_representation[1]
        local_range_edge_len = self.local_range_edge_len
        half_length = local_range_edge_len // 2
        region = np.full((local_range_edge_len, local_range_edge_len), -1.)  # Initialize the region with -1 values
        x_min = max(x - half_length, 0)  # Compute the minimum x index for the region
        x_max = min(x + half_length, grid.shape[0] - 1)  # Compute the maximum x index for the region
        y_min = max(y - half_length, 0)  # Compute the minimum y index for the region
        y_max = min(y + half_length, grid.shape[1] - 1)  # Compute the maximum y index for the region

        grid_region = grid[x_min:x_max, y_min:y_max]

        # Copy the valid region of the grid into the output region array
        region[(half_length - (x - x_min)):(half_length + (x_max - x)), (half_length - (y - y_min)):(half_length + (y_max - y))] = grid_region
        return region

    def downsize_grid(self, grid):
        cell_size = self.grid_edge_len
        new_shape = self.map_div_num
        g = grid.reshape((new_shape, cell_size, new_shape, cell_size))
        g = np.mean(g, axis=(1, 3)) # np.mean produces some artifacts at the edges where there is -1
        return g

    def construct_terrain_map(self):
        pi, mu, sigma = self.nonlocalized_trgt_gaussians
        gaussian_num = pi.shape[-1]
        terrain_map_gaussians = []
        for i in range(gaussian_num):
            mu_center = mu[0][i]
            region = self.get_region(int(mu_center[0]), int(mu_center[1]))
            region = (region - 0.2) / 0.6 # Normalize the grid (original is between 0.2 and 0.8)
            terrain_map = self.downsize_grid(region)
            terrain_map_gaussians.append(terrain_map)
        return terrain_map_gaussians

    def construct_agent_map(self, blue_agents, group_indices):
        map = np.zeros((len(blue_agents), self.map_div_num, self.map_div_num))
        for curr_ag_idx, (curr_ag, curr_group_idx) in enumerate(zip(blue_agents, group_indices)):
            for ag_i, ag in enumerate((blue_agents)):
                normalized_ag_loc_x = ag.location[0] / self.dim_x
                normalized_ag_loc_y = ag.location[1] / self.dim_y
                normalized_ag_loc = (normalized_ag_loc_x, normalized_ag_loc_y)
                x_coord_grid, y_coord_grid = self.loc_to_grid(ag, curr_group_idx)
                if self.grid_coord_in_range(x_coord_grid, y_coord_grid):
                    if ag_i == curr_ag_idx:
                        map[curr_ag_idx, x_coord_grid, y_coord_grid] = 2
                    else:
                        map[curr_ag_idx, x_coord_grid, y_coord_grid] = 1
        return map

    def construct_vel_map(self, blue_agents, group_indices):
        map = np.zeros((2, self.map_div_num, self.map_div_num))
        for ag_i, (ag, group_idx) in enumerate(zip(blue_agents, group_indices)):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            normalized_ag_loc = (normalized_ag_loc_x, normalized_ag_loc_y)
            x_coord_grid, y_coord_grid = self.loc_to_grid(ag, group_idx)
            if self.grid_coord_in_range(x_coord_grid, y_coord_grid):
                map[0, x_coord_grid, y_coord_grid] = ag.normalized_vel[0]
                map[1, x_coord_grid, y_coord_grid] = ag.normalized_vel[1]
        return map

    def grid_coord_in_range(self, x_coord_grid, y_coord_grid):
        if x_coord_grid < self.map_div_num and y_coord_grid < self.map_div_num and x_coord_grid >= 0 and y_coord_grid >= 0:
            return True
        else:
            return False

    def loc_to_grid(self, ag, group_idx):
        pi, mu, sigma = self.nonlocalized_trgt_gaussians
        x_coord_grid = int((ag.location[0] - mu[0][group_idx][0]*self.dim_x + self.local_range_edge_len / 2) / (self.grid_edge_len))
        y_coord_grid = int((ag.location[1] - mu[0][group_idx][1]*self.dim_y + self.local_range_edge_len / 2) / self.grid_edge_len)
        return (x_coord_grid, y_coord_grid)

    def init_probmap(self):
        pi, mu, sigma = self.nonlocalized_trgt_gaussians
        gaussian_num = pi.shape[-1]
        prob_map_gaussians = []
        for i in range(gaussian_num):
            x = np.linspace(mu[0][i][0]*self.dim_x-self.local_range_edge_len/2,mu[0][i][0]*self.dim_x+self.local_range_edge_len/2, self.map_div_num)
            y = np.linspace(mu[0][i][1]*self.dim_y-self.local_range_edge_len/2,mu[0][i][1]*self.dim_y+self.local_range_edge_len/2, self.map_div_num)
            xv,yv = np.meshgrid(x,y)
            target_coord = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=-1)
            prob_map_gaussian = self.mdn_likelihood(pi[0][i:i+1], mu[0][i:i+1], sigma[0][i:i+1], target_coord/self.dim_x)
            prob_map_gaussians.append(prob_map_gaussian)
        return (prob_map_gaussians, mu)


    def update_probmap(self, blue_agents, blue_detect_idx, group_indices):
        prev_probmap, old_mu = copy.deepcopy(self.probmap)
        curr_probmap_before_obs, new_mu = self.init_probmap()

        dists = np.linalg.norm((new_mu[0] - old_mu[0]) * self.dim_x, axis=-1) # TODO: right now mu's may be ordered differently
        dists[dists >= 30] = 30 # set all the gaussians that jumped more than 30 cells to max dist
        alpha = np.clip(0.9 - np.maximum((dists - 15), 0) / 30, 0, 1) # When dists are close, alpha is high, incorporate more of the old map

        probmap_updated = [alpha[i] * prev_probmap[i] + (1 - alpha[i]) * curr_probmap_before_obs[i] for i in range(len(prev_probmap))]

        for i in range(len(blue_agents)):
            for group_idx in range(8):
                blue_agent = blue_agents[i]
                x_coord_grid, y_coord_grid = self.loc_to_grid(blue_agent, group_idx)
                if self.grid_coord_in_range(x_coord_grid, y_coord_grid):
                    self.update_without_detect(group_idx, x_coord_grid, y_coord_grid, blue_agent, probmap_updated)

        if len(blue_detect_idx) != 0:
            for group_idx in range(8):
                x_coord_grid, y_coord_grid = self.loc_to_grid(self.prisoner, group_idx)
                if self.grid_coord_in_range(x_coord_grid, y_coord_grid):
                    self.update_with_detect(group_idx, x_coord_grid, y_coord_grid, probmap_updated)

        # probmap = [probmap_updated[i]/np.sum(probmap_updated[i]) for i in range(len(probmap_updated))]
        probmap = probmap_updated
        self.probmap = (probmap, new_mu)

        return self.probmap
    
    def update_with_detect(self, group_idx, x_coord_grid, y_coord_grid, prob_map_gaussians):
        prob_map_gaussians[group_idx][x_coord_grid*self.map_div_num+y_coord_grid] = 0.99

    def update_without_detect(self, group_idx, x_coord_grid, y_coord_grid, agent, prob_map_gaussians):
        """ Base this number on the terrain at this location """
        grid = prob_map_gaussians[group_idx].reshape((15, 15))
        inner_range = agent.base_100_pod_distance(self.fugitive_speed_limit) # 15 is default fugitive speed
        outer_range = inner_range * 3

        # Calculate the distance of each cell in the map to center
        x, y = np.indices(grid.shape)
        distances = np.sqrt((x - x_coord_grid)**2 + (y - y_coord_grid)**2) * 30

        grid[distances <= outer_range] = np.clip(grid[distances <= outer_range], 0, 0.05)
        grid[distances <= inner_range] = np.clip(grid[distances <= inner_range], 0, 0.01)

        prob_map_gaussians[group_idx] = grid.reshape(225)

    def set_filter(self, filter_model):
        self.filter_model = filter_model

    def set_rl_algorithm(self, rl_alg_obj):
        self.maddpg_agents = rl_alg_obj

    def mdn_negative_log_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training 
        
        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        sigma = np.maximum(sigma, 0.005)
        prob_density_target = np.prod((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(target - mu)**2 / (2 * sigma**2)), axis=-1)
        prob_density_sum = np.sum(prob_density_target)
        ln_prob_grid = np.log((prob_density_target) / (prob_density_sum - prob_density_target))
        # print(np.min(prob_density_sum - prob_density_target))
        # logprob = -np.log(sigma) - (math.log(2 * math.pi) / 2) - \
        #     ((target - mu) / sigma)**2 / 2
        
        # inner = np.log(pi) + np.sum(logprob, axis=-1) # Sum the log probabilities of (x, y) for each 2D Gaussian
        return ln_prob_grid

    def mdn_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training 
        
        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        sigma = np.maximum(sigma, 0.02) # 0.02
        prob_density_target = np.prod((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(target - mu)**2 / (2 * sigma**2)), axis=-1)
        prob_density_sum = np.sum(prob_density_target)
        ln_prob_grid = prob_density_target / prob_density_sum

        return ln_prob_grid

    def generate_nonlocalized_filter_output(self):
        # PMC
        high_last_two_detections_vel_register = self.get_t_init_le_timeInterval()
        mlp_input_tensors_register = torch.Tensor(high_last_two_detections_vel_register).unsqueeze(0).to("cuda")
        nonlocalized_trgt_gaussians = self.filter_model(*self.split_filtering_input(mlp_input_tensors_register))
        # MLP
        # high_last_two_detections_vel_register = self.get_mlp_quasi_sel_input()
        # mlp_input_tensors_register = torch.Tensor(high_last_two_detections_vel_register).unsqueeze(0).to("cuda")
        # nonlocalized_trgt_gaussians = self.filter_model(mlp_input_tensors_register)       
        # PMC
        # high_last_two_detections_vel_register = self.get_new_t_init_lastDetection_timeInterval()
        # mlp_input_tensors_register = torch.Tensor(high_last_two_detections_vel_register).unsqueeze(0).to("cuda")
        # prior_input, dynamic_input, sel_input = self.split_new_pmc_input(mlp_input_tensors_register)
        # nonlocalized_trgt_gaussians = self.filter_model(prior_input, dynamic_input, sel_input)
        pi, mu, sigma = sort_filtering_output(nonlocalized_trgt_gaussians)
        self.nonlocalized_trgt_gaussians = (pi.detach().cpu().numpy(), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy())
        return self.nonlocalized_trgt_gaussians

    def generate_localized_filter_output(self, agent_actions_high):
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        localized_trgt_gaussian_locations = []
        localized_trgt_gaussians = []
        for ag, ac in zip(blue_agents, agent_actions_high):
            sorted_localized_filtering_out = localize_filtering_mu(self.nonlocalized_trgt_gaussians, np.array([ag.location])/self.dim_x)
            pi, mu, sigma = sorted_localized_filtering_out
            group_idx = np.nonzero(ac)[0]
            trgt_gaussian_pi = pi[0,group_idx]
            trgt_gaussian_mu = mu[0,group_idx,:]
            trgt_gaussian_sigma = sigma[0,group_idx,:]
            trgt_gaussian = np.concatenate((np.expand_dims(trgt_gaussian_pi, axis=1), trgt_gaussian_mu, trgt_gaussian_sigma), axis=-1)
            localized_trgt_gaussian_locations.append(trgt_gaussian_mu)
            localized_trgt_gaussians.append(trgt_gaussian)
        return localized_trgt_gaussian_locations, localized_trgt_gaussians

    def split_filtering_input(self, filtering_input):
        prior_input = filtering_input[..., 0:3]
        dynamic_input = filtering_input[..., 3:]
        sel_input = filtering_input
        return [prior_input, dynamic_input, sel_input]

    def _construct_each_blue_observation_last_detections(self, last_two_detects, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                        observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                        observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)
            # last two detections
            # for detect in last_two_detects:
            #     observation.append(detect[0])
            #     observation.append(detect[1] - normalized_ag_loc_x)
            #     observation.append(detect[2] - normalized_ag_loc_y)

            # last two detections with velocity
            observation.append(last_two_detects[1])
            observation.append(last_two_detects[2]-normalized_ag_loc_x)
            observation.append(last_two_detects[3]-normalized_ag_loc_y)
            observation.extend(last_two_detects[4:6])

            observation.append(last_two_detects[6])
            observation.append(last_two_detects[7]-normalized_ag_loc_x)
            observation.append(last_two_detects[8]-normalized_ag_loc_y)
            observation.extend(last_two_detects[9:11])

            # normalized_last_two_detects = [last_two_detects]
            # observation.extend(last_k_detects)
            # observation.append(predicted_locations_from_last_two_detections[0] - normalized_ag_loc_x)
            # observation.append(predicted_locations_from_last_two_detections[1] - normalized_ag_loc_y)
            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations


    def _construct_each_blue_observation_no_detections_with_gaussians(self, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        pi, mu, sigma = self.nonlocalized_trgt_gaussians
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            localized_gaussians = np.concatenate([pi.reshape(-1), (mu.squeeze() - np.array([[normalized_ag_loc_x, normalized_ag_loc_y]])).reshape(-1), sigma.reshape(-1)], axis=-1)
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            # for camera in self.camera_list:
            #     observation.append(camera.location[0] / self.dim_x - normalized_ag_loc_x)
            #     observation.append(camera.location[1] / self.dim_y - normalized_ag_loc_y)
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                    observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            # for blue_idx, other_ag in enumerate(blue_agents):
            #     if other_ag is not ag:
            #         observation.extend(self.comm[blue_idx])
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)

            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observation = np.concatenate((observation, localized_gaussians))
            observations.append(observation)

        return observations

    def _construct_each_blue_observation_no_detections(self, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                    observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)

            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations

    def _construct_each_blue_observation_no_detections_with_high_actions(self, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        maddpg_blue_agents = self.maddpg_agents
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                    observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)

            # Start to append the high-level actions
            if maddpg_blue_agents == None:
                for ag in blue_agents:
                     observation.extend(np.array([0,0,0,0,0,0,0,1]))
            else:
                observation.extend(maddpg_blue_agents.agents[blue_idx].high_level_action)
                for other_ag in blue_agents:
                    if other_ag is not ag:
                        observation.extend(maddpg_blue_agents.agents[blue_idx].high_level_action)

            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations


    def _construct_each_blue_observation_no_detections_high(self, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            # for camera in self.camera_list:
            #     observation.append(camera.location[0] / self.dim_x - normalized_ag_loc_x)
            #     observation.append(camera.location[1] / self.dim_y - normalized_ag_loc_y)
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                    observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            # for blue_idx, other_ag in enumerate(blue_agents):
            #     if other_ag is not ag:
            #         observation.extend(self.comm[blue_idx])
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)

            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations

    def _construct_each_blue_observation_no_detections_low(self, blue_detect_idx, terrain, include_start_location_blue_obs=False, 
            include_fugitive_location_in_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """
        observations = []
        blue_agents = [agent for agent in (self.search_parties_list + self.helicopters_list)]
        for blue_idx, ag in enumerate(blue_agents):
            normalized_ag_loc_x = ag.location[0] / self.dim_x
            normalized_ag_loc_y = ag.location[1] / self.dim_y
            observation = [self.timesteps / self.max_timesteps, normalized_ag_loc_x, normalized_ag_loc_y]
            # for camera in self.camera_list:
            #     observation.append(camera.location[0] / self.dim_x - normalized_ag_loc_x)
            #     observation.append(camera.location[1] / self.dim_y - normalized_ag_loc_y)
            for hideout in self.hideout_list:
                if hideout.known_to_good_guys:
                    observation.append(hideout.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(hideout.location[1] / self.dim_y - normalized_ag_loc_y)
            for other_ag in blue_agents:
                if other_ag is not ag:
                    observation.append(other_ag.location[0] / self.dim_x - normalized_ag_loc_x)
                    observation.append(other_ag.location[1] / self.dim_y - normalized_ag_loc_y)
            # for blue_idx, other_ag in enumerate(blue_agents):
            #     if other_ag is not ag:
            #         observation.extend(self.comm[blue_idx])
            detection_flag = 1 if blue_idx in blue_detect_idx else -1
            observation.append(detection_flag)

            if include_start_location_blue_obs:
                observation.append(self.prisoner_start_location[0]/ self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner_start_location[1] / self.dim_y - normalized_ag_loc_y)

            if include_fugitive_location_in_blue_obs:
                observation.append(self.prisoner.location[0] / self.dim_x - normalized_ag_loc_x)
                observation.append(self.prisoner.location[1] / self.dim_y - normalized_ag_loc_y)

            observation = np.array(observation)
            observation = np.concatenate((observation, terrain))
            observations.append(observation)

        return observations

    def __construct_partial_blue_observation(self, prisoner_loc_xy):

        observation = []
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)
        # if prisoner_loc_xy == [-1, -1, -1, -1]:
        #     observation = observation + [-1, -1, -1, -1]
        # if prisoner_loc_xy[2:4] == [0, 0]:
        #     observation = observation + [loc/2428.0 for loc in prisoner_loc_xy[0:2]] + [0, 0]
        # else:
        observation = observation + [(loc if loc==-1 else loc/2428.0) for loc in prisoner_loc_xy]

        observation = np.array(observation)

        return observation

    def cell_to_obs(self, cell):
        """
        Map a grid cell to the coordinates emitted in observations
        :param cell: integer sequence of length 2 within the range [(0, 0), (dim_x, dim_y))
        :return: np.ndarray of shape (2,) in the range [0, 1) of type np.float32
        """
        return np.array([cell[0] / self.dim_x, cell[1] / self.dim_y], dtype=np.float32)

    def obs_to_cell(self, coord):
        """
        Map a float coordinate in the observation space to the grid cell it most closely represents
        :param coord: float sequence of length 2 in the range [0, 1)
        :return: np.ndarray of shape (2,) in the range [(0, 0), (dim_x, dim_y))
        """
        return np.array([coord[0] * self.dim_x, coord[1] * self.dim_y], dtype=np.int)

    def set_seed(self, seed):
        """
        Set the seed for the random number generator.
        :param seed: the seed for the random number generator
        """
        np.random.seed(seed)
        random.seed(seed)

    def reset_env(self, seed=None):
        """
        Reset the environment. Should be called whenever done==True
        :return: observation
        """
        self.prisoner_detected_loc_history = [0, 0]
        self.prisoner_detected_loc_history2 = [-1, -1, -1, -1]
        self.nonlocalized_trgt_gaussians = None
        


        # self.predicted_locations_from_last_two_detections = [[-1, -1]]
        if seed is not None:
            self.set_seed(seed)

        self.set_terrain_paramaters()
        self.prisoner = Fugitive(self.terrain, [2400, 2400], fugitive_speed_limit=self.fugitive_speed_limit)  # the actual spawning will happen in set_up_world
        # Randomize the terrain

        self.timesteps = 0
        self.last_detected_timestep = 0
        self.near_goal = False
        self.done = False
        self.is_detected = False
        self.is_cam_detected = False

        self.set_up_world()
        init_detection = [0] + (np.array(self.prisoner_start_location) / 2428).tolist()
        init_detection_vel = init_detection + [0, 0]

        self.last_k_blue_obs_only_red_loc = [[0, 0] for _ in range(15)] + [(np.array(self.prisoner_start_location) / 2428).tolist()]

        if not self.include_start_location_blue_obs:
            self.last_k_fugitive_detections_vel = [[-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0],
                                           [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0]]  
            self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                            [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        else:
            self.last_k_fugitive_detections_vel = [[-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0],
                                            [-1, -1, -1, 0, 0], [-1, -1, -1, 0, 0], init_detection_vel, init_detection_vel]
            self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                            [-1, -1, -1], [-1, -1, -1], init_detection, init_detection]

        # INFO: Initialize the last k red detection of blue
        self.last_k_fugitive_detect_blue_posVel = np.zeros((8, self.num_helicopters+self.num_search_parties, 5))

        self.last_k_red_blue_states = np.zeros((16, 12))
        self.last_k_red_blue_states = np.roll(self.last_k_red_blue_states, shift=-1, axis=0) # first to last
        self.last_k_red_blue_states[-1] = self.get_red_blue_state()

        self.lask_k_blue_detection_of_red = np.zeros((16, 1))
        self.lask_k_blue_detection_of_red = np.roll(self.lask_k_blue_detection_of_red, shift=-1, axis=0) # first to last
        self.lask_k_blue_detection_of_red[-1] = 0

        self.total_agents_num = self.num_known_cameras + self.num_unknown_cameras + self.num_helicopters + self.num_search_parties
        self.passive_blue_agents_num = self.num_known_cameras + self.num_unknown_cameras
        self.current_prisoner_velocity = np.zeros(2)
        parties_detection_of_fugitive, self._gnn_agent_obs = self._determine_blue_detection_of_red(0.0)
        

        # fugitive_detection_of_parties, parties_detection_of_fugitive = self._determine_detection(0.0)

        
        # _, self._gnn_agent_obs = self._determine_passive_blue_detection_of_red(0.0)
        

        self.gnn_sequence_array = np.zeros((16,) + self._gnn_agent_obs.shape) #for storing past sequence of gnn observations
        self.gnn_sequence_array[-1] = self._gnn_agent_obs
        self.prisoner_location_history = [self.prisoner.location.copy()]

        # INFO: piecewise RL
        self.waypoints = None
        self.waypt_idx = 1

        return
        


    def reset_obs(self, seed=None):
        # self.generate_nonlocalized_filter_output()
        # self.generate_localized_filter_output()
        # self.probmap = self.init_probmap()
        parties_detection_of_fugitive, self._gnn_agent_obs = self._determine_blue_detection_of_red(0.0)
        fugitive_detection_of_parties = self._determine_red_detection_of_blue(0.0)
        self._detected_blue_states = self._construct_detected_blue_state(fugitive_detection_of_parties)
        self._fugitive_observation = self._construct_fugitive_observation([0.0, 0.0], self._detected_blue_states[-1], self.terrain) # fugitive_detection_of_parties
        
        self._prediction_observation = self._construct_prediction_observation([0.0, 0.0], fugitive_detection_of_parties, self._terrain_embedding)
        """Add prisoner location history"""
        self._ground_truth_observation = self._construct_ground_truth([0.0, 0.0], fugitive_detection_of_parties, parties_detection_of_fugitive, self._terrain_embedding)
        
        """One prisoner detected history to observation"""
        parties_detection_of_fugitive_one_hot_og = transform_blue_detection_of_fugitive(parties_detection_of_fugitive, self.prisoner_detected_loc_history)
        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot_og, self._terrain_embedding, self.include_start_location_blue_obs, self.include_fugitive_location_in_blue_obs)
        
        """Two prisoner detected history to observation"""
        parties_detection_of_fugitive = transform_predicted_detection_of_fugitive(parties_detection_of_fugitive, self.predicted_locations_from_last_two_detections)
        active_parties_detection_of_fugitive = self._determine_active_blue_detection_of_red(speed=0.0)
        # 
        """Ego-centric observation"""
        blue_detect_idx = self.which_blue_detect(active_parties_detection_of_fugitive)
        self._modified_blue_observation = self._construct_each_blue_observation(self.predicted_locations_from_last_two_detections, blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        self._partial_blue_observation = self.__construct_partial_blue_observation(self.prisoner_detected_loc_history2)

        self._modified_blue_observation_last_detections = self._construct_each_blue_observation_last_detections(self.get_last_two_detections_vel_time(), blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)

        self._modified_blue_observation_no_detections = self._construct_each_blue_observation_no_detections(blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        # self._modified_blue_observation_no_detections_with_gaussians = self._construct_each_blue_observation_no_detections_with_gaussians(blue_detect_idx, self._terrain_embedding, \
        #     include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        self._modified_blue_observation_no_detections_with_actions = self._construct_each_blue_observation_no_detections_with_high_actions(blue_detect_idx, self._terrain_embedding, \
            include_start_location_blue_obs=self.include_start_location_blue_obs, include_fugitive_location_in_blue_obs=self.include_fugitive_location_in_blue_obs)
        
        all_blue_detect_idx = self.which_blue_detect(parties_detection_of_fugitive)
        # self._modified_blue_observation_map = self._construct_each_blue_observation_probmap(all_blue_detect_idx)
        self.blue_obs_sequence_array = np.zeros((4,) + np.array(self._modified_blue_observation_no_detections).shape) #for storing past sequence of gnn observations
        self.blue_obs_sequence_array[-1] = np.array(self._modified_blue_observation_no_detections)
        assert self._blue_observation.shape == self.blue_observation_space.shape, "Wrong observation shape %s, %s" % (self._blue_observation.shape, self.blue_observation_space.shape)
        assert self._partial_blue_observation.shape == self.blue_partial_observation_space.shape, "Wrong observation shape %s, %s" % (self._partial_blue_observation.shape, self.blue_partial_observation_space.shape)
        # assert self._ground_truth_observation.shape == self.gt_observation_space.shape, "Wrong observation shape %s, %s"  % (self._ground_truth_observation.shape, self.gt_observation_space.shape)
        # assert self._fugitive_observation.shape == self.fugitive_observation_space.shape, "Wrong observation shape %s, %s" % (self._fugitive_observation.shape, self.fugitive_observation_space.shape)
        # assert self._prediction_observation.shape == self.prediction_observation_space.shape, "Wrong observation shape %s, %s" % (self._prediction_observation.shape, self.prediction_observation_space.shape)


        # # construct observation
        # if self.observation_type == ObservationType.Fugitive:
        #     return self._fugitive_observation
        # elif self.observation_type == ObservationType.GroundTruth:
        #     return self._ground_truth_observation
        # elif self.observation_type == ObservationType.Blue:
        #     return self._blue_observation
        # elif self.observation_type == ObservationType.Prediction:
        #     return self._prediction_observation
        # else:
        #     raise ValueError("self.observation_type incorrect")

        return self._fugitive_observation

    def get_prediction_observation(self):
        return self._prediction_observation

    def get_fugitive_observation(self):
        return self._fugitive_observation

    def get_detected_blue_states(self):
       return self._detected_blue_states

    def get_ground_truth_observation(self):
        return self._ground_truth_observation

    def get_blue_observation(self):
        return self._blue_observation

    def get_modified_blue_observation_linear_estimator(self):
        return self._modified_blue_observation

    # def get_last_two_detections(self):
    #     last_k_detections = self.last_k_fugitive_detections
    #     return np.concatenate((last_k_detections[-2][-2:], last_k_detections[-2][0:1], np.array([self.timesteps/self.max_timesteps]), last_k_detections[-1][-2:], last_k_detections[-1][0:1], np.array([self.timesteps/self.max_timesteps])))

    def get_last_two_detections(self):
        last_k_detections = self.last_k_fugitive_detections
        return np.concatenate((np.array([self.timesteps/self.max_timesteps]), last_k_detections[-2], last_k_detections[-1]))

    def get_t_d1_d0_vel(self):
        last_k_detections = self.get_last_two_detections()
        x_vel = (last_k_detections[5:7] - last_k_detections[2:4]) / (last_k_detections[4:5] - last_k_detections[1:2] + 1e-10)
        t_d1_d0_vel = np.concatenate((last_k_detections, x_vel), axis=-1)
        return t_d1_d0_vel

    def get_last_two_detections_vel_time(self):
        last_k_detection_vel = self.last_k_fugitive_detections_vel
        return np.concatenate((np.array([self.timesteps/self.max_timesteps]), last_k_detection_vel[-2], last_k_detection_vel[-1]))

    def get_last_k_detections_vel_time(self):
        last_k_detection_vel = self.last_k_fugitive_detections_vel
        return np.concatenate((np.array([self.timesteps/self.max_timesteps]), np.array(last_k_detection_vel).reshape(-1)))


    def get_modified_blue_obs_last_detections(self):
        return self._modified_blue_observation_last_detections
    
    def get_modified_blue_obs_no_detections(self):
        return self._modified_blue_observation_no_detections

    def get_modified_blue_obs_no_detections_with_high_actions(self):
        return self._modified_blue_observation_no_detections_with_actions

    def get_modified_blue_obs_no_detections_with_gaussians(self):
        return self._modified_blue_observation_no_detections_with_gaussians

    def get_modified_blue_obs_map(self):
        return self._modified_blue_observation_map

    def get_flat_seq_blue_obs_no_detections(self):
        agents_seq_feats = self.blue_obs_sequence_array.transpose(1,0,2)
        return agents_seq_feats.reshape(agents_seq_feats.shape[0], agents_seq_feats.shape[1]*agents_seq_feats.shape[2])

    def get_last_k_fugitive_detections(self):
        return self.last_k_fugitive_detections

    def get_last_k_fugitive_detections_vel(self):
        return self.last_k_fugitive_detections_vel        

    def get_last_k_fugitive_with_timestep(self):
        return [i + [self.timesteps/self.max_timesteps] for i in self.last_k_fugitive_detections]

    def get_last_k_blue_obs_only_red_loc(self):
        return self.last_k_blue_obs_only_red_loc
    
    def set_last_k_blue_obs_only_red_loc(self, filter_out_prisoner_loc):
        self.last_k_blue_obs_only_red_loc[-1] = filter_out_prisoner_loc

    def get_new_t_init_lastDetection_timeInterval(self):
        last_k_detections = self.last_k_fugitive_detections_vel
        t_init = self.get_t_init_red_loc()
        last_two_detections = np.concatenate((last_k_detections[-2], last_k_detections[-1]))
        timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
        return np.concatenate((t_init, self.predicted_locations_from_last_two_detections, timeInterval, t_init, last_two_detections, timeInterval))

    # def get_mlp_quasi_sel_input(self):
    #     t_init = self.get_t_init_red_loc()
    #     last_two_detections = self.get_last_two_detections_vel_time()
    #     timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
    #     return np.concatenate((t_init, last_two_detections, timeInterval))

    def split_new_pmc_input(self, pmc_input):
        prior_input = pmc_input[..., 0:3]
        dynamic_input = pmc_input[..., 3:6]
        sel_input = pmc_input[..., 6:]
        return [prior_input, dynamic_input, sel_input]

    def get_t_init_red_loc(self):
        return np.concatenate((np.array([self.timesteps/self.max_timesteps]), np.array(self.prisoner_start_location) / 2428))

    def get_t_init_le_timeInterval(self):
        t_init = self.get_t_init_red_loc()
        timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
        return np.concatenate((t_init, self.predicted_locations_from_last_two_detections, timeInterval))

    def get_tInterval_lastDetection(self):
        timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
        last_two_detections = self.get_last_two_detections()
        return np.concatenate((timeInterval, last_two_detections[5:7]))

    def get_tInterval_lastDetection_le_tInterval(self):
        tInterval_lastDetection = self.get_tInterval_lastDetection()
        timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
        return np.concatenate((tInterval_lastDetection, self.predicted_locations_from_last_two_detections, timeInterval))

    def get_mlp_quasi_sel_input(self):
        t_init = self.get_t_init_red_loc()
        last_two_detections = self.get_last_two_detections_vel_time()
        timeInterval = np.array([self.timesteps/self.max_timesteps] - self.get_last_two_detections()[4:5])
        return np.concatenate((t_init, last_two_detections, timeInterval))

    def get_gnn_obs(self):
        # return tuple for gnn observations
        # may also need to add last k detections and starting location to this tuple?
        
        return (self._gnn_agent_obs, 
                self._known_hideouts, 
                np.array([self.timesteps/self.max_timesteps]), 
                np.array(self.total_agents_num)) 

    def get_gnn_sequence(self):
        """ return the last k timesteps of gnn observations """
        return (self.gnn_sequence_array,
                self._known_hideouts, 
                np.array([self.timesteps/self.max_timesteps]), 
                np.array(self.total_agents_num))

    def get_detection_range(self):
        detection_range_chs = []
        for camera in self.camera_list:
            detection_range_chs.append(camera.detection_range)
        for helicopter in self.helicopters_list:
            detection_range_chs.append(helicopter.detection_range)
        for search_party in self.search_parties_list:
            detection_range_chs.append(search_party.detection_range)
        return np.array(detection_range_chs)

    def set_localized_mus(self, filter_outputs, gaussian_num=8):
        mus = [filter_outputs[i][:,gaussian_num:3*gaussian_num].detach().cpu() for i in range(len(filter_outputs))]
        self.localized_mus = np.concatenate(mus, axis=0).reshape(-1, 8, 2)
        return 

    @property
    def cached_terrain_image(self):
        """
        cache terrain image to be more efficient when rendering
        :return:
        """
        if self._cached_terrain_image is None:
            # self._cached_terrain_images = [terrain.visualize(just_matrix=True) for terrain in self.terrain_list]
            self._cached_terrain_image = self.terrain_list[0].visualize(just_matrix=True)
        return self._cached_terrain_image
    
    @property
    def predicted_locations_from_last_two_detections(self):
        last_fugitive_detected_loc = np.array(self.last_k_fugitive_detections[-1][-2:])
        last_fugitive_detected_time = np.array(self.last_k_fugitive_detections[-1][0])
        second_last_fugitive_detected_loc = np.array(self.last_k_fugitive_detections[-2][-2:])
        second_last_fugitive_detected_time = np.array(self.last_k_fugitive_detections[-2][0])
        if last_fugitive_detected_time - second_last_fugitive_detected_time == 0:
            velocity = 0
        else:
            # velocity = (last_fugitive_detected_loc - second_last_fugitive_detected_loc) / (last_fugitive_detected_time - second_last_fugitive_detected_time)
            velocity = np.array([self.last_k_fugitive_detections_vel[-1][3], self.last_k_fugitive_detections_vel[-1][4]])
        delta_t = self.timesteps / self.max_timesteps - last_fugitive_detected_time
        predicted_loc = last_fugitive_detected_loc + delta_t * velocity
        # print("The prediction error is: ", np.abs(predicted_loc*2428-self.get_prisoner_location()))
        return predicted_loc.tolist()
    
    # def predict_curr_blue_state(self):
    #     last_detections_of_blue = self.last_k_fugitive_detect_blue_posVel[-1] # dim: [k,blue_ag_num,feature:5=time+loc+vel]
    #     # delta_t = 
    #     return np.concatenate((np.array([self.timesteps/self.max_timesteps]), last_k_detections[-2], last_k_detections[-1]))
    
    @property
    def predicted_relative_blue_locations_from_last_two_detections(self):
        last_detections_of_blue = self.last_k_fugitive_detect_blue_posVel[-1] # dim: [blue_ag_num,feature:5=time+loc+vel]
        ag_time, ag_loc, ag_vel = np.split(last_detections_of_blue, indices_or_sections=[1, 3], axis=-1)
        curr_blue_loc = ag_loc + (self.timesteps / self.max_timesteps - ag_time) * self.max_timesteps * ag_vel
        relative_loc = np.clip(curr_blue_loc, a_min=0, a_max=1) - np.expand_dims(np.array(self.prisoner.location)/self.dim_x, axis=0)
        # print("The prediction error is: ", np.abs(predicted_loc*2428-self.get_prisoner_location()))
        return relative_loc.flatten().tolist()


    @property
    def last_two_fugitive_detections(self):
        """ Return flattened last two fugitive detections that are flattened include (timestep_0, x_0, y_0) """
        return self.last_k_fugitive_detections[-2:]

    def fast_render_canvas(self, show=True, scale=3, predicted_prisoner_location=None, show_delta=False):
        """
        We allow the predicted prisoner location to be passed in which renders a predicted prisoner location
        show_delta: is a bool whether or not to display the square around the fugitive
        """
        # Init the canvas
        self.canvas = self.cached_terrain_image
        self.canvas = cv2.flip(self.canvas, 0)

        def calculate_appropriate_image_extent_cv(loc, radius=0.4):
            y_new = -loc[1] + self.dim_y
            return list(map(int, [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                                  max(y_new - radius, 0), min(y_new + radius, self.dim_y)]))

        def draw_radius_of_detection(location, radius):
            radius = int(radius)
            color = (0, 0, 1)  # red detection circle
            location = (int(location[0]), self.dim_y - int(location[1]))
            cv2.circle(self.canvas, location, radius, color, 2)

        def draw_image_on_canvas_cv(image, location, asset_size):

            asset_size = int(asset_size)

            if asset_size % 2 != 0:
                asset_size = asset_size - 1
            if asset_size <= 0:
                asset_size = asset_size + 1
            x_min, x_max, y_min, y_max = calculate_appropriate_image_extent_cv(location, asset_size)

            img = cv2.resize(image, (x_max - x_min, y_max - y_min))

            # create mask based on alpha channel
            mask = img[:, :, 3]
            mask[mask > 50] = 255
            mask = cv2.bitwise_not(mask)

            # cut out portion of the background where we want to paste image
            cut_background = self.canvas[y_min:y_max, x_min:x_max, :]
            img_with_background = cv2.bitwise_and(cut_background, cut_background, mask=mask) + img[:, :, 0:3] / 255

            # insert new image into background/canvas
            self.canvas[y_min:y_max, x_min:x_max, :] = img_with_background

        # fugitive_speed = prisoner.
        if self.is_detected:
            draw_image_on_canvas_cv(self.detected_prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        else:
            draw_image_on_canvas_cv(self.prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        # draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_ranges[0])
        # draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_ranges[1])
        # draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_ranges[2])

        # draw predicted prisoner location
        if predicted_prisoner_location is not None:
            # flip for canvas
            predicted_prisoner_location[1] = self.dim_y - predicted_prisoner_location[1]
            cv2.circle(self.canvas, predicted_prisoner_location, 20, (0, 0, 1), -1)

        # towns
        for town in self.town_list:
            draw_image_on_canvas_cv(self.town_pic_cv, town.location, self.default_asset_size)
        # search parties
        for search_party in self.search_parties_list:
            draw_image_on_canvas_cv(self.search_party_pic_cv, search_party.location, self.default_asset_size)
            # draw_radius_of_detection(search_party.location,
            #                          search_party.base_100_pod_distance(self.current_prisoner_speed))
            draw_radius_of_detection(search_party.location,
                                     search_party.detection_ranges[0])

        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_pic_cv, helicopter.location, self.default_asset_size)
                # draw_radius_of_detection(helicopter.location,
                #                          helicopter.base_100_pod_distance(self.current_prisoner_speed))
                draw_radius_of_detection(helicopter.location,
                                         helicopter.detection_ranges[0])
        else:
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_no_pic_cv, helicopter.location, self.default_asset_size)

        if show_delta:
            # Added by Manisha (Check first before pushing changes) delta = 0.05 = 121.4 on the map
            x1, y1 = self.prisoner.location[0] - 121, 2428 - self.prisoner.location[1] + 121
            x2, y2 = self.prisoner.location[0] + 121, 2428 - self.prisoner.location[1] - 121
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 1), 2)

        # hideouts
        for hide_id, hideout in enumerate(self.hideout_list):
            if hide_id == 0:
                draw_image_on_canvas_cv(self.known_hideout_pic_cv, hideout.location, self.hideout_radius)
            elif hide_id == 1:
                draw_image_on_canvas_cv(self.unknown_hideout1_pic_cv, hideout.location, self.hideout_radius)
            else:
                draw_image_on_canvas_cv(self.unknown_hideout2_pic_cv, hideout.location, self.hideout_radius)

        # cameras
        camera_detection_locs_ranges = []
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                draw_image_on_canvas_cv(self.known_camera_pic_cv, camera.location, camera.detection_ranges[0])
            else:
                draw_image_on_canvas_cv(self.unknown_camera_pic_cv, camera.location, camera.detection_ranges[0])
            camera_detection_locs_ranges.append([camera.location[0], camera.location[1], camera.detection_ranges[0]])

        # # Path to the YAML file
        # yaml_file_path = 'camera.yaml'

        # # Write the list of lists to the YAML file
        # with open(yaml_file_path, 'w') as file:
        #     yaml.dump(my_list, file)

        # for mountains in self.terrain.mountain_locations:
        #     # mountains[1] = -mountains[1] + self.dim_y
        #     mountain_loc = (mountains[1], mountains[0])
        #     draw_image_on_canvas_cv(self.prisoner_pic_cv, mountain_loc, 20)

        x, y, _ = self.canvas.shape
        self.canvas = cv2.resize(self.canvas, (x // scale, y // scale))
        # print(np.max(self.canvas))
        if show:
            cv2.imshow("test", self.canvas)
            cv2.waitKey(1)
        return (self.canvas * 255).astype('uint8')

    def custom_render_canvas(self, option=["prisoner", "terrain", "hideouts", "search", "cameras"], show=True, scale=3, predicted_prisoner_location=None, show_delta=False, **kwargs):
        """
        We allow the predicted prisoner location to be passed in which renders a predicted prisoner location
        show_delta: is a bool whether or not to display the square around the fugitive
        """
        large_icons = kwargs['large_icons']

        # INFO: Init the canvas
        self.canvas = self.cached_terrain_image
        self.canvas = cv2.flip(self.canvas, 0)

        def calculate_appropriate_image_extent_cv(loc, radius=0.4):
            y_new = -loc[1] + self.dim_y
            return list(map(int, [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                                  max(y_new - radius, 0), min(y_new + radius, self.dim_y)]))

        def draw_radius_of_detection(location, radius):
            radius = int(radius)
            color = (0, 0, 1)  # red detection circle
            location = (int(location[0]), self.dim_y - int(location[1]))
            cv2.circle(self.canvas, location, radius, color, 2)

        def draw_image_on_canvas_cv(image, location, asset_size):

            asset_size = int(asset_size)

            if asset_size % 2 != 0:
                asset_size = asset_size - 1
            if asset_size <= 0:
                asset_size = asset_size + 1
            x_min, x_max, y_min, y_max = calculate_appropriate_image_extent_cv(location, asset_size)

            img = cv2.resize(image, (x_max - x_min, y_max - y_min))

            # create mask based on alpha channel
            mask = img[:, :, 3]
            mask[mask > 50] = 255
            mask = cv2.bitwise_not(mask)

            # cut out portion of the background where we want to paste image
            cut_background = self.canvas[y_min:y_max, x_min:x_max, :]
            img_with_background = cv2.bitwise_and(cut_background, cut_background, mask=mask) + img[:, :, 0:3] / 255

            # insert new image into background/canvas
            self.canvas[y_min:y_max, x_min:x_max, :] = img_with_background

        # fugitive_speed = prisoner.

        # INFO: draw predicted prisoner location
        if predicted_prisoner_location is not None:
            # flip for canvas
            predicted_prisoner_location[1] = self.dim_y - predicted_prisoner_location[1]
            cv2.circle(self.canvas, predicted_prisoner_location, 20, (0, 0, 1), -1)

        # INFO: towns
        for town in self.town_list:
            draw_image_on_canvas_cv(self.town_pic_cv, town.location, self.default_asset_size)

        # INFO: search parties
        if "search" in option:
            for i, search_party in enumerate(self.search_parties_list):
                # icon = [self.search_party_pic_large_cv_a, self.search_party_pic_large_cv_b][i % 2]
                if large_icons:
                    draw_image_on_canvas_cv(icon, search_party.location, self.large_asset_size)
                else:
                    draw_image_on_canvas_cv(self.search_party_pic_cv, search_party.location, self.default_asset_size)

                draw_radius_of_detection(search_party.location, 70)

            # INFO: helicopters
            if self.is_helicopter_operating():
                for helicopter in self.helicopters_list:
                    if large_icons:
                        draw_image_on_canvas_cv(self.helicopter_pic_large_cv, helicopter.location, self.large_asset_size)
                    else:
                        draw_image_on_canvas_cv(self.helicopter_pic_cv, helicopter.location, self.default_asset_size)
                    draw_radius_of_detection(helicopter.location, 100)
            else:
                for helicopter in self.helicopters_list:
                    if large_icons:
                        draw_image_on_canvas_cv(self.helicopter_pic_large_cv, helicopter.location, self.large_asset_size)
                    else:
                        draw_image_on_canvas_cv(self.helicopter_no_pic_cv, helicopter.location, self.default_asset_size)

        if show_delta:
            # Added by Manisha (Check first before pushing changes) delta = 0.05 = 121.4 on the map
            x1, y1 = self.prisoner.location[0] - 121, self.dim_y - self.prisoner.location[1] + 121
            x2, y2 = self.prisoner.location[0] + 121, self.dim_y - self.prisoner.location[1] - 121
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 1), 2)

        # INFO: prisoner
        if "prisoner" in option:
            if self.is_detected:
                if large_icons:
                    draw_image_on_canvas_cv(self.detected_prisoner_pic_large_cv, self.prisoner.location, self.large_asset_size)
                else:
                    draw_image_on_canvas_cv(self.detected_prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
            else:
                if large_icons:
                    draw_image_on_canvas_cv(self.prisoner_pic_large_cv, self.prisoner.location, self.large_asset_size)
                else:
                    draw_image_on_canvas_cv(self.prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
            # draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_range)

        # INFO: hideouts
        if "hideouts" in option:
            for hide_id, hideout in enumerate(self.hideout_list):
                if 1:
                    draw_image_on_canvas_cv(self.known_hideout_pic_cv, hideout.location, 1.5*self.hideout_radius)

        # INFO: cameras
        if "cameras" in option:
            for camera in self.camera_list:
                if camera.known_to_fugitive:
                    draw_image_on_canvas_cv(self.known_camera_pic_cv, camera.location, camera.detection_range)
                else:
                    draw_image_on_canvas_cv(self.unknown_camera_pic_cv, camera.location, camera.detection_range*2)

        # INFO: mountains if it is not in the terrain map
        # for mountains in self.terrain.mountain_locations:
        #     # mountains[1] = -mountains[1] + self.dim_y
        #     mountain_loc = (mountains[1], mountains[0])
        #     draw_image_on_canvas_cv(self.prisoner_pic_cv, mountain_loc, 20)

        x, y, _ = self.canvas.shape
        self.canvas = cv2.resize(self.canvas, (x // scale, y // scale))
        # print(np.max(self.canvas))
        if show:
            cv2.imshow("test", self.canvas)
            cv2.waitKey(1)
        return (self.canvas[..., ::-1] * 255).astype('uint8')
    
    def render(self, mode, show=True, fast=False, scale=3, show_delta=False):
        """
        Render the environment.
        :param mode: required by `gym.Env` but we ignore it
        :param show: whether to show the rendered image
        :param fast: whether to use the fast version for render. The fast version takes less time to render but the render quality is lower.
        :param scale: scale for fast render
        :param show_delta: is a bool whether or not to display the square around the fugitive
        :return: opencv img object
        """
        if fast:
            return self.fast_render_canvas(show, scale, show_delta=show_delta)
        else:
            return self.slow_render_canvas(show)

    def slow_render_canvas(self, show=True):
        """
        Provide a visualization of the current status of the environment.

        In rendering, imshow interprets the matrix as:
        [x, 0]
        ^
        |
        |
        |
        |
        |----------->[0, y]
        However, the extent of the figure is still:
        [0, y]
        ^
        |
        |
        |
        |
        |----------->[x, 0]
        Read https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html for more explanations.

        :param show: whether to show the visualization directly or just return
        :return: an opencv img object
        """

        def calculate_appropriate_image_extent(loc, radius=0.4):
            """
            :param loc: the center location to put a picture
            :param radius: the radius (size) of the figure
            :return: [left, right, bottom, top]
            """
            return [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                    max(loc[1] - radius, 0), min(loc[1] + radius, self.dim_y)]

        fig, ax = plt.subplots(figsize=(20, 20))
        # Show terrain
        im = ax.imshow(self.cached_terrain_image, origin='lower')
        # labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # prisoner_history
        prisoner_location_history = np.array(self.prisoner_location_history)
        ax.plot(prisoner_location_history[:, 0], prisoner_location_history[:, 1], "r")

        # prisoner
        if self.is_detected:
            ax.imshow(self.detected_prisoner_pic,
                      extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))
        else:
            ax.imshow(self.prisoner_pic, extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))

        # towns
        for town in self.town_list:
            ax.imshow(self.town_pic, extent=calculate_appropriate_image_extent(town.location, radius=30))
        # search parties
        for search_party in self.search_parties_list:
            ax.imshow(self.search_party_pic, extent=calculate_appropriate_image_extent(search_party.location,
                                                                                       radius=search_party.detection_range))
        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                ax.imshow(self.helicopter_pic, extent=calculate_appropriate_image_extent(helicopter.location,
                                                                                         radius=helicopter.detection_range))
        # hideouts
        for hide_id, hideout in enumerate(self.hideout_list):
            if hide_id == 0:
                ax.imshow(self.known_hideout_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))
            elif hide_id == 1:
                ax.imshow(self.unknown_hideout1_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))
            else:
                ax.imshow(self.unknown_hideout2_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))                

        # cameras
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                ax.imshow(self.known_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                           radius=camera.detection_range))
            else:
                ax.imshow(self.unknown_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                             radius=camera.detection_range))
        # finalize
        ax.axis('scaled')
        plt.savefig("simulator/temp.png")
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if show:
            plt.show()
        plt.close()

        return img

    def get_prisoner_location(self):
        return self.prisoner.location

    def get_prisoner_velocity(self):
        return self.prisoner.step_dist_xy

    def get_blue_locations(self):
        search_party_locations = []
        for search_party in self.search_parties_list:
            search_party_locations.append(search_party.location.copy())
        helicopter_locations = []
        for helicopter in self.helicopters_list:
            helicopter_locations.append(helicopter.location.copy())
        return (search_party_locations), (helicopter_locations)

    def get_blue_velocities(self):
        search_party_velocities = []
        for search_party in self.search_parties_list:
            search_party_velocities.append(search_party.step_dist_xy.copy())
        helicopter_velocities = []
        for helicopter in self.helicopters_list:
            helicopter_velocities.append(helicopter.step_dist_xy.copy())
        return (search_party_velocities), (helicopter_velocities)

    def get_hs_locVels(self):
        helicopter_locVels = []
        for helicopter in self.helicopters_list:
            helicopter_locVel = helicopter.location.copy() + helicopter.step_dist_xy.tolist()
            helicopter_locVels.append(np.array(helicopter_locVel))
        search_party_locVels = []
        for search_party in self.search_parties_list:
            search_party_locVel = search_party.location.copy() + search_party.step_dist_xy.tolist()
            search_party_locVels.append(np.array(search_party_locVel))
        return helicopter_locVels + search_party_locVels

    def get_relative_hs_loc(self):
        helicopter_locs = []
        for helicopter in self.helicopters_list:
            relative_loc = np.array(helicopter.location.copy()) - np.array(self.prisoner.location)
            helicopter_locs.append(relative_loc)
        search_party_locs = []
        for search_party in self.search_parties_list: 
            relative_loc = np.array(search_party.location.copy()) - np.array(self.prisoner.location)
            search_party_locs.append(relative_loc)
        return np.vstack(helicopter_locs + search_party_locs)

    def get_relative_hs_locVels(self):
        helicopter_locVels = []
        for helicopter in self.helicopters_list:
            helicopter_locVel = helicopter.location.copy() + helicopter.step_dist_xy.tolist()
            helicopter_locVels.append(np.array(helicopter_locVel)-np.array(self.prisoner.location+[0,0]))
        search_party_locVels = []
        for search_party in self.search_parties_list:
            search_party_locVel = search_party.location.copy() + search_party.step_dist_xy.tolist()
            search_party_locVels.append(np.array(search_party_locVel)-np.array(self.prisoner.location+[0,0]))
        return helicopter_locVels + search_party_locVels
    
    def get_red_blue_state(self):
        red_loc = np.array(self.prisoner.location) / self.dim_x
        red_vel = self.current_prisoner_velocity / self.max_timesteps
        blue_relatives = np.concatenate(self.get_relative_hs_locVels(), axis=0) / self.dim_x
        return np.concatenate((red_loc, red_vel, blue_relatives), axis=0)

    def get_red_blue_state_detect_seq(self, seq_len_in, seq_len_out):
        return self.last_k_red_blue_states[-seq_len_in:], self.lask_k_blue_detection_of_red[-seq_len_out:]

    def generate_policy_heatmap(self, current_state, policy, num_timesteps=2500, num_rollouts=20, end=False):
        """
        Generates the heatmap displaying probabilities of ending up in certain cells
        :param current_state: current location of prisoner, current state of world
        :param policy: must input state, output action
        :param num_timesteps: how far in time ahead, remember time is in 15 minute intervals.
        """

        # Create 2D matrix
        display_matrix = np.zeros((self.dim_x + 1, self.dim_y + 1))

        for num_traj in tqdm(range(num_rollouts), desc="generating_heatmap"):
            observation = self.reset()
            for j in range(num_timesteps):
                if policy == 'rand':
                    action = self.action_space.sample()
                else:
                    action = policy.predict(observation, deterministic=False)[0]
                    # action = policy(observation)
                    # theta = policy([observation])
                    # action = np.array([7.5, theta[0]], dtype=np.float32)
                observation, reward, done, _ = self.step(action)
                # update count
                if not end:
                    display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                if done:
                    if end:
                        display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                    break
            if end:
                display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                # self.render('human', show=True)
        fig, ax = plt.subplots()
        display_matrix = np.transpose(display_matrix)
        from scipy.ndimage import gaussian_filter
        # smooth the matrix
        smoothed_matrix = gaussian_filter(display_matrix, sigma=50)
        # Set 0s to None as they will be ignored when plotting
        # smoothed_matrix[smoothed_matrix == 0] = None
        display_matrix[display_matrix == 0] = None
        # Plot the data
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                sharex=False, sharey=True,
                                figsize=(5, 5))
        # ax1.matshow(display_matrix, cmap='hot')
        # ax1.set_title("Original matrix")
        im = ax1.matshow(smoothed_matrix)
        num_hours = str((num_timesteps / 60).__round__(2))

        ax1.set_title("Heatmap at Time t=" + num_hours + ' hours')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([])
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        cbar.ax.invert_yaxis()
        plt.show()
        # fig, ax1 = plt.subplots(nrows=1, ncols=1,
        #                                sharex=False, sharey=True,
        #                                figsize=(5, 5))
        # plt.imshow(smoothed_matrix)
        # # plt.set_title("Heatmap at Time t="+ str(num_timesteps) + ' minutes')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

        print("saving heatmap")
        plt.savefig("simulator/temp.png")

        # im = ax.imshow(display_matrix, cmap="hot", origin="lower", vmin=0, vmax=np.max(display_matrix), interpolation='None')
        # # colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cbar = fig.colorbar(im, cax=cax)
        # cbar.ax.get_yaxis().labelpad = 15
        # cbar.set_label('Count', rotation=270)
        #
        # plt.grid()
        # plt.show()


# class PrisonerGoalEnv(PrisonerRedEnv):
#     """
#     PrisonerEnv with a goal coordinate appended to its observations
#     """

#     def __init__(self,
#                  goal_mode=None,
#                  shape_reward_scale=0.,
#                  **kwargs):
#         """
#         :param goal_mode: specifies how to treat the goal signal. Can be:
#             None, 'fixed': goal is initialized to 0, 0 and never changed; can be moved manually
#             'single': a single hideout goal is chosen at environment reset
#         :param shape_reward_scale: scale of an extra reward component to encourage exploration towards the goal
#             The value of this parameter is the reward when the agent and goal are on opposite corners.
#             Interpolates linearly to zero.
#         """
#         # Future: replace shape_reward_scale with some more sophisticated input if we emply multiple kinds of reward shaping
#         super().__init__(observation_step_type="Fugitive", **kwargs)
#         self.goal_mode = goal_mode
#         self.goal = np.array([0., 0.])
#         self.shape_reward_scale = shape_reward_scale
#         self.obs_names.add_name('goal', 2)

#         # append two slots to the observation space
#         o = self.observation_space
#         h, l, d = o.high.tolist(), o.low.tolist(), o.dtype
#         h.extend([1., 1.])
#         l.extend([0., 0.])
#         h, l = np.array(h), np.array(l)
#         self.observation_space = gym.spaces.Box(high=h, low=l, dtype=d)

#     def reset(self):
#         sub_obs = super().reset()
#         obs = np.zeros(self.observation_space.shape)
#         obs[:-2] = sub_obs
#         if self.goal_mode == 'single':
#             hideout_id = np.random.randint(len(self.hideout_list))
#             self.goal = self.hideout_list[hideout_id].location
#         elif self.goal_mode == 'switch':
#             self.timer = np.random.randint(500,1500)
#             hideout_id = np.random.randint(len(self.hideout_list))
#             self.goal = self.hideout_list[hideout_id].location
#         obs[-2:] = self.cell_to_obs(self.goal)
#         return obs

#     def step(self, action):
#         if self.goal_mode == 'switch':
#             if ((self.timesteps % self.timer) == 0) and ((self.max_timesteps-self.timesteps)>self.timer):
#                 hideout_id = np.random.randint(len(self.hideout_list))
#                 self.goal = self.hideout_list[hideout_id].location
#         sub_obs, r, d, i = super().step(action)
#         # sparse goal reward
#         # r = -5e-4
#         # hideout = self.near_hideout()
#         # if hideout is not None:
#         #     if np.all(hideout.location == self.goal):
#         #         r = 2.0

#         obs = np.zeros(self.observation_space.shape)
#         obs[:-2] = sub_obs
#         obs[-2:] = self.cell_to_obs(self.goal)
#         if self.shape_reward_scale > 0:
#             r -= self.shape_reward_scale * self.dist_to_goal() / \
#                  np.linalg.norm(np.array([self.dim_x, self.dim_y]))
#         return obs, r, d, i

#     def set_hideout_goal(self, index):
#         """
#         set the current goal to the location of one of the hideouts by index
#         """
#         self.goal = np.array(self.hideout_list[index].location)

#     def vector_to_goal(self, obs=None):
#         if obs is None:
#             goal = np.array(self.goal)
#             prisloc = np.array(self.prisoner.location)
#         else:
#             obs = self.obs_names(obs) # old obs accessible as obs.array
#             prisloc = self.obs_to_cell(obs['prisoner_loc'])
#             goal = self.obs_to_cell(obs['goal'])
#         return goal - prisloc

#     def dist_to_goal(self, obs=None):
#         return np.linalg.norm(self.vector_to_goal(obs))

#     def angle_to_goal(self, obs=None):
#         vector = self.vector_to_goal(obs)
#         return math.atan2(vector[1], vector[0])


if __name__ == "__main__":
    np.random.seed(20)
    # p = PrisonerEnv()
    # p.generate_policy_heatmap(p.reset(), policy='rand', end=True)