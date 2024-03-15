from simulator.prisoner_env import PrisonerBothEnv, ObservationType
from red_bc.heuristic import BlueHeuristic, SimplifiedBlueHeuristic
# from blue_policies.heuristic import BlueHeuristic
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
import numpy as np
import gym
from gym import spaces

DIM_X = 2428
DIM_Y = 2428

class PrisonerEnv(gym.Wrapper):
    """ Produce environment to match our previous implementation to hot swap in
    
    This environment returns red observations and takes in red actions
    """
    def __init__(self,
                 env: PrisonerBothEnv,
                 blue_policy: None):
        super().__init__(env)
        self.env = env
        self.blue_policy = blue_policy
        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        self.observation_space = self.env.fugitive_observation_space
        self.obs_names = self.env.fugitive_obs_names

    def reset(self, seed=None):
        self.env.reset(seed)
        if type(self.blue_policy) == BlueHeuristic or (type(self.blue_policy) == SimplifiedBlueHeuristic):
            self.blue_policy.reset()
            self.blue_policy.init_behavior()
        return self.env._fugitive_observation
        
    def step(self, red_action):
        # get red observation for policy
        blue_obs_in = self.env._blue_observation
        if type(self.blue_policy) == SimplifiedBlueHeuristic:
            blue_action = self.blue_policy.predict_full_observation(blue_obs_in)
        else:
            blue_action = self.blue_policy.predict(blue_obs_in)
            
        # red_obs, _, reward, done, i = self.env.step_both(red_action, blue_action)
        red_obs, _, reward, dist_reward, done, i = self.env.step_both(red_action, blue_action)
        return red_obs, reward, done, i 



class PrisonerRedEnv(gym.Wrapper):
    """ Produce environment to match our previous implementation to hot swap in
    
    This environment returns red observations & red partial observations and takes in red actions
    """
    def __init__(self,
                 env: PrisonerBothEnv,
                 blue_policy: None):
        super().__init__(env)
        self.env = env
        self.blue_policy = blue_policy
        # self.red_policy = red_policy
        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        self.observation_space = self.env.fugitive_observation_space
        self.obs_names = self.env.fugitive_obs_names
        self.partial_observation_space_shape = None
        self.goal_location = None
        self.red_action = np.array([15.0, np.pi])
        self.prisoner_state = [env.prisoner.location[0], env.prisoner.location[1], self.red_action[1], self.red_action[0], self.red_action[1]]

    def get_prisoner_state(self):
        self.prisoner_state = [self.env.prisoner.location[0], self.env.prisoner.location[1], self.red_action[1], self.red_action[0], self.red_action[1]]
        return self.prisoner_state

    def update_prisoner_state(self):
        self.prisoner_state = [self.env.prisoner.location[0], self.env.prisoner.location[1], self.red_action[1], self.red_action[0], self.red_action[1]]
        return 

    def reset(self, seed=None, reset_type=None, red_policy=None, waypt_seed=None):
        self.env.reset_env(seed)
        if red_policy:
            red_policy.get_scaled_path(waypt_seed)
        self.env.reset_obs(seed)
        if type(self.blue_policy) == BlueHeuristic or (type(self.blue_policy) == SimplifiedBlueHeuristic):
            self.blue_policy.reset()
            self.blue_policy.init_behavior()

        # red_obs_names = self.env.fugitive_obs_names
        wrapped_red_observation = self.obs_names(self.env._fugitive_observation)
        """include all hideout goals"""
        # simplified_red_obs = np.concatenate([wrapped_red_observation["hideout_known_0"], wrapped_red_observation["hideout_loc_0"], wrapped_red_observation["hideout_known_1"], wrapped_red_observation["hideout_loc_1"], wrapped_red_observation["hideout_known_2"], wrapped_red_observation["hideout_loc_2"], wrapped_red_observation["prisoner_loc"]])
        """include no hideout goals"""
        # simplified_red_obs = wrapped_red_observation["prisoner_loc"]
        """goal hideout location when goes to unknown hideout"""
        # goal_location = self.env.closest_unknown_hideout_location/DIM_X
        """goal hideout location when goes to known hideout"""
        # goal_location = self.env.closest_known_hideout_location/DIM_X
        # goal_location = closest_unknown_hideout.location
        """include goal location"""
        # simplified_red_obs = np.concatenate([goal_location, simplified_red_obs])
        # INFO: include all detections
        # final_camera_idx = self.env.num_known_cameras + self.env.num_unknown_cameras - 1
        # red_detections_of_blue = wrapped_red_observation.get_section_include_terminals(key_start="helicopter_detect_0", key_end="camera_detect_%d"%final_camera_idx)
        # INFO: include no detections
        red_detections_of_blue = np.array([])
        # simplified_red_obs = np.concatenate([simplified_red_obs, self.env._fugitive_observation[-18:]])
        # self.partial_observation_space_shape = simplified_red_obs.shape
        self.red_action = np.array([15.0, np.pi])
        self.prisoner_state = [self.env.prisoner.location[0], self.env.prisoner.location[1], self.red_action[1], 0, 0]
        return [self.env._fugitive_observation], [red_detections_of_blue]
        
    def step(self, red_action):
        self.red_action = red_action
        # get red observation for policy
        blue_obs_in = self.env._blue_observation
        if type(self.blue_policy) == SimplifiedBlueHeuristic:
            blue_action = self.blue_policy.predict_full_observation(blue_obs_in)
        else:
            blue_action = self.blue_policy.predict(blue_obs_in)
            
        # red_obs, _, reward, done, i = self.env.step_both(red_action, blue_action)
        red_obs, _, reward, done, i, red_detected_by_hs_flag = self.env.step_both(red_action, blue_action)
        self.update_prisoner_state()

        # red_obs_names = self.env.red_obs_names
        wrapped_red_observation = self.obs_names(red_obs)
        """include all hideout goals"""
        # simplified_red_obs = np.concatenate([wrapped_red_observation["hideout_known_0"], wrapped_red_observation["hideout_loc_0"], wrapped_red_observation["hideout_known_1"], wrapped_red_observation["hideout_loc_1"], wrapped_red_observation["hideout_known_2"], wrapped_red_observation["hideout_loc_2"], wrapped_red_observation["prisoner_loc"]])
        """include no hideout goals"""
        # simplified_red_obs = wrapped_red_observation["prisoner_loc"]
        """goal hideout location"""
        # if self.goal_location is not None:
        #     goal_location = self.goal_location/DIM_X
        #     """include goal location"""
        #     simplified_red_obs = np.concatenate([goal_location, simplified_red_obs])
        # else:
        #     goal_location = None
        #     simplified_red_obs = np.array([])
        # # goal_location = closest_unknown_hideout.location

        # self.partial_observation_space_shape = simplified_red_obs.shape
        # INFO: include all detections
        # final_camera_idx = self.env.num_known_cameras + self.env.num_unknown_cameras - 1
        # red_detections_of_blue = wrapped_red_observation.get_section_include_terminals(key_start="helicopter_detect_0", key_end="camera_detect_%d"%final_camera_idx)

        # INFO: include no detections
        red_detections_of_blue = np.array([])

        return [red_obs], [reward], done, i, red_detections_of_blue, red_detected_by_hs_flag


class PrisonerBlueEnv(gym.Wrapper):
    """ This environment return blue observations and takes in blue actions """
    def __init__(self,
                 env: PrisonerBothEnv,
                 fugitive_policy,
                 subpolicy_num = 4,
                 shared_obs = False): # Shared obs is for the case where we want to share the observation and actions between the subpolicies
        
        super().__init__(env)
        self.env = env
        self.fugitive_policy = fugitive_policy

        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        # self.observation_space = self.env.blue_observation_space # Manisha gail
        """Partial Blue Observation"""
        self.observation_space = self.env.blue_observation_space
        self.action_space = spaces.Box(low=-np.ones(12), high=np.ones(12))
        self.blue_partial_observation_space = self.env.blue_partial_observation_space
        self.obs_names = self.env.blue_obs_names
        self.subpolicy_shape = (9,)
        self.subpolicy_num = subpolicy_num
        self.max_para_num = 3
        """Full Blue Observation"""

    # def reset(self, seed=None):
    #     self.env.reset(seed)
    #     if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
    #         self.fugitive_policy.reset()
    #     return self.env._blue_observation, self.env._partial_blue_observation

    def reset(self, seed=None, reset_type=None):
        if reset_type == "env":
            self.env.reset_env(seed)
            if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
                self.fugitive_policy.reset()
            return 
        elif reset_type == "obs":
            self.env.reset_obs(seed)
            if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
                self.fugitive_policy.reset()
            return self.env._blue_observation, self.env._partial_blue_observation
        else:
            self.env.reset_env(seed)
            self.env.reset_obs(seed)
            if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
                self.fugitive_policy.reset()
            return self.env._blue_observation, self.env._partial_blue_observation
             


        
        
    def step(self, blue_action, localized_trgt_gaussians=None):
        """Full Blue Observation"""
        red_obs_in = self.env._fugitive_observation
        red_action = self.fugitive_policy.predict(red_obs_in)
        _, blue_obs, detect_reward, dist_reward, done, blue_detect_idx = self.env.step_both(red_action[0], blue_action, localized_trgt_gaussians)
        # return blue_obs, {}, reward, done, i# Manisha gail        
        """Partial Blue Observation"""
        # # get red observation for policy
        # red_obs_in = self.env._fugitive_observation
        # red_action = self.fugitive_policy.predict(red_obs_in)
        # _, blue_obs, blue_partial_obs, reward, done, i = self.env.step_partial_blue_obs(red_action[0], blue_action)
        return blue_obs, self.env._partial_blue_observation, detect_reward, dist_reward, done, blue_detect_idx # Manisha gail
        # # return blue_partial_obs, reward, done, i