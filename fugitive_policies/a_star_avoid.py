import numpy as np
import torch

from .utils import clip_theta, distance, c_str
import matplotlib.pyplot as plt
import time
from fugitive_policies.custom_queue import QueueFIFO
# from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial, Plotter
from fugitive_policies.a_star_policy import AStarPolicy
from fugitive_policies.a_star.a_star import a_star
from fugitive_policies.a_star.utils import plot_path
from fugitive_policies.diffusion_policy import generate_uniform_path
from robot_lib.spline import Spline2D


DIM_X = 2428
DIM_Y = 2428

# MOUNTAIN_OUTER_RANGE = 150
MOUNTAIN_INNER_RANGE = 140
import math
import random
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

class AStarAdversarialAvoid(AStarPolicy):
    def __init__(self, env,             
            max_speed=7.5,
            cost_coeff=1000,
            visualize=False):

        super().__init__(env,             
            max_speed=max_speed,
            cost_coeff=cost_coeff,
            visualize=visualize)
        self.DEBUG = False
        self.MIN_DIST_TO_HIDEOUT = 50
        self.mountain_travel = "optimal"
        self.reset()
        self.evasive_arr = self.initialize_evasive_array(self.terrain.forest_density_array)

    def initialize_evasive_array(self, terrain):
        """ Cache the evasive array so we do not have to compute every time
        Compare with old evasive code to see where numbers come from 

        """
        array = terrain
        array = (array < 0.4) * 1.0 # turn to float

        array_pool = pool2d(array, 6, 1, pool_mode='avg') * 36 # 6x6 grid so multiply by 36
        array_thresh = (array_pool > 2) * 255

        nonzero = np.where(array_thresh > 0)
        nonzero = np.stack(nonzero, axis=-1)

        return nonzero

    def reset(self):
        self.reset_plan()
        self.current_behavior = None
        self.current_hideout_goal = None
        self.behaviors = ['evade heli', 'evade search party', 'speed to known hideout',
                          'speed to unknown hideout']
        self.last_action = None
        self.behavior_completed = False
        self.being_tracked_for_n_timesteps = []

    def action_to_different_unknown_hideout(self, current_goal):
        hideout_distances = {}
        for hideout in self.observations.unknown_hideout_list:
            if (hideout.location == current_goal.location).all():
                continue
            dist = distance(self.observations.location, hideout.location)
            hideout_distances[hideout] = dist

        # choose closest distance hideout
        hid = min(hideout_distances.items(), key=lambda x:x[1])

        # hid = min(hideout_distances, key=hideout_distances.get())

        theta = self.calculate_desired_heading(self.observations.location, hid[0].location)
        self.current_hideout_goal = hid[0]
        self.current_behavior_heading = theta
        return np.array([1, theta], dtype=np.float32)

    def straight_line_action_to_closest_unknown_hideout(self):
        _, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_unknown_hideout.location)
        dist = distance(self.observations.location, closest_unknown_hideout.location)
        speed = np.clip(dist, 0, 7.5)
        self.current_hideout_goal = closest_unknown_hideout
        return np.array([speed, theta], dtype=np.float32)

    def action_to_closest_unknown_hideout(self, plot=False):
        """ Uses A* to plan path to closest unknown hideout """

        # mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(self.observations.location, 160)
        # if mountain_in_range:
        #     # if we're within a mountain, don't use A*
        #     return self.wrapper_avoid_mountain(self.straight_line_action_to_closest_unknown_hideout())

        # self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
            self.current_hideout_goal = closest_unknown_hideout
            goal_location = closest_unknown_hideout.location
            start_pos = list(map(int, self.observations.location))
            scale_start_pos = (int(start_pos[0] / self.x_scale), int(start_pos[1] / self.y_scale))
            scale_goal_location = (int(goal_location[0] / self.x_scale), int(goal_location[1] / self.y_scale))
            path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement='8N', occupancy_cost_factor=self.cost_coeff)
            scaled_path = self.scale_a_star_path(path, start_pos, goal_location)
            self.actions = self.convert_path_to_actions(scaled_path)
            assert len(path) > 0, "No path found"
            if plot:
                self.gmap.plot()
                plot_path(path_px)
        return self.actions.pop(0)

    def action_to_random_hideout(self, plot=False):
        """ Uses A* to plan path to a random hideout """

        # mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(self.observations.location, 160)
        # if mountain_in_range:
        #     # if we're within a mountain, don't use A*
        #     return self.wrapper_avoid_mountain(self.straight_line_action_to_closest_unknown_hideout())

        # self.observations.process_observation(observation)
        if len(self.actions) == 0:
            # self.observations.location
            random_hideout = self.get_random_hideout()
            self.current_hideout_goal = random_hideout
            goal_location = random_hideout.location
            start_pos = list(map(int, self.observations.location))
            scale_start_pos = (int(start_pos[0] / self.x_scale), int(start_pos[1] / self.y_scale))
            scale_goal_location = (int(goal_location[0] / self.x_scale), int(goal_location[1] / self.y_scale))
            path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement='8N', occupancy_cost_factor=self.cost_coeff)
            scaled_path = self.scale_a_star_path(path, start_pos, goal_location)
            self.scaled_path = generate_uniform_path(scaled_path, goal_location, total_points=60, normalized_path=False, normalized_hideout=False)
            self.actions = self.convert_path_to_actions(self.scaled_path)
            assert len(path) > 0, "No path found"
            if plot:
                self.gmap.plot()
                plot_path(self.scaled_path, [goal_location])
        return self.actions.pop(0)

    def reset_plan(self):
        """ Removes our A* plan """
        self.actions = []

    def get_desired_action(self, observation):
        self.observations.process_observation(observation)

        detected_helicopter = self.observations.detected_helicopter()
        detected_search_party = self.observations.detected_search_party()

        closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
        camera_in_range = self.observations.in_range_of_camera(self.observations.action[0])
        if self.DEBUG:
            print('t=', self.env.timesteps)
            print('fugitive current location', self.observations.location)
            print('detected helicopter', detected_helicopter)
            print('detected search party ',  detected_search_party)
            print('camera in range', camera_in_range)
        self.update_tracking(detected_helicopter, detected_search_party, camera_in_range)

        # if detected_helicopter or detected_search_party:
            # print("Detected helicopter or search party")

        if camera_in_range and detected_helicopter and detected_search_party:
            # Go to hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
        elif detected_helicopter and detected_search_party and np.sum(self.being_tracked_for_n_timesteps[-100:]) > 50:
            if self.DEBUG:
                print('you have been tracked for too long and failed')
            # Failed, go to hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            self.reset_plan()
            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))

        # only detected by helicopter and have seen it in 7/10 of the last 10 timesteps
        elif detected_helicopter and np.sum(self.being_tracked_for_n_timesteps[-10:]) >= 7:
            # check if you are in dense forest
            in_dense_forest = self.in_dense_forest(self.observations.location)
            # 7/10 helicopter - but helicopter is far away now: to hideout
            if distance(self.observations.location, detected_helicopter.location) > 100:
                if self.DEBUG:
                    print('dont worry about heli, its too far')
                self.last_action = 'dont worry about heli, its too far'
                desired_action = self.action_to_closest_unknown_hideout()
            # 7/10 helicopter - but helicopter is far away in dense forest: to hideout
            elif in_dense_forest and distance(self.observations.location, detected_helicopter.location) > 50:
                # you have distanced yourself enough away from the heli
                if self.DEBUG:
                    print('you have distanced yourself enough away from the heli')
                desired_action = self.action_to_closest_unknown_hideout()
            # 7/10 helicopter - and the helicopter is not away
            else:
                # you are in forest, start evading, or continue evading
                if in_dense_forest and self.current_behavior == self.behaviors[0]:
                    # you have almost evaded, slow down, change direction
                    if self.last_action == 'you have almost evaded, slow down, change direction':
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                        if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                            if self.DEBUG:
                                print('you have been tracked for too long and failed')
                            # Go to hideout
                            theta = self.calculate_desired_heading(self.observations.location,
                                                                   closest_known_hideout.location)
                            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(self.action_to_different_unknown_hideout(self.current_hideout_goal))
                        self.current_behavior_heading = desired_action[1]
                        self.last_action = 'you have almost evaded, slow down, change direction'
                # you are evading, but not in dense forest yet
                elif self.current_behavior == self.behaviors[0]:
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                    if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                        if self.DEBUG:
                            print('you have been tracked for too long and failed')
                        # zoom to the closest known hideout
                        theta = self.calculate_desired_heading(self.observations.location,
                                                               closest_known_hideout.location)
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
                # you are not evading, but you should evade from now
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[0]
                    self.current_behavior_heading = theta
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([5, theta], dtype=np.float32))

        elif detected_search_party and np.sum(self.being_tracked_for_n_timesteps[-10:]) >= 7:
            # speed up and try and run away from search parties
            if self.DEBUG:
                print('speed up and try and run away from search parties')
            in_dense_forest = self.in_dense_forest(self.observations.location)
            if distance(self.observations.location, detected_search_party.location) > 50:
                if self.DEBUG:
                    print('dont worry about search party, its too far')
                self.last_action = 'dont worry about heli, its too far'
                desired_action = self.action_to_closest_unknown_hideout()
            elif in_dense_forest and distance(self.observations.location, detected_search_party.location) > 20:
                # you have distanced yourself enough away from the heli
                if self.DEBUG:
                    print('you have distanced yourself enough away from the search part')
                desired_action = self.action_to_closest_unknown_hideout()
            # 7/10 of search party - start evading, or continue evading
            else:
                # you are now in the dense forest and evading heli
                if in_dense_forest and self.current_behavior == self.behaviors[0]:
                    self.reset_plan()
                    # you have almost evaded, slow down, change direction
                    if self.last_action == 'you have almost evaded, slow down, change direction':
                        desired_action = self.wrapper_avoid_mountain(np.array([2, self.current_behavior_heading], dtype=np.float32))
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(self.action_to_different_unknown_hideout(self.current_hideout_goal))
                        self.last_action = 'you have almost evaded, slow down, change direction'
                elif self.current_behavior == self.behaviors[1]:
                    # you are evading, but not in dense forest yet
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                    
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[1]
                    self.current_behavior_heading = theta
                    self.reset_plan()
                    desired_action = np.array([7.5, theta], dtype=np.float32)
        else:
            if self.DEBUG:
                print('you are detecting nothing, move to closest hideout with rrt star')
            desired_action = self.action_to_closest_unknown_hideout()
        # import time
        # time.sleep(1)
        if self.DEBUG:
            print('---------------------------')
        return desired_action

    def wrapper_avoid_mountain(self, desired_action):
        """Takes a desired action and ensures we don't hit mountain with it """
        new_location = self.simulate_action(self.observations.location, desired_action)
        mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(new_location)

        # planning around mountains
        if mountain_in_range:
            # if we are within the inner bounds of the mountain, move directly outwards
            if mountain_dist <= MOUNTAIN_INNER_RANGE:
                # print("inner")
                theta = self.calculate_desired_heading(self.observations.location, mountain_in_range)
                if theta < 0:
                    theta += np.pi
                else:
                    theta -= np.pi
                desired_action = np.array([7.5, theta], dtype=np.float32)
            else:
                heading = desired_action[1]
                desired_heading = self.get_angle_away(mountain_in_range, self.observations.location, heading)
                desired_action = np.array([7.5, desired_heading], dtype=np.float32)

        _, distance_from_closest_hideout = self.get_closest_hideout(self.observations.location, self.observations.known_hideout_list + self.observations.unknown_hideout_list)
        if distance_from_closest_hideout < self.MIN_DIST_TO_HIDEOUT:
            desired_action = [0.0, 0.0]

        return desired_action

    def in_dense_forest(self,  current_location):
        dense_forest = self.terrain.forest_density_array < .4
        i = int(current_location[0])
        j = int(current_location[1])

        if np.sum(dense_forest[i - 3:i + 3, j - 3:j + 3]) > 17:
            if self.DEBUG:
                print(current_location, ' in dense forest.')
            return True
        else:
            if self.DEBUG:
                print(current_location, ' NOT in dense forest.')
            return False
    
    def calculate_desired_heading(self, start_location, end_location):
        return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])

    def get_angle_away(self, mountain_in_range, location, theta):
        # location_to_mountain_theta = self.arctan_clipped(location, mountain_in_range)
        # location_to_mountain_theta = np.arctan2(location[1] - mountain_in_range[1], location[0] - mountain_in_range[0])
        location_to_mountain_theta = np.arctan2(mountain_in_range[1] - location[1], mountain_in_range[0] - location[0])
        if -np.pi < location_to_mountain_theta < -np.pi / 2:
            theta_one = location_to_mountain_theta + np.pi / 2
            theta_two = location_to_mountain_theta + 3 * np.pi / 2
            # in bottom left quadrant, have to adjust bounds
            if theta < theta_one or theta > theta_two:
                # need to move away from mountain
                # print("move away 3")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_two
                else:
                    return theta_one
                # return clip_theta(location_to_mountain_theta - np.pi/2)
            else:
                # print("move towards 3")
                return theta
        elif np.pi / 2 < location_to_mountain_theta < np.pi:
            theta_one = location_to_mountain_theta - np.pi / 2
            theta_two = location_to_mountain_theta - 3 * np.pi / 2
            # in bottom right quadrant
            if theta > theta_one or theta < theta_two:
                # need to move away from mountain
                # print("move away 2")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_one
                else:
                    return theta_two
            else:
                # print("move towards 2")
                return theta
        else:
            theta_one = location_to_mountain_theta - np.pi / 2
            theta_two = location_to_mountain_theta + np.pi / 2
            if theta_one < theta < theta_two:
                # print("move away 14")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_one
                else:
                    return theta_two
            else:
                # print("move towards 14")
                return theta

    def update_tracking(self, detected_helicopter, detected_search_party, camera_in_range):
        if detected_helicopter or detected_search_party or camera_in_range:
            self.being_tracked_for_n_timesteps.append(1)
        else:
            self.being_tracked_for_n_timesteps.append(0)

    def calculate_evasive_angle_slow(self, current_location, hideouts):
        """
        This function will look at the fugitives current location, hideouts, and terrain, and choose a direction to go
        to evade detection (into the forest)
        :param current_location:
        :param hideouts:
        :return:
        """
        # find locations where forest is pretty dense
        dense_forest = self.terrain.forest_density_array < .4

        # check around some fixed region of the fugitive
        ran = 250
        lb_x = max(int(current_location[0] - ran), 0) # lower bound x
        ub_x = min(int(current_location[0] + ran), DIM_X) # upper bound x
        lb_y = max(int(current_location[1] - ran), 0) # lower bound y
        ub_y = min(int(current_location[1] + ran), DIM_Y) # upper bound y
        best_dist = np.inf
        candidate = None
        for i in range(lb_x, ub_x):
            for j in range(lb_y, ub_y):
                if i == current_location[0] and j== current_location[1]:
                    continue
                s = (i, j)
                dist = np.linalg.norm(s - current_location)
                # if its a patch of forest and not a one off
                if np.sum(dense_forest[i - 3:i + 3, j - 3:j + 3]) < 18:
                    continue
                if dist <= best_dist:
                    best_dist = dist
                    candidate = s

        if candidate is None:
            candidate = (1500, 1500)
        
        angle = self.calculate_desired_heading(current_location, candidate)
        if self.DEBUG:
            print("Candidate location is: ", candidate)

        return angle

    # faster implementation
    def calculate_evasive_angle(self, current_location, hideouts):
        target = np.array(current_location)
        distanced = np.sqrt((self.evasive_arr[:,0] - target[0]) ** 2 + (self.evasive_arr[:,1] - target[1]) ** 2)
        nearest_index = np.argmin(distanced)
        # print(distanced[nearest_index])

        location = self.evasive_arr[nearest_index]
        manhattan_dist = np.abs(location - target)

        if manhattan_dist[0] > 250 or manhattan_dist[1] > 250: # manhattan distance just to match previous implementation
            candidate = (1500, 1500)
        else:
            candidate = (self.evasive_arr[nearest_index] + 3)

        angle = self.calculate_desired_heading(current_location, candidate)
        return angle

    def predict(self, observation, deterministic=True):
        return (self.get_desired_action(observation), None)

class AStarOnly(AStarAdversarialAvoid):
    def __init__(self, env, max_speed=7.5, cost_coeff=1000, visualize=False):
        super().__init__(env, max_speed, cost_coeff, visualize)
    def get_desired_action(self, observation):
        self.observations.process_observation(observation)
        if self.DEBUG:
            print('t=', self.env.timesteps)
            print('fugitive current location', self.observations.location)

        desired_action = self.action_to_random_hideout(plot=False)
        # import time
        # time.sleep(1)
        if self.DEBUG:
            print('---------------------------')
        return desired_action

    def get_scaled_path(self, plot=False):
        # INFO: process the current red observation
        # self.observations.process_observation(observation)

        random_hideout = self.get_random_hideout()
        self.current_hideout_goal = random_hideout
        goal_location = random_hideout.location
        start_pos = self.env.prisoner.location
        scale_start_pos = (int(start_pos[0] / self.x_scale), int(start_pos[1] / self.y_scale))
        scale_goal_location = (int(goal_location[0] / self.x_scale), int(goal_location[1] / self.y_scale))
        path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement='8N', occupancy_cost_factor=self.cost_coeff)
        scaled_path = self.scale_a_star_path(path, start_pos, goal_location)
        self.scaled_path = generate_uniform_path(scaled_path, goal_location, total_points=10, normalized_path=False, normalized_hideout=False)
        self.actions = self.convert_path_to_actions(self.scaled_path)
        self.env.waypoints = self.scaled_path
        assert len(path) > 0, "No path found"
        if plot:
            self.gmap.plot()
            plot_path(self.scaled_path, [goal_location], point_period=1)
        return self.scaled_path.tolist()

class HeuScheduler(object):
    def __init__(self, subpolicy_num):
        super().__init__()
        self.subpolicy_num = subpolicy_num

    def step(self, high_observation):
        candidate_actions_size = 2 * self.subpolicy_num
        candidate_actions = high_observation[-candidate_actions_size:].view(-1, 2)

        # INFO: get distance to helicopter
        helicopter_relative_location = high_observation[16:18]
        dist_to_heli = torch.linalg.norm(helicopter_relative_location)

        # INFO: get distance to search party
        searchParty_relative_location = high_observation[20:22]
        dist_to_sp = torch.linalg.norm(searchParty_relative_location)

        # INFO: get distance to all hideouts
        hideout_relative_locations = high_observation[1:10].view(-1, 3)[:,1:]
        sel_goal_idx = torch.argmin(torch.linalg.norm(hideout_relative_locations.view(-1, 2), dim=1))

        if dist_to_heli < 0.1 or dist_to_sp < 0.1:
            subpolicy_idx = 3
        else:
            subpolicy_idx = sel_goal_idx

        action = candidate_actions[subpolicy_idx].clamp(-1, 1)

        return [action]


