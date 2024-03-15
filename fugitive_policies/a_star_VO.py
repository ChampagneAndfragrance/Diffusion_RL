from fugitive_policies.a_star_policy import AStarPolicy
from fugitive_policies.dynamic_window_approach import dwa_local_planner
from fugitive_policies.velocity_obstacle_approach import vo_local_planner
from fugitive_policies.base_policy import Observation
from fugitive_policies.a_star.gridmap import OccupancyGridMap
from fugitive_policies.a_star.a_star import a_star
import numpy as np

class AStarVO(AStarPolicy):
    def __init__(self, env, max_speed=7.5, cost_coeff=1000, visualize=False):
        super().__init__(env, max_speed, cost_coeff, visualize)
        self.local_planner = vo_local_planner(agent_radius=150.0, obstacle_radius=150.0, time_horizon=15.0)
        self.global_plan_refresh_period = 239
        self.local_plan_refresh_period = 5
        self.red_action = np.array([15.0, np.pi])
        self.u = None
        self.global_path = None
        self.goal = None

        # INFO: For VO
        self.prev_mode = "vo"
        self.curr_mode = "a_star"
        self.executing_VO = False
        self.switch_mode_flag = False
        self.prev_vo_empty = False

    def reset(self):
        self.u = None
        self.global_path = None
        self.goal = None
        # INFO: For VO
        self.prev_mode = "vo"
        self.curr_mode = "a_star"
        self.executing_VO = False
        self.switch_mode_flag = False
        self.prev_vo_empty = False
        
    def get_prisoner_state(self):
        self.prisoner_state = [self.env.prisoner.location[0], self.env.prisoner.location[1], self.red_action[1], self.red_action[0], self.red_action[1]]
        return self.prisoner_state

    def scale_paths(self, path):
        """ Converts list of points on path to list of actions (speed, thetas)
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        plan = []
        currentpos = path[0]
        for nextpos in path[1:]:
            p = self.get_path_between_two_points(currentpos, nextpos)
            currentpos = nextpos
            plan.extend(p)
        return plan

    def get_path_between_two_points(self, startpos, endpos):
        """ Returns list of actions (speed, thetas) to traverse between two points.
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        currentpos = startpos
        # actions = []
        poses = []
        while np.array_equal(currentpos, endpos) == False:
            dist = (np.linalg.norm(np.asarray(currentpos) - np.asarray(endpos)))
            speed = min(dist, self.max_speed)
            theta = np.arctan2(endpos[1] - currentpos[1], endpos[0] - currentpos[0])
            action = np.array([speed, theta], dtype=np.float32)
            # actions.append(action)
            currentpos = self.simulate_action(currentpos, action)
            poses.append(currentpos/self.env.dim_x)

        return poses

    def get_global_path(self, observation, goal='closest', deterministic=True, plot=False):
        self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
            goal_location = closest_unknown_hideout.location
            
            start_pos = list(map(int, self.observations.location))
            scale_start_pos = (int(start_pos[0] / self.x_scale), int(start_pos[1] / self.y_scale))
            scale_goal_location = (int(goal_location[0] / self.x_scale), int(goal_location[1] / self.y_scale))
            path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement='8N', occupancy_cost_factor=self.cost_coeff)
            scaled_path = self.scale_a_star_path(path, start_pos, goal_location)
            # self.actions = self.convert_path_to_actions(scaled_path)
        # return [self.actions.pop(0)]
        return scaled_path, goal_location

    def get_scaled_path(self, plot=True):
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
        self.actions = self.convert_path_to_actions(scaled_path)
        assert len(path) > 0, "No path found"
        return scaled_path, goal_location

    def blue_in_range(self, agent_position, obstacle_position):
        relative_locations = agent_position - obstacle_position
        relative_distances = np.linalg.norm(relative_locations, axis=1)
        blue_in_range = (relative_distances < np.array([500, 1500]))
        return blue_in_range

    def predict(self, observation, **kwargs):
        # self.observations.process_observation(observation)
        agent_position = np.array(self.env.get_prisoner_location())
        agent_velocity = np.array(self.env.get_prisoner_velocity())
        obstacle_positions = np.vstack(self.env.get_blue_locations())
        obstacle_velocities = np.vstack(self.env.get_blue_velocities())
        # INFO: Decide if red is in a dangerous situation
        blue_in_range = self.blue_in_range(agent_position, obstacle_positions)

        if not np.any(blue_in_range):
            self.curr_mode = "a_star"
            # INFO: replan an A* path if it transits from VO mode
            if self.prev_mode == "vo":
                scaled_path, goal_location = self.get_scaled_path()
            self.prev_mode = "a_star"
            return [self.actions.pop(0)]

        if np.any(blue_in_range):
            self.curr_mode = "vo"
            vo = self.local_planner.compute_velocity_obstacle(agent_position, agent_velocity, obstacle_positions, obstacle_velocities) 
            if vo == []:
                # INFO: if there will not be potential collision, continue executing planned A* path
                self.prev_mode = "vo"
                if not self.prev_vo_empty:
                    scaled_path, goal_location = self.get_scaled_path()
                self.prev_vo_empty = True
                return [self.actions.pop(0)]
            else:
                self.prev_mode = "vo"
                self.prev_vo_empty = False
                return self.local_planner.sample_safe_velocity(vo, speed_limit=self.max_speed)

            

