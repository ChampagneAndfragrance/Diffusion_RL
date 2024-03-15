from fugitive_policies.a_star_policy import AStarPolicy
from fugitive_policies.dynamic_window_approach import dwa_local_planner
from fugitive_policies.base_policy import Observation
from fugitive_policies.a_star.gridmap import OccupancyGridMap
from fugitive_policies.a_star.a_star import a_star
import numpy as np

class AStarLocalPlanner(AStarPolicy):
    def __init__(self, env, max_speed=7.5, cost_coeff=1000, visualize=False):
        super().__init__(env, max_speed, cost_coeff, visualize)
        self.local_planner = dwa_local_planner()
        self.global_plan_refresh_period = 239
        self.local_plan_refresh_period = 5
        self.red_action = np.array([15.0, np.pi])
        self.u = None
        self.global_path = None
        self.goal = None

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

    def predict(self, observation, **kwargs):
        # INFO: Get the blue locations and velocities if red detects the blue
        red_detect_blue_id, detected_blue_states = self.env.get_detected_blue_states()
        new_obstacle_flag = False
        if len(detected_blue_states) != 0:
            obstacle_locs_seen = [[detected_blue_states[i][0][0]*self.env.dim_x, detected_blue_states[i][0][1]*self.env.dim_y, 80] for i in range(len(detected_blue_states))]
            obstacle_id_seen = red_detect_blue_id
            for obstacle_id in obstacle_id_seen:
                if obstacle_id not in self.local_planner.ob_id or obstacle_id < 3:
                    new_obstacle_flag = True
                else:
                    new_obstacle_flag = False
        else:
            obstacle_locs_seen = []
            obstacle_id_seen = []

        # INFO: Update the global plan every global_plan_refresh_period steps
        if self.env.timesteps % self.global_plan_refresh_period == 0:
            # update the global path
            # global_path, goal = red_policy.generate_path_only(incremental_dataset, n_samples=10, plot=True)
            path, self.goal = self.get_global_path(observation)
            self.global_path = self.scale_paths(path)
            self.local_planner.update_global_path(self.global_path)

        # INFO: Update the local plan every local_plan_refresh_period steps
        if self.env.timesteps % self.local_plan_refresh_period == 0 or new_obstacle_flag:
            x = self.get_prisoner_state()
            if new_obstacle_flag:
                self.local_planner.add_to_obstacle(obstacle_locs_seen, obstacle_id_seen)
            # update the local trajectory
            self.u, trajectory = self.local_planner.dwa_control(x, self.goal, self.global_path) # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
            # print("Show the velocity: ", u)
        red_actions = np.array([self.u[0], self.u[1]])
        self.red_action = red_actions
        return [red_actions]
