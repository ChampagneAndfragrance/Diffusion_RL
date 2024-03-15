import matplotlib.pyplot as plt
import time
import numpy as np
import torch

from blue_bc.utils import HierScheduler


class BlueHeuristic:
    def __init__(self, env, debug=False):
        self.env = env
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.debug = debug
        self.fig, self.ax = plt.subplots()
        plt.close(self.fig)
        plt.clf()

    def reset(self):
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []

    def predict(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)
        new_detection = wrapped_blue_observation["prisoner_detected"]
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
        else:
            new_detection = (new_detection*2428).tolist()
        return self.step(new_detection)

    def step(self, new_detection):
        if new_detection is not None:
            self.detection_history.append((new_detection, self.timesteps))
            if len(self.detection_history) == 1:
                self.command_each_party("plan_path_to_loc", new_detection)
            else:
                vector = np.array(new_detection) - np.array(self.detection_history[-2][0])
                speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
                direction = np.arctan2(vector[1], vector[0])
                self.command_each_party("plan_path_to_intercept", speed, direction, new_detection)
        if self.debug:
            self.debug_plot_plans()
        # self.command_each_party("move_according_to_plan")
        # instead of commanding each party to move, grab the action that we pass into the environment

        # self.command_each_party("get_action_according_to_plan
        return self.get_each_action()

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

    @property
    def timesteps(self):
        return self.env.timesteps

    def init_pos(self):
        # a heuristic strategy to initialize position of each blue?
        pass

    def init_behavior(self):
        # initialize the behavior at the beginning before any detection is made?
        self.command_each_party("plan_path_to_random")

    def command_each_party(self, command, *args, **kwargs):
        for search_party in self.env.search_parties_list:
            getattr(search_party, command)(*args, **kwargs)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                getattr(helicopter, command)(*args, **kwargs)

    def debug_plot_arrow(self, from_x, from_y, to_x, to_y, **kwargs):
        self.ax.arrow(from_x, from_y, to_x - from_x, to_y - from_y, **kwargs)

    def debug_plot_plans(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 2428])
        self.ax.set_ylim([0, 2428])
        self.ax.set_aspect('equal')

        all_blue = self.search_parties + self.helicopters if self.env.is_helicopter_operating() else self.search_parties

        for i in range(len(self.detection_history) - 1):
            self.debug_plot_arrow(self.detection_history[i][0][0], self.detection_history[i][0][1],
                                  self.detection_history[i+1][0][0], self.detection_history[i+1][0][1],
                                  color='red', head_width=10)

        if len(self.detection_history) > 1:
            print("second_last_detection:", self.detection_history[-2])
        if len(self.detection_history) > 0:
            print("last_detection:", self.detection_history[-1])

        for blue_agent in all_blue:
            planned_path = blue_agent.planned_path
            current_loc = blue_agent.location
            for plan in planned_path:
                if self.debug:
                    print(blue_agent, "loc:", blue_agent.location, "plan:", plan)
                if plan[0] == 'l':
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='black', head_width=20)
                    current_loc = (plan[1], plan[2])
                elif plan[0] == 'd':
                    length_of_direction = 50
                    self.ax.arrow(current_loc[0], current_loc[1],
                                  plan[1] * length_of_direction, plan[2] * length_of_direction,
                                  color='pink', head_width=20)
                else:
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='orange', head_width=20)
                    current_loc = (plan[1], plan[2])
        plt.savefig("logs/temp/debug_plan_%d.png" % self.timesteps)
        plt.savefig("logs/temp/debug_plan.png")
        plt.close(self.fig)
        plt.clf()
        # input("Enter to continue")


class SimplifiedBlueHeuristicPara:
    def __init__(self, env, debug=False):
        self.env = env
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.debug = debug
        self.fig, self.ax = plt.subplots()
        plt.close(self.fig)
        plt.clf()

    def reset(self):
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []

    def predict(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. """
        # blue_obs_names = self.env.blue_obs_names
        # wrapped_blue_observation = blue_obs_names(blue_observation)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        # # check if new_detection equals [-1, -1]
        # # print("new_detection: ", new_detection)
        # if np.array_equiv(new_detection, np.array([-1, -1])):
        #     new_detection = None
        # else:
        #     new_detection = (new_detection*2428).tolist()
        if np.array_equiv(np.array(self.env.prisoner_detected_loc_history), np.array([0, 0])):
            goal_loc = np.array([0, 0])
        else:
            goal_loc = np.array(self.env.prisoner_detected_loc_history)/2428.0

        blue_locations = self.extract_blue_loc_from_partial_obs(blue_observation)
        blue_actions = self.simplified_tracking(blue_locations, goal_loc)
        
        return blue_actions

    def predict_full_observation(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. In addition, inform if the prisoner is detected now """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)

        blue_team_number = self.env.num_known_cameras + self.env.num_unknown_cameras + self.env.num_helicopters + self.env.num_search_parties
        key_start = "known_camera_0"
        key_end = "search_party_%d" % (self.env.num_search_parties - 1)
        blue_detection_n_hot = wrapped_blue_observation.get_section_include_terminals(key_start, key_end)
        # blue_detection_n_hot = blue_observation[-4-blue_team_number:-4]
        prisoner_detected = any(blue_detection_n_hot)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        new_detection = np.array(self.env.prisoner_detected_loc_history2[0:2])
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if not prisoner_detected or np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
            switch_subpolicy = False
        else:
            new_detection = (new_detection).tolist()
            switch_subpolicy = True
        actions, subpolicy_idx, speed_ratio, new_detection = self.step(new_detection)
        return actions, subpolicy_idx, speed_ratio, new_detection

    def predict_full_observation_para(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. In addition, inform if the prisoner is detected now """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)

        blue_team_number = self.env.num_known_cameras + self.env.num_unknown_cameras + self.env.num_helicopters + self.env.num_search_parties
        key_start = "known_camera_0"
        key_end = "search_party_%d" % (self.env.num_search_parties - 1)
        blue_detection_n_hot = wrapped_blue_observation.get_section_include_terminals(key_start, key_end)
        # blue_detection_n_hot = blue_observation[-4-blue_team_number:-4]
        prisoner_detected = any(blue_detection_n_hot)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        new_detection = np.array(self.env.prisoner_detected_loc_history2[0:2])
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if not prisoner_detected or np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
            switch_subpolicy = False
        else:
            new_detection = (new_detection).tolist()
            switch_subpolicy = True
        actions, subpolicy_idx = self.step(new_detection)
        return actions, subpolicy_idx

    def predict_intercept(self, blue_observation):
        last_goal = np.array(self.env.prisoner_detected_loc_history2[0:2])
        second_last_goal = np.array(self.env.prisoner_detected_loc_history2[2:4])
        not_detected_goal = np.array([-1, -1])
        blue_locations = self.extract_blue_loc_from_full_obs(blue_observation)

        if np.array_equiv(last_goal, not_detected_goal) and np.array_equiv(second_last_goal, not_detected_goal):
            goal_loc = np.array([0, 0])
            blue_actions = self.simplified_tracking(blue_locations, goal_loc)     
        elif np.array_equiv(last_goal, not_detected_goal) and (not np.array_equiv(second_last_goal, not_detected_goal)):
            goal_loc = last_goal/2428.0
            blue_actions = self.simplified_tracking(blue_locations, goal_loc)  
        elif (not np.array_equiv(last_goal, not_detected_goal)) and (not np.array_equiv(second_last_goal, not_detected_goal)):
            last_goal_loc = last_goal/2428.0
            second_last_goal_loc = second_last_goal/2428.0
            blue_actions = self.simplified_intercepting(blue_locations, last_goal_loc, second_last_goal_loc)

        # blue_locations = self.extract_blue_loc_from_partial_obs(blue_observation)
        # blue_actions = self.simplified_tracking(blue_locations, goal_loc)    

        return blue_actions  

    def extract_blue_loc_from_partial_obs(self, blue_observation):

        blue_team_num = self.env.num_search_parties + self.env.num_helicopters
        blue_locs_and_detected_loc = np.split(blue_observation, blue_team_num + 1) # hardcode the prisoner num as 1
        blue_locs = blue_locs_and_detected_loc[0:blue_team_num]

        return blue_locs

    def simplified_tracking(self, blue_locations, goal_loc):
        helicopter_actions = []
        search_party_actions = []
        for i in range(len(blue_locations)):
            if i < self.env.num_helicopters:
                helicopter_location = blue_locations[i]
                helicopter_action = self.get_helicopter_action(helicopter_location, goal_loc)
                helicopter_actions.append(np.array(helicopter_action))
            else:
                search_party_location = blue_locations[i]
                search_party_action = self.get_helicopter_action(search_party_location, goal_loc)
                search_party_action[2] = 6.5
                search_party_actions.append(np.array(search_party_action))
        blue_actions = search_party_actions + helicopter_actions

        return blue_actions

    def get_helicopter_action(self, helicopter_location, goal_loc):
        helicopter_location_np = np.array(helicopter_location)
        goal_loc_np = np.array(goal_loc)
        head_direction = goal_loc_np - helicopter_location_np
        normailized_head_direction = head_direction / (np.linalg.norm(head_direction) + 1e-10)
        speed = 127.0
        helicopter_action = normailized_head_direction.tolist() + [speed]
        return helicopter_action

    def step(self, new_detection):
        subpolicy_name = None
        subpolicy_idx = None
        speed_ratio = None
        if new_detection is not None:
            self.detection_history.append((new_detection, self.timesteps))
            if len(self.detection_history) == 1:
                subpolicy_name = "plan_path_to_loc_para"
                subpolicy_idx = np.array([2])
                speed_ratio = np.ones(self.env.num_helicopters + self.env.num_search_parties)
                self.command_each_party(subpolicy_name, speed_ratio, new_detection)
            else:
                vector = np.array(new_detection) - np.array(self.detection_history[-2][0])
                speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
                direction = np.arctan2(vector[1], vector[0])
                subpolicy_name = "plan_path_to_intercept_para"
                subpolicy_idx = np.array([4])
                speed_ratio = np.array([1.0 for _ in range(self.env.num_search_parties)] + [0.1])
                self.command_each_party(subpolicy_name, speed_ratio, speed, direction, new_detection)
        if self.debug:
            self.debug_plot_plans()
        # self.command_each_party("move_according_to_plan")
        # instead of commanding each party to move, grab the action that we pass into the environment

        # self.command_each_party("get_action_according_to_plan
        return self.get_each_action(), subpolicy_idx, speed_ratio, np.array(new_detection)

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan_para())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan_para())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

    @property
    def timesteps(self):
        return self.env.timesteps

    def init_pos(self):
        # a heuristic strategy to initialize position of each blue?
        pass

    def init_behavior(self):
        # initialize the behavior at the beginning before any detection is made?
        speed_ratio = np.ones(self.env.num_helicopters + self.env.num_search_parties)
        self.command_each_party("plan_path_to_random_para", speed_ratio)
        # self.command_each_party("plan_path_to_stop")

    def command_each_party(self, command, speed_ratio, *args, **kwargs):
        blue_ag_idx = 0
        for search_party in self.env.search_parties_list:
            getattr(search_party, command)(speed_ratio[blue_ag_idx], *args, **kwargs)
            blue_ag_idx = blue_ag_idx + 1

        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                getattr(helicopter, command)(speed_ratio[blue_ag_idx], *args, **kwargs)
                blue_ag_idx = blue_ag_idx + 1

    def debug_plot_arrow(self, from_x, from_y, to_x, to_y, **kwargs):
        self.ax.arrow(from_x, from_y, to_x - from_x, to_y - from_y, **kwargs)

    def debug_plot_plans(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 2428])
        self.ax.set_ylim([0, 2428])
        self.ax.set_aspect('equal')

        all_blue = self.search_parties + self.helicopters if self.env.is_helicopter_operating() else self.search_parties

        for i in range(len(self.detection_history) - 1):
            self.debug_plot_arrow(self.detection_history[i][0][0], self.detection_history[i][0][1],
                                  self.detection_history[i+1][0][0], self.detection_history[i+1][0][1],
                                  color='red', head_width=10)

        if len(self.detection_history) > 1:
            print("second_last_detection:", self.detection_history[-2])
        if len(self.detection_history) > 0:
            print("last_detection:", self.detection_history[-1])

        for blue_agent in all_blue:
            planned_path = blue_agent.planned_path
            current_loc = blue_agent.location
            for plan in planned_path:
                if self.debug:
                    print(blue_agent, "loc:", blue_agent.location, "plan:", plan)
                if plan[0] == 'l':
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='black', head_width=20)
                    current_loc = (plan[1], plan[2])
                elif plan[0] == 'd':
                    length_of_direction = 50
                    self.ax.arrow(current_loc[0], current_loc[1],
                                  plan[1] * length_of_direction, plan[2] * length_of_direction,
                                  color='pink', head_width=20)
                else:
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='orange', head_width=20)
                    current_loc = (plan[1], plan[2])
        plt.savefig("logs/temp/debug_plan_%d.png" % self.timesteps)
        plt.savefig("logs/temp/debug_plan.png")
        plt.close(self.fig)
        plt.clf()
        # input("Enter to continue")


class HierRLBlue:
    def __init__(self, env, maddpg, device, debug=False):
        self.env = env
        self.maddpg = maddpg
        # self.low_policy =low_policy
        # self.high_policy = high_policy
        # self.policy = high_policy
        self.device = device
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = [([1214, 1214], 0)]
        self.debug = debug
        self.fig, self.ax = plt.subplots()
        self.t = 0
        self.t_limit = 50
        plt.close(self.fig)
        plt.clf()

    def reset(self):
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.t = 0
        self.detection_history = [([1214, 1214], 0)]

    def new_detection(self, blue_observation):
        """ Inform the heuristic about the observation of the blue agents. In addition, inform if the prisoner is detected now """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)

        blue_team_number = self.env.num_known_cameras + self.env.num_unknown_cameras + self.env.num_helicopters + self.env.num_search_parties
        key_start = "known_camera_0"
        key_end = "search_party_%d" % (self.env.num_search_parties - 1)
        blue_detection_n_hot = wrapped_blue_observation.get_section_include_terminals(key_start, key_end)
        # blue_detection_n_hot = blue_observation[-4-blue_team_number:-4]
        prisoner_detected = any(blue_detection_n_hot)
        return prisoner_detected

    def predict_full_observation(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. In addition, inform if the prisoner is detected now """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)

        blue_team_number = self.env.num_known_cameras + self.env.num_unknown_cameras + self.env.num_helicopters + self.env.num_search_parties
        key_start = "known_camera_0"
        key_end = "search_party_%d" % (self.env.num_search_parties - 1)
        blue_detection_n_hot = wrapped_blue_observation.get_section_include_terminals(key_start, key_end)
        # blue_detection_n_hot = blue_observation[-4-blue_team_number:-4]
        prisoner_detected = any(blue_detection_n_hot)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        new_detection = np.array(self.env.prisoner_detected_loc_history2[0:2])
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if not prisoner_detected or np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
            switch_subpolicy = False
        else:
            new_detection = (new_detection).tolist()
            self.detection_history.append((new_detection, self.timesteps))
            switch_subpolicy = True

        subpolicy, paras = self.maddpg.step(torch.Tensor(blue_observation).to(self.device), explore=True)
        hier_scheduler = HierScheduler(self.env, subpolicy, paras, new_detection, self.detection_history, self.timesteps)
        hier_scheduler.plan_path()
        actions = hier_scheduler.get_each_action()
        # actions, subpolicy_idx, speed_ratio, new_detection = self.step(new_detection)
        return actions, new_detection, torch.cat((subpolicy.view(-1), paras.view(-1)))

    def predict_full_observation_period(self, blue_observation: np.ndarray):
        update_high_policy_flag = False
        update_low_policy_flag = False
        high_policy_output = None
        high_policy_input = None
        low_policy_output = None
        low_policy_input = None
        # prisoner_detected = self.new_detection(blue_observation)
        last_predicted_detection = np.array(self.env.predicted_locations_from_last_two_detections) * 2428

        # if prisoner_detected is True:
        #     new_detection = np.array(self.env.last_k_fugitive_detections[-1][-2:]) * 2428
        #     self.detection_history.append((new_detection, self.timesteps / self.max_timesteps))
        # else:
        #     new_detection = None

        hier_scheduler = HierScheduler(self.env, self.timesteps) 
        if self.t == 0:
            """update high level policy when reaching time limit"""
            high_policy_input = blue_observation
            self.subpolicies = self.maddpg.step_high(torch.Tensor(high_policy_input).to(self.device), explore=True)
            update_high_policy_flag = True
            # hier_scheduler.update(self.subpolicies, self.paras, new_detection)
            # high_policy_period_start_flag = True
        # elif self.t == (self.t_limit - 1):
        #     high_policy_period_end_flag = True
        """generate new parameter every step"""
        if True:
            low_policy_input = np.concatenate((np.array(blue_observation), self.subpolicies.cpu().detach().numpy()), axis=-1)
            self.paras = self.maddpg.step_low(torch.Tensor(low_policy_input).to(self.device), explore=True)
            low_policy_output = self.paras.view(-1)
            update_low_policy_flag = True
            hier_scheduler.update(self.subpolicies, self.paras, last_predicted_detection)
            hier_scheduler.plan_path()
        actions = hier_scheduler.get_each_action()
        self.t = (self.t + 1) % self.t_limit

        high_policy_output = self.subpolicies.view(-1)
        
        # high_critic_input = np.concatenate((high_policy_input, high_policy_output))
        
        
        # low_critic_input = np.concatenate((low_policy_input, low_policy_output))
        

        return update_high_policy_flag, high_policy_output, high_policy_input, update_low_policy_flag, low_policy_output, low_policy_input, actions

    def predict_full_observation_random(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. In addition, inform if the prisoner is detected now """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)

        blue_team_number = self.env.num_known_cameras + self.env.num_unknown_cameras + self.env.num_helicopters + self.env.num_search_parties
        key_start = "known_camera_0"
        key_end = "search_party_%d" % (self.env.num_search_parties - 1)
        blue_detection_n_hot = wrapped_blue_observation.get_section_include_terminals(key_start, key_end)
        # blue_detection_n_hot = blue_observation[-4-blue_team_number:-4]
        prisoner_detected = any(blue_detection_n_hot)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        new_detection = np.array(self.env.prisoner_detected_loc_history2[0:2])
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if not prisoner_detected or np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
            switch_subpolicy = False
        else:
            new_detection = (new_detection).tolist()
            self.detection_history.append((new_detection, self.timesteps))
            switch_subpolicy = True

        subpolicy, paras, logp = self.policy(torch.Tensor(blue_observation).to(self.device))
        hier_scheduler = HierScheduler(self.env, subpolicy, paras, new_detection, self.detection_history, self.timesteps)
        hier_scheduler.plan_path()
        actions = hier_scheduler.get_each_action()
        # actions, subpolicy_idx, speed_ratio, new_detection = self.step(new_detection)
        return actions, new_detection, torch.cat((subpolicy.view(-1), paras.view(-1))), logp

    def step(self, new_detection):
        subpolicy_name = None
        subpolicy_idx = None
        speed_ratio = None
        if new_detection is not None:
            self.detection_history.append((new_detection, self.timesteps))
            if len(self.detection_history) == 1:
                subpolicy_name = "plan_path_to_loc_para"
                subpolicy_idx = np.array([2])
                speed_ratio = np.ones(self.env.num_helicopters + self.env.num_search_parties)
                self.command_each_party(subpolicy_name, speed_ratio, new_detection)
            else:
                vector = np.array(new_detection) - np.array(self.detection_history[-2][0])
                speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
                direction = np.arctan2(vector[1], vector[0])
                subpolicy_name = "plan_path_to_intercept_para"
                subpolicy_idx = np.array([4])
                speed_ratio = np.array([1.0 for _ in range(self.env.num_search_parties)] + [0.1])
                self.command_each_party(subpolicy_name, speed_ratio, speed, direction, new_detection)
        if self.debug:
            self.debug_plot_plans()
        # self.command_each_party("move_according_to_plan")
        # instead of commanding each party to move, grab the action that we pass into the environment

        # self.command_each_party("get_action_according_to_plan
        return self.get_each_action(), subpolicy_idx, speed_ratio, np.array(new_detection)

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan_para())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan_para())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

    @property
    def timesteps(self):
        return self.env.timesteps

    def init_pos(self):
        # a heuristic strategy to initialize position of each blue?
        pass

    def init_behavior(self):
        # initialize the behavior at the beginning before any detection is made?
        speed_ratio = np.ones(self.env.num_helicopters + self.env.num_search_parties)
        self.command_each_party("plan_path_to_random_para", speed_ratio)
        # self.command_each_party("plan_path_to_stop")

    def command_each_party(self, command, speed_ratio, *args, **kwargs):
        blue_ag_idx = 0
        for search_party in self.env.search_parties_list:
            getattr(search_party, command)(speed_ratio[blue_ag_idx], *args, **kwargs)
            blue_ag_idx = blue_ag_idx + 1

        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                getattr(helicopter, command)(speed_ratio[blue_ag_idx], *args, **kwargs)
                blue_ag_idx = blue_ag_idx + 1

    def debug_plot_arrow(self, from_x, from_y, to_x, to_y, **kwargs):
        self.ax.arrow(from_x, from_y, to_x - from_x, to_y - from_y, **kwargs)

    def debug_plot_plans(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 2428])
        self.ax.set_ylim([0, 2428])
        self.ax.set_aspect('equal')

        all_blue = self.search_parties + self.helicopters if self.env.is_helicopter_operating() else self.search_parties

        for i in range(len(self.detection_history) - 1):
            self.debug_plot_arrow(self.detection_history[i][0][0], self.detection_history[i][0][1],
                                  self.detection_history[i+1][0][0], self.detection_history[i+1][0][1],
                                  color='red', head_width=10)

        if len(self.detection_history) > 1:
            print("second_last_detection:", self.detection_history[-2])
        if len(self.detection_history) > 0:
            print("last_detection:", self.detection_history[-1])

        for blue_agent in all_blue:
            planned_path = blue_agent.planned_path
            current_loc = blue_agent.location
            for plan in planned_path:
                if self.debug:
                    print(blue_agent, "loc:", blue_agent.location, "plan:", plan)
                if plan[0] == 'l':
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='black', head_width=20)
                    current_loc = (plan[1], plan[2])
                elif plan[0] == 'd':
                    length_of_direction = 50
                    self.ax.arrow(current_loc[0], current_loc[1],
                                  plan[1] * length_of_direction, plan[2] * length_of_direction,
                                  color='pink', head_width=20)
                else:
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='orange', head_width=20)
                    current_loc = (plan[1], plan[2])
        plt.savefig("logs/temp/debug_plan_%d.png" % self.timesteps)
        plt.savefig("logs/temp/debug_plan.png")
        plt.close(self.fig)
        plt.clf()
        # input("Enter to continue")

