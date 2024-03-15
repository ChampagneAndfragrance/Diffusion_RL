import matplotlib.pyplot as plt
import time
import numpy as np
from blue_bc.utils import blue_obs_type_from_estimator, get_modified_blue_obs, Estimator, split_directions_to_direction_speed
from blue_bc.maddpg_filtering import MADDPGFiltering, AttentionMADDPGFiltering
from blue_bc.config_loader import config_loader
import copy
import os
import torch
from torch.autograd import Variable

class BlueRandom:
    def __init__(self, env, debug=False):
        self.env = env
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.debug = False
        if debug:
            self.fig, self.ax = plt.subplots()
            plt.close(self.fig)
            plt.clf()

    def predict(self, blue_obs):
         # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(self.get_random_action(self.env.search_party_speed))
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(self.get_random_action(self.env.helicopter_speed))
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions      

    def get_random_action(self, speed_limit):
        direction_vector = np.random.uniform(low=-1, high=1, size=2) 
        speed = np.random.uniform(low=0, high=speed_limit, size=1) 
        normalized_direction_vector = direction_vector / (np.linalg.norm(direction_vector) + 1e-16)
        action = np.array([normalized_direction_vector[0], normalized_direction_vector[1], speed])
        return action

class BlueHeuristic:
    def __init__(self, env, debug=False):
        self.env = env
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.debug = debug
        if debug:
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

class QuasiEED:
    def __init__(self, env, debug=False):
        self.env = env
        # TODO: Code for the previous blue location
        self.prev_x = np.stack([np.array(ag.location) / env.dim_x for ag in env.search_parties_list+env.helicopters_list], axis=0)
        self.init_x = np.stack([np.array(ag.location) / env.dim_x for ag in env.search_parties_list+env.helicopters_list], axis=0)
        self.max_vel_mat = np.array([env.search_party_speed for _ in range(env.num_search_parties)] + [env.helicopter_speed for _ in range(env.num_helicopters)]) / env.dim_x
        self.prev_v = np.zeros((len(env.search_parties_list+env.helicopters_list), 2))
        self.v = np.zeros((len(env.search_parties_list+env.helicopters_list), 2))
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.agent_num = env.num_helicopters + env.num_search_parties
        self.debug = debug
        if debug:
            self.fig, self.ax = plt.subplots()
            plt.close(self.fig)
            plt.clf()

    def init_behavior(self):
        return 

    def reset(self):
        return
        
    def predict(self, blue_observation: np.ndarray):
        actions = []
        # EED initialization at the start of each episode
        a_Rmin, a_Rmax, d, delta = 0.002, 0.005, 1, 0.0001
        a_R = 0.5 * (a_Rmin + a_Rmax)
        omega, c = 1, 10

        if self.env.is_detected:
            if a_R > a_Rmin:
                a_R = a_R - delta
            else:
                pass
        else:
            if a_R < a_Rmax:
                a_R = a_R + delta
            else:
                pass
        # calculate the repulsion velocity
        v_rep = np.zeros((self.agent_num, 2))
        # for i, x_i in enumerate(x):
        #     for j, x_j in enumerate(x):
        #         if i != j:
        #             r_ij = x_j - x_i
        #             v_rep[i] = v_rep[i] + (-(a_R / r_ij) ** d)
        x = np.stack([np.array(ag.location) / self.env.dim_x for ag in self.env.search_parties_list+self.env.helicopters_list], axis=0)
        
        for i, (x_i, sa) in enumerate(zip(x, self.env.search_parties_list+self.env.helicopters_list)):
            # calculate the N_best
            if self.env.is_detected:
                N_best = np.array(self.env.prisoner.location) / self.env.dim_x
                omega = 0
                c = 1
            else:
                N_best = x_i
                omega = 1
                c = 0
            # calculate the v_i
            approch_vec = (N_best - x_i)
            if np.linalg.norm(approch_vec) >= self.max_vel_mat[i]:
                approch_vel = approch_vec / np.linalg.norm(approch_vec)
            else:
                approch_vel = approch_vec / self.max_vel_mat[i]
                # approch_vel = approch_vec
            v_pso_i = omega * self.v[i] + c * approch_vel
            self.v[i] = v_pso_i + v_rep[i]
            self.v[i] = self.v[i] / np.linalg.norm(self.v[i]) if np.linalg.norm(self.v[i]) >= 1 else self.v[i]
            actions.append(np.concatenate((self.v[i] / (np.linalg.norm(self.v[i])+1e-10), np.linalg.norm(self.v[i], keepdims=True)*sa.speed), axis=-1))
        return actions

class RLWrapper:
    def __init__(self, env, path) -> None:
        self.env = env
        self.config = config_loader(os.path.join(path, 'parameter/parameters_network.yaml'))
        self.device = "cuda" if self.config["environment"]["cuda"] else "cpu"
        self.maddpg = self.construct_MADDPG()
        


    def construct_MADDPG(self):
        self.blue_obs_type = blue_obs_type_from_estimator(self.config["environment"]["estimator"], Estimator) 
        blue_observation = get_modified_blue_obs(self.env, self.blue_obs_type, Estimator)
        # filtering_input = copy.deepcopy(env.get_t_d1_d0_vel())
        filtering_input = copy.deepcopy(self.env.get_tInterval_lastDetection())
        prisoner_loc = copy.deepcopy(self.env.get_prisoner_location())
        """Load the model"""
        agent_num = self.env.num_helicopters + self.env.num_search_parties
        action_dim_per_agent = 2
        # filtering_input_dims = [[len(filtering_input), len(filtering_input[0])] for i in range(agent_num)]
        filtering_input_dims = [[len(filtering_input)] for i in range(agent_num)]
        obs_dims=[blue_observation[i].shape[0] for i in range(agent_num)]
        ac_dims=[action_dim_per_agent for i in range(agent_num)]
        loc_dims = [len(prisoner_loc) for i in range(agent_num)]
        obs_ac_dims = [obs_dims, ac_dims]
        obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
        maddpg = MADDPGFiltering(
                            filtering_model_config = self.config["train"]["filtering_model_config"],
                            filtering_model_path = self.config["train"]["filtering_model_path"],
                            agent_num = agent_num, 
                            num_in_pol = blue_observation[0].shape[0], 
                            num_out_pol = action_dim_per_agent, 
                            num_in_critic = (blue_observation[0].shape[0] + action_dim_per_agent) * agent_num, 
                            discrete_action = False, 
                            gamma=self.config["train"]["gamma"], tau=self.config["train"]["tau"], critic_lr=self.config["train"]["critic_lr"], policy_lr=self.config["train"]["policy_lr"], filter_lr=self.config["train"]["filter_lr"], hidden_dim=self.config["train"]["hidden_dim"], device=self.device)

        maddpg.init_from_save(os.path.join(self.config["environment"]["dir_path"],'model', '0.pth'))
        maddpg.prep_rollouts(device=self.device)
        explr_pct_remaining = max(0, self.config["train"]["n_exploration_eps"] - 0) / self.config["train"]["n_exploration_eps"]
        maddpg.scale_noise(self.config["train"]["final_noise_scale"] + (self.config["train"]["init_noise_scale"] - self.config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()
        return maddpg

    def predict(self, blue_observation):
        # # INFO: run episode
        # t = t + 1
        blue_observation = get_modified_blue_obs(self.env, self.blue_obs_type, Estimator)
        torch_obs = [Variable(torch.Tensor(blue_observation[i]), requires_grad=False).to(self.device) for i in range(self.maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]
        # gnn_sequence_obs = env.get_gnn_sequence()
        # last_two_detections_vel = env.get_t_d1_d0_vel()
        last_two_detections_vel = self.env.get_t_init_le_timeInterval()
        mlp_input_tensors = torch.Tensor(last_two_detections_vel).unsqueeze(0).to(self.device)
        # get actions as torch Variables
        torch_agent_actions = self.maddpg.step(torch_obs, mlp_input_tensors, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        splited_agent_actions = split_directions_to_direction_speed(np.concatenate(agent_actions), self.env.search_party_speed)
        return splited_agent_actions
        
class SimplifiedBlueHeuristic:
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
        """ Inform the heuristic about the observation of the blue agents. """
        # blue_obs_names = self.env.blue_obs_names
        # wrapped_blue_observation = blue_obs_names(blue_observation)
        # new_detection = wrapped_blue_observation["prisoner_detected"]
        new_detection = np.array(self.env.prisoner_detected_loc_history2[0:2])
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
        else:
            new_detection = (new_detection).tolist()
        return self.step(new_detection)

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

    def extract_blue_loc_from_full_obs(self, full_blue_observation):
        pass

    def simplified_intercepting(self, blue_locations, last_goal_loc, second_last_goal_loc):
        pass

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
        self.command_each_party("plan_path_to_random_para")

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
