import math
import torch
from torch import nn
import numpy as np

def sample_sequence_from_buffer(buffer, batch_size, sequence_length, observation_space, max_timesteps):
    """
    This function takes a stable baselines 3 buffer and samples a lentgh of n steps from it using timestamp from observation.
    """
    observation_shape = observation_space.shape

    batch = []
    indexes = np.random.randint(buffer.states.shape[0], size=batch_size)
    # reshape to rows, timestamps, features
    for index in indexes:
        last_observation = buffer.states[index]
        timestep = torch.round(last_observation[0] * max_timesteps).detach().cpu().numpy().astype(np.int)

        if timestep >= sequence_length - 1:
            sequence = torch.squeeze(buffer.states[index-sequence_length+1:index+1], 1)

        else:
            shape = (sequence_length-timestep-1,) + observation_shape
            empty_sequences = torch.zeros(shape).to("cuda")
            sequence = torch.squeeze(buffer.states[index-timestep:index+1], 1)
            sequence = torch.concat((empty_sequences, sequence), axis=0)

        batch.append(sequence)
        desired_shape = (sequence_length,) + observation_shape
        assert sequence.shape == desired_shape, "Wrong shape: %s, %s" % (sequence.shape, desired_shape)
    actions =  buffer.actions[indexes]
    return torch.stack(batch), torch.squeeze(actions, 1)

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # return gaussian_log_probs - torch.log(
    #     1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return gaussian_log_probs


def reparameterize(means, log_stds):
    noises = torch.randn_like(means) # return standard normal distribution sample
    us = means + noises * log_stds.exp() # e ^ (log_std)
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def asigmoid(x):
    return torch.log((x + 1e-8) / (1 + 1e-8 - x))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def construct_cost_obs_feature_map(costmap, one_batch_observations, device):
    map_width = costmap.shape[0]
    costmap = costmap[0:map_width:12, 0:map_width:12]
    map_width = costmap.shape[0]
    costmap_observation_layers = []
    batch_size = one_batch_observations.shape[0]
    for i in range(batch_size):
        observations = one_batch_observations[i, :]
        # observations.shape = (120,), 120 = 1(timestep) + 44(known camera num) * 2 + 3(hideout num) * 3(known_to_good_guys+hideout loc) + 2(prisoner location) + 2(prisoner action) + 18(fugitive_detection_of_parties) + 0(terrain)
        feature_num = 8 # (timestep, known camera, hideout known to police, hideout unknown to police, prisoner loc, prisoner_vel_norm, prisoner_vel_theta, detected_blue_loc)
        feature_len_list = [1, 88, 9, 2, 2, 18]
        # feature_final_idx_list = [1, 89, 98, 100, 102]
        obs_timestep, obs_known_camera, obs_hidden_place, obs_prisoner_loc, obs_prisoner_act, obs_detection_of_parties = torch.split(observations, feature_len_list)
        
        # feature_map = np.zeros(feature_num, map_width, map_width)
        """timestep feature"""
        feature_timestep = torch.ones((map_width, map_width)).to(device) * obs_timestep
        """camera feature"""
        known_camera_num = feature_len_list[1] // 2
        feature_known_camera = torch.zeros(map_width, map_width).to(device)
        for i in range(known_camera_num):
            camera_x_coord = to_pixel_loc(obs_known_camera[i*2], map_width)
            camera_y_coord = to_pixel_loc(obs_known_camera[i*2+1], map_width)
            feature_known_camera[camera_x_coord, camera_y_coord] = 1
        """hidden out feature"""
        hiddenout_num = feature_len_list[2] // 3
        hidden_places = torch.split(obs_hidden_place, hiddenout_num)
        feature_blue_known_hiddenout = torch.zeros(map_width, map_width).to(device)
        feature_blue_unknown_hiddenout = torch.zeros(map_width, map_width).to(device)
        for hiddenout_known_loc in hidden_places:
            hiddenout_x_coord = to_pixel_loc(hiddenout_known_loc[1], map_width)
            hiddenout_y_coord = to_pixel_loc(hiddenout_known_loc[2], map_width)
            if hiddenout_known_loc[0] == 0: # unknown?
                feature_blue_unknown_hiddenout[hiddenout_x_coord, hiddenout_y_coord] = 1
            else:
                feature_blue_known_hiddenout[hiddenout_x_coord, hiddenout_y_coord] = 1
        """prisoner location feature"""
        feature_prisoner_loc = torch.zeros(map_width, map_width).to(device)
        prisoner_loc_x_coord = to_pixel_loc(obs_prisoner_loc[0], map_width)
        prisoner_loc_y_coord = to_pixel_loc(obs_prisoner_loc[1], map_width)
        feature_prisoner_loc[prisoner_loc_x_coord, prisoner_loc_y_coord] = 1
        """prisoner action feature"""
        feature_prisoner_act_v = torch.zeros(map_width, map_width).to(device)
        feature_prisoner_act_theta = torch.zeros(map_width, map_width).to(device)
        feature_prisoner_act_v[prisoner_loc_x_coord, prisoner_loc_y_coord] = obs_prisoner_act[0]
        feature_prisoner_act_theta[prisoner_loc_x_coord, prisoner_loc_y_coord] = obs_prisoner_act[1]
        """detection_of_parties feature"""
        feature_detection_of_parties = torch.zeros(map_width, map_width).to(device)
        detection_of_parties_num = feature_len_list[5] // 3
        detection_of_parties = torch.split(obs_detection_of_parties, detection_of_parties_num)
        for detection_loc in detection_of_parties:
            if detection_loc[0] == 1: # Detected?
                detected_x_coord = to_pixel_loc(detection_loc[1], map_width)
                detected_y_coord = to_pixel_loc(detection_loc[2], map_width)
                feature_detection_of_parties[detected_x_coord, detected_y_coord] = 1
        
        costmap_observation_layers.append(torch.stack((torch.Tensor(costmap).to(device), feature_timestep, feature_known_camera, feature_blue_known_hiddenout, feature_blue_unknown_hiddenout, feature_prisoner_loc, feature_prisoner_act_v, feature_prisoner_act_theta, feature_detection_of_parties), axis=0))
    return torch.stack(costmap_observation_layers, axis=0)


def to_pixel_loc(normalized_loc, map_width):
    pixel_loc = int(normalized_loc * map_width)
    return pixel_loc

