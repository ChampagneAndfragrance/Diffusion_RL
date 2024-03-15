import numpy as np

def transform_obs(self, obs, obs_names, num_known_cameras, num_unknown_cameras, num_helicopters, num_search_parties, num_known_hideouts, num_unknown_hideouts, total_agents_num):
    """ This function creates three numpy arrays, the first representing all the agents,
    the second representing the hideouts, and the third the timestep"""
    obs_names = obs_names
    obs_named = obs_names(obs)
    

    names = [[num_known_cameras, 'known_camera_', 'known_camera_loc_'], 
            [num_unknown_cameras, 'unknown_camera_', 'unknown_camera_loc_'],
            [num_helicopters, 'helicopter_', 'helicopter_location_'],
            [num_search_parties, 'search_party_', 'search_party_location_']]

    
    # (detect bool, x_loc, y_loc)
    gnn_obs = np.zeros((total_agents_num, 3))
    j = 0
    for num, detect_name, location_name in names:
        for i in range(num):
            detect_key = f'{detect_name}{i}'
            loc_key = f'{location_name}{i}'
            gnn_obs[j, 0] = obs_named[detect_key]
            gnn_obs[j, 1:] = obs_named[loc_key]
            j += 1

    timestep = obs_named['time']

    hideouts = np.zeros((num_known_hideouts, 2))
    for i in range(num_known_hideouts):
        key = f'hideout_loc_{i}'
        hideouts[i, :] = obs_named[key]
    hideouts = hideouts.flatten()

    num_agents = np.array(total_agents_num)

    return gnn_obs, hideouts, timestep, num_agents

def format_gnn_obs(agent_obs, hideouts, detected_location, agent_dict, timestep_obs, num_agents, detected_location_bool=True, timestep_bool=True):
    max_agent_size = len(agent_obs)
    agent_obs = np.squeeze(agent_obs)
    # [timesteps, agents, 3]
    # print(agent_obs.shape)
    num_agents = agent_obs.shape[1]

    if detected_location_bool:
        detected_bools = agent_obs[:, :, 0]
        detected_agent_locs = np.einsum("ij,ik->ijk", detected_bools, detected_location) # timesteps, agents, 2
        agent_obs = np.concatenate((agent_obs, detected_agent_locs), axis=2)

    timesteps = timestep_obs
    if timestep_bool:
        t = np.expand_dims(timesteps, axis=1)
        t = np.repeat(t, num_agents, axis=1)
        # print(t.shape)
        agent_obs = np.concatenate((agent_obs, t), axis=2)

    agent_obs = np.pad(agent_obs, ((0, 0), (0, max_agent_size - num_agents), (0, 0)), 'constant')

    num_timesteps = agent_obs.shape[0]
    one_hots = _create_one_hot_agents(agent_dict, num_timesteps)
    one_hots = np.pad(one_hots, ((0, 0), (0, max_agent_size - num_agents), (0, 0)), 'constant')
    return agent_obs

def _create_one_hot_agents(agent_dict, timesteps):
    one_hot_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    agents = [agent_dict["num_known_cameras"] + agent_dict["num_unknown_cameras"],
                agent_dict["num_helicopters"], agent_dict["num_search_parties"]]
    a = np.repeat(one_hot_base, agents, axis=0) # produce [num_agents, 3] (3 for each in one-hot)
    # one_hot = np.repeat(np.expand_dims(a, 0), self.sequence_length, axis=0) # produce [seq_len, num_agents, 3]
    one_hot = np.repeat(np.expand_dims(a, 0), timesteps, axis=0) # produce [timesteps, num_agents, 3]
    return one_hot
