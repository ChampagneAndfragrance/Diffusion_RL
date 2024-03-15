from tqdm import tqdm
import numpy as np
import torch

from buffer import Buffer
from utils import save_video


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, blue_heuristic, red_policy, buffer_size, device, seed=0):

    """Start to Revise"""
    t = 0
    total_return = 0.0
    num_episodes = 0
    episode_return = 0.0
    red_full_observation, red_partial_observation = env.reset()

    buffer = Buffer(
        buffer_size=buffer_size,
        # state_shape=env.blue_partial_observation_space.shape,
        # state_shape=env.partial_observation_space_shape,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )
    """End Revising"""

    imgs = []
    filled_buffer_len = 0
    # for _ in tqdm(range(1, buffer_size + 1)):
    while filled_buffer_len < buffer_size:
        # INFO: run episode
        t = t + 1

        """Full Red Obs"""
        try:
            red_action = red_policy.predict(red_full_observation)
        except:
            done = True
        # red_action = red_policy.predict(red_full_observation)
        # print("helicopter_actions = ", blue_actions[5])
        # print("blue_actions = ", blue_actions)
        next_red_full_observation, reward, done, _, next_partial_observation = env.step(red_action[0])
        # next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(blue_actions)
        # next_blue_obs = env.get_blue_observation()
        mask = False if t == env.max_timesteps else done

        """Full Red Obs"""
        buffer.append(red_full_observation, normalized_vel_theta(red_action[0]), reward, mask, next_red_full_observation)
        """Partial Red Obs"""
        # buffer.append(red_partial_observation, normalized_vel_theta(red_action[0]), reward, mask, next_partial_observation)
        
        filled_buffer_len = filled_buffer_len + 1
        print("Now the buffer filling process completes %f" % (filled_buffer_len/buffer_size))
        # print("blue_partial_observation = ", blue_observation)
        # print("next_blue_partial_observation = ", next_blue_observation)
        # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
        # print("helicopter_action_theta_speed", blue_actions[5])
        episode_return += reward

        red_full_observation = next_red_full_observation
        red_partial_observation = next_partial_observation

        game_img = env.render('Policy', show=False, fast=True)
        imgs.append(game_img)
        if done:
            num_episodes += 1
            total_return += episode_return

            red_full_observation, red_partial_observation = env.reset()
            t = 0
            episode_return = 0.0
        
        """End Revising"""
    save_video(imgs, "/home/wu/GatechResearch/Zixuan/PrisonerEscape/buffers/video/Red_LSTM_Exp" + "/%d.mp4" % buffer_size, fps=10)
    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

def normalized_vel_theta(red_action):
    norm_vel = red_action[0] / 15.0
    norm_theta = red_action[1] / np.pi
    return np.array([norm_vel, norm_theta])

def to_angle_speed(blue_actions):
    for idx in range(len(blue_actions)):
        if idx < 5:
            theta = np.arctan2(blue_actions[idx][1], blue_actions[idx][0])
            if theta < 0:
                theta = np.pi * 2 + theta
            theta = (theta / (2 * np.pi)) * 2 + (-1)
            speed = (blue_actions[idx][2] / 6.5) * 2 + (-1)
            blue_actions[idx] = np.array([theta, speed])
            # if blue_actions[idx][2] == 0:
            #     blue_actions[idx] = np.array([-1, -1])
        elif idx < 6:
            theta = np.arctan2(blue_actions[idx][1], blue_actions[idx][0])
            if theta < 0:
                theta = np.pi * 2 + theta
            theta = (theta / (2 * np.pi)) * 2 + (-1)
            speed = (blue_actions[idx][2] / 127) * 2 + (-1)
            blue_actions[idx] = np.array([theta, speed])   
            # if blue_actions[idx][2] == 0:
            #     blue_actions[idx] = np.array([-1, -1])      
    return blue_actions

def to_velocity_vector(blue_actions):
    velocities = []
    for idx in range(len(blue_actions)):
        direction_vector = np.array([blue_actions[idx][0], blue_actions[idx][1]])
        normalized_v_x = np.clip(direction_vector[0], a_min=-1, a_max=1)
        normalized_v_y = np.clip(direction_vector[1], a_min=-1, a_max=1)
        velocities.append(np.array([normalized_v_x, normalized_v_y]))
    return velocities
            


            
