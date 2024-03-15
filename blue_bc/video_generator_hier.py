import os
import argparse

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator import BlueSequenceEnv
from simulator.prisoner_env import PrisonerBothEnv
from simulator.prisoner_perspective_envs import PrisonerBlueEnv
from fugitive_policies.heuristic import HeuristicPolicy
from heuristic import HierRLBlue
import matplotlib
import torch
import numpy as np
from tqdm import tqdm

matplotlib.use('agg')
import matplotlib.pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils import save_video
import random
from simulator.load_environment import load_environment

def generate_demo_video(env, blue_policy, path, device):
    blue_hier_policy = HierRLBlue(env, blue_policy)
    blue_hier_policy.reset()
    blue_hier_policy.init_behavior()
    next_blue_observation, next_blue_partial_observation = env.reset()
    imgs = []
    t = 0
    episode_return = 0.0
    while 1:
        # INFO: run episode
        t = t + 1
        # print("current t = ", t)
        """Partial Blue Obs"""
        # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
        """Full Blue Obs"""
        action = blue_hier_policy.predict_full_observation(next_blue_observation)

        next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(split_directions_to_direction_speed(action.cpu().detach().numpy()))
        # print("helicopter_actions = ", blue_actions[5])
        # print("blue_actions = ", blue_actions)
        mask = False if t == env.max_timesteps else done
        # print("blue_partial_observation = ", blue_partial_observation)
        # print("next_blue_partial_observation = ", next_blue_partial_observation)
        # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
        # print("helicopter_action_theta_speed", blue_actions[5])
        episode_return += reward

        blue_observation = next_blue_observation
        blue_partial_observation = next_blue_partial_observation
        game_img = env.render('Policy', show=False, fast=True)
        imgs.append(game_img)
        if done:

            break
        
        """End Revising"""
    save_video(imgs, path, fps=10)


def generate_multiple_demo_video(env, video_num, model_path, video_dir, device):
    
    for vn in tqdm(range(video_num)):
        """Reset the environment"""
        next_blue_observation, next_blue_partial_observation = env.reset()
        """Load the model"""
        blue_policy = torch.load(model_path).to(device)
        blue_hier_policy = HierRLBlue(env, blue_policy, device)
        blue_hier_policy.reset()
        blue_hier_policy.init_behavior()

        imgs = []
        t = 0
        # total_return = 0.0
        # num_episodes = 0
        episode_return = 0.0
        done = False
        while not done:
            # INFO: run episode
            t = t + 1
            # print("current t = ", t)
            # red_action = red_policy.predict(red_observation)
            # blue_actions = blue_heuristic.step_observation(blue_observation)
            """Partial Blue Obs"""
            # action = blue_policy(torch.Tensor(next_blue_partial_observation).cuda())
            """Full Blue Obs"""
            action = blue_hier_policy.predict_full_observation(next_blue_observation)

            next_blue_observation, next_blue_partial_observation, reward, done, _ = env.step(action)
            # print("helicopter_actions = ", blue_actions[5])
            # print("blue_actions = ", blue_actions)
            mask = False if t == env.max_timesteps else done
            # print("blue_partial_observation = ", blue_partial_observation)
            # print("next_blue_partial_observation = ", next_blue_partial_observation)
            # print("to_velocity_vector(blue_actions)) = ", to_velocity_vector(blue_actions))
            # print("helicopter_action_theta_speed", blue_actions[5])
            episode_return += reward

            blue_observation = next_blue_observation
            blue_partial_observation = next_blue_partial_observation
            game_img = env.render('Policy', show=False, fast=True)
            imgs.append(game_img)
            print("t = ", t)
            if done:
                print("go into done branch")
                video_path = video_dir + "_" + str(vn) + ".mp4"
                save_video(imgs, video_path, fps=10)

                blue_observation, blue_partial_observation = env.reset()
                t = 0
                episode_return = 0.0
                imgs = []

                break
    return
        

def split_directions_to_direction_speed(directions):
    blue_actions_norm_angle_vel = []
    blue_actions_directions = np.split(directions, 6)
    search_party_v_limit = 6.5
    helicopter_v_limit = 127
    for idx in range(len(blue_actions_directions)):
        if idx < 5:
            search_party_direction = blue_actions_directions[idx]
            if np.linalg.norm(search_party_direction) > 1:
                search_party_direction = search_party_direction / np.linalg.norm(search_party_direction)
            search_party_speed = search_party_v_limit
            blue_actions_norm_angle_vel.append(np.array(search_party_direction.tolist() + [search_party_speed]))
        elif idx < 6:
            helicopter_direction = blue_actions_directions[idx]
            if np.linalg.norm(helicopter_direction) > 1:
                helicopter_direction = helicopter_direction / np.linalg.norm(helicopter_direction)
            helicopter_speed = helicopter_v_limit
            blue_actions_norm_angle_vel.append(np.array(helicopter_direction.tolist()+ [helicopter_speed]))  

    return blue_actions_norm_angle_vel    

if __name__ == '__main__':

    """add some configurations"""
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--video_dir', type=str, required=True)

    
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    """if we use lstm"""
    p.add_argument('--seq_len', type=int, default=50)
    args = p.parse_args()

    """Revision Begins Here"""
    device = 'cuda' if args.cuda else 'cpu'
    epsilon = 0.1
    variation = 0
    print(f"Loaded environment variation {variation} with seed {args.seed}")

    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = load_environment('simulator/configs/blue_bc.yaml')
    env.seed(args.seed)
    fugitive_policy = HeuristicPolicy(env, epsilon=epsilon)
    env = PrisonerBlueEnv(env, fugitive_policy)
    """if we use lstm"""
    # env = BlueSequenceEnv(env, sequence_len=args.seq_len)

    # model_path = "/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/bc/20220621-2231/policy_epoch_1500.pth"
    # video_dir = "/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/bc/20220621-2231/final_video/"
    generate_multiple_demo_video(env, video_num=5, model_path=args.model_path, video_dir=args.video_dir, device=device)



        
    # red_policy = RRTStarAdversarialAvoid(env, terrain_cost_coef=2000, n_iter=2000, max_speed=7.5)
    
