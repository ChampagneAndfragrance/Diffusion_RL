""" This script collects a triple of agent observations, hideout location, and timestep. """
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from simulator.load_environment import load_environment
from simulator import PrisonerRedEnv
from red_bc.heuristic import BlueHeuristic
from blue_policies.heuristic import BlueRandom, RLWrapper, QuasiEED
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid, RRTStarOnly
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid, AStarOnly
from fugitive_policies.a_star_local_planner import AStarLocalPlanner
from fugitive_policies.diffusion_policy import  DiffusionGlobalPlannerHideout, DiffusionStateOnlyGlobalPlanner
from diffuser.datasets.multipath import NAgentsIncrementalDataset
from diffuser.graders.traj_graders import joint_traj_grader
from matplotlib import pyplot as plt

import argparse
import random
import time

global_device_name = "cuda"
global_device = torch.device("cuda")

mse_loss_func = torch.nn.MSELoss()

def collect_demonstrations(epsilon, num_runs, 
                    starting_seed, 
                    random_cameras, 
                    folder_name,
                    heuristic_type,
                    blue_type,
                    env_path,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """
    num_detections = 0
    total_timesteps = 0
    batch_size = 32
    detect_rates = []
    traj_grader = joint_traj_grader(input_dim=6, hidden_dim=32)
    traj_grader_optimizer = Adam(traj_grader.parameters(), lr=0.01)
    # traj_grader_dataset = NAgentsRewardDataset()
    # dataloader_init = False
    
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        detect = 0
        t = 0
        agent_reward = []
        hideout_observations = []
        timestep_observations = []
        detected_locations = []
        blue_observations = []
        red_observations = []
        last_k_fugitive_detections = []
        last_k_fugitive_detections_vel = []
        t_d1_d0_vel = []
        detection_ranges = []

        red_locations = []
        blue_locations = []

        normalized_red_detect_blue = []

        dones = []

        num_random_known_cameras = random.randint(25, 30)
        num_random_unknown_cameras = random.randint(25, 30)

        env = load_environment(env_path)
        
        # env.seed(seed)
        print("Running with seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)


        if heuristic_type == 'AStar_only':
            red_policy = AStarOnly(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        elif heuristic_type == 'RRTStarOnly':
            red_policy = RRTStarOnly(env, max_speed=env.fugitive_speed_limit, terrain_cost_coef=0, n_iter=2000, goal_sample_rate=0.0, step_len=75, search_radius=75)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"            
        else:
            raise NotImplementedError
        if random_cameras:
            path += "_random_cameras"

        if not os.path.exists(path):
            os.makedirs(path)

        
        if blue_type == "heuristic":
            blue_policy = BlueHeuristic(env, debug=False)
        elif blue_type == "random":
            blue_policy = BlueRandom(env, debug=False)
        elif blue_type == "RL":
            blue_policy = RLWrapper(env, path="/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/marl/20221205-001146/regular_a_star_filtering/")
        elif blue_type == "quasiEED":
            blue_policy = QuasiEED(env, debug=False)
        else:
            raise NotImplementedError

        env = PrisonerRedEnv(env, blue_policy)
        env.reset()

        incremental_dataset = NAgentsIncrementalDataset(env)
        # env = PrisonerGNNEnv(env)
        # gnn_obs, blue_obs = env.reset()

        
        

        got_stuck = False
        done = False
        imgs = []
        while not done:
            if heuristic_type == 'diffusion':
                incremental_dataset.push()
            t += 1
            if "AStar" in heuristic_type or "RRTStar" in heuristic_type:
                red_actions, red_hideout = red_policy.predict(env.get_fugitive_observation(), plot=True)
            else:
                red_actions, red_hideout = red_policy.predict(env.get_fugitive_observation(), dataset=incremental_dataset)
            _, reward, done, _, detections, _ = env.step(red_actions)
            # print("reward = ", reward)
            prisoner_location = env.get_prisoner_location()
            search_party_locations, helicopter_locations = env.get_blue_locations()
            blue_location = helicopter_locations + search_party_locations
            # detected_location = blue_obs[-2:]

            # blue_obs_wrapped = env.blue_obs_names(blue_obs)
            # detected_location = blue_obs_wrapped["prisoner_detected"]
            
            blue_observation = env.get_blue_observation()
            red_observation = env.get_fugitive_observation()
            detection_range = env.get_detection_range()

            red_observations.append(red_observation)
            blue_observations.append(blue_observation)
            agent_reward.append(np.array(reward))
            hideout_observations.append(np.concatenate(env.hideout_locations)/env.dim_x)
            timestep_observations.append(np.array(env.timesteps/env.max_timesteps))
            red_locations.append(prisoner_location)
            blue_locations.append(blue_location)
            normalized_red_detect_blue.append(env.construct_gt_blue_state(freq=1))
            dones.append(done)
            detected_locations.append(detections)
            detection_ranges.append(detection_range)

            # if heuristic_type == 'diffusion':
            #     traj_grader_dataset.push(prisoner_location, blue_location, reward, done, red_hideout)
            # if heuristic_type == 'diffusion' and dataloader_init == False:
            #     collate_fn = traj_grader_dataset.collate_loc_reward()
            #     loc_rew_collate_fn = lambda batch: collate_fn(batch, gamma=0.99, period=10)
            #     traj_grader_dataloader = cycle(torch.utils.data.DataLoader(traj_grader_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True, collate_fn=loc_rew_collate_fn))
            #     dataloader_init = True

            if env.is_detected:
                num_detections += 1
                detect += 1

            if store_last_k_fugitive_detections:
                last_k_fugitive_detections.append(np.array(env.get_last_k_fugitive_detections()))
                last_k_fugitive_detections_vel.append(np.array(env.get_last_k_fugitive_detections_vel()))
            if store_last_two_fugitive_detections:
                t_d1_d0_vel.append(np.array(env.get_last_two_detections_vel_time()))

            if show:
                game_img = env.render('heuristic', show=True, fast=True)
                # imgs.append(game_img)

                # if env.timesteps > 100:
                #     video_path = path + '/' + (str(seed) + ".mp4")
                #     save_video(imgs, str(video_path), fps=10)

            if done:
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                    got_stuck = True
                detect_rates.append(detect/t)
                print(f"{detect}/{t} = {detect/t} detection rate")
                break
                
        # if heuristic_type == 'diffusion':
        #     if len(traj_grader_dataset) > batch_size:
        #         batches_seqLen_agentLocations, batches_seqLen_redRews = traj_grader_dataloader.__next__()
        #         loss = traj_grader.cal_loss(batches_seqLen_agentLocations, batches_seqLen_redRews, loss_func=mse_loss_func)
        #         traj_grader_optimizer.zero_grad()
        #         loss.backward()
        #         traj_grader_optimizer.step()
        #         print("loss = ", loss)


        agent_dict = {"num_known_cameras": env.num_known_cameras,
                      "num_unknown_cameras": env.num_unknown_cameras,
                      "num_helicopters": env.num_helicopters,
                      "num_search_parties": env.num_search_parties}

        blue_obs_dict = env.blue_obs_names._idx_dict
        prediction_obs_dict = env.prediction_obs_names._idx_dict

        red_detect_num = 0
        for detect in detected_locations:
            if np.any(detect[1::4]):
                red_detect_num = red_detect_num + 1
        print("red_detect_num = ", red_detect_num)

        if not got_stuck:
            np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz", 
                blue_observations = blue_observations,
                red_observations = red_observations,
                hideout_observations=hideout_observations,
                timestep_observations=timestep_observations, 
                detected_locations = detected_locations,
                agent_reward=agent_reward,
                red_locations=red_locations, 
                # path_pts = red_policy.get_scaled_path(),
                blue_locations=blue_locations,
                normalized_red_detect_blue=normalized_red_detect_blue,
                dones=dones,
                agent_dict = agent_dict,
                detect=detect,
                last_k_fugitive_detections=last_k_fugitive_detections,
                last_k_fugitive_detections_vel=last_k_fugitive_detections_vel,
                t_d1_d0_vel=t_d1_d0_vel,
                blue_obs_dict = blue_obs_dict,
                prediction_obs_dict = prediction_obs_dict,
                detection_ranges = detection_ranges
                )
    print(detect_rates)

def collect_waypoints(epsilon, num_runs, 
                    starting_seed, 
                    random_cameras, 
                    folder_name,
                    heuristic_type,
                    blue_type,
                    env_path,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """
    num_detections = 0
    total_timesteps = 0
    batch_size = 32
    detect_rates = []
    traj_grader = joint_traj_grader(input_dim=6, hidden_dim=32)
    traj_grader_optimizer = Adam(traj_grader.parameters(), lr=0.01)
    # traj_grader_dataset = NAgentsRewardDataset()
    # dataloader_init = False
    
    break_goal_sample_rate = 0
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        goal_sample_rate = 0.1
        # goal_sample_rate = break_goal_sample_rate + (0.6 - break_goal_sample_rate) / num_runs * (seed - starting_seed)
        print("goal_sample_rate: ", goal_sample_rate)
        detect = 0
        t = 0
        agent_reward = []
        hideout_observations = []
        timestep_observations = []
        detected_locations = []
        blue_observations = []
        red_observations = []
        last_k_fugitive_detections = []
        last_k_fugitive_detections_vel = []
        t_d1_d0_vel = []
        detection_ranges = []

        red_locations = []
        blue_locations = []

        normalized_red_detect_blue = []

        dones = []

        num_random_known_cameras = random.randint(25, 30)
        num_random_unknown_cameras = random.randint(25, 30)

        env = load_environment(env_path)
        
        # env.seed(seed)
        print("Running with seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)

        if heuristic_type == 'Normal':
            red_policy = HeuristicPolicy(env, epsilon=epsilon)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_eps_{epsilon}_{heuristic_type}"
        elif heuristic_type == 'AStar':
            red_policy = AStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        elif heuristic_type == 'AStar_only':
            red_policy = AStarOnly(env, max_speed=env.fugitive_speed_limit, cost_coeff=0)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        elif heuristic_type == 'diffusion':
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
            # diffusion_path = "./saved_models/diffusion_models/one_hideout/random/hs/diff_995000.pt"
            # ema_path = "./saved_models/diffusion_models/one_hideout/random/hs/ema_995000.pt"
            diffusion_path = "./saved_models/diffusion_models/red_only_one_hideout/corner/H240_T10/diff_98000.pt"
            ema_path = "./saved_models/diffusion_models/red_only_one_hideout/corner/H240_T10/ema_98000.pt"
            estimator_path = "./saved_models/traj_graders/H240_T100/est_p4_cam_rew_only_corner_gamma100/grader_log/best.pth"
            diffusion_model = torch.load(diffusion_path).to(global_device_name)
            ema_diffusion_model = torch.load(ema_path).to(global_device_name)
            estimator_model = torch.load(estimator_path).to(global_device_name)
            red_policy = DiffusionGlobalPlannerHideout(env, diffusion_model, ema_diffusion_model, estimator_model, max_speed=env.fugitive_speed_limit)
        elif heuristic_type == 'AStarLocal':
            red_policy = AStarLocalPlanner(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_eps_{epsilon}_{heuristic_type}"
        elif heuristic_type == 'RRTStar':
            red_policy = RRTStarAdversarialAvoid(env, max_speed=env.fugitive_speed_limit, n_iter=2000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        elif heuristic_type == 'RRTStarOnly':
            red_policy = RRTStarOnly(env, max_speed=env.fugitive_speed_limit, terrain_cost_coef=0, n_iter=2000, goal_sample_rate=goal_sample_rate, step_len=50, search_radius=50)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"            
        else:
            raise NotImplementedError
        if random_cameras:
            path += "_random_cameras"

        if not os.path.exists(path):
            os.makedirs(path)

        
        if blue_type == "heuristic":
            blue_policy = BlueHeuristic(env, debug=False)
        elif blue_type == "random":
            blue_policy = BlueRandom(env, debug=False)
        elif blue_type == "RL":
            blue_policy = RLWrapper(env, path="/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/marl/20221205-001146/regular_a_star_filtering/")
        elif blue_type == "quasiEED":
            blue_policy = QuasiEED(env, debug=False)
        else:
            raise NotImplementedError

        env = PrisonerRedEnv(env, blue_policy)
        env.reset(seed=3)

        # INFO: set random seed for the diffusion denoise process
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        incremental_dataset = NAgentsIncrementalDataset(env)

        got_stuck = False
        done = False
        imgs = []

        downsample_ratio, waypt_num = 4, 10
        traj_num = 75 # 150
        for traj_idx in range(traj_num):
            if heuristic_type == 'RRTStarOnly':
                red_locs = red_policy.global_plan(downsample_ratio, waypt_num, env.get_fugitive_observation())
            elif heuristic_type == 'AStar_only':
                red_locs = np.array(red_policy.get_scaled_path(plot=False))
            else:
                raise NotImplementedError
            plt.plot(red_locs[:,0], red_locs[:,1])
            plt.imshow(env.custom_render_canvas(show=False, option=["terrain", "cameras", "hideouts"], large_icons=False), extent=(0, 2428, 0, 2428))
            plt.axis("off")
        plt.xlim(0, 2428)
        plt.ylim(0, 2428)
        # plt.show()
        # INFO: Following are collecting the RRT paths into datasets
        red_locations.append(red_locs)
        timestep_observations.append(np.arange(waypt_num)/waypt_num)
        hideout_observations.append(np.concatenate([np.expand_dims(np.array(env.hideout_locations), axis=0)]*waypt_num, axis=0))
        # plt.savefig("diverse_rrt_paths.png", bbox_inches='tight')
        if not got_stuck:
            np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz", 
                hideout_observations=hideout_observations,
                timestep_observations=timestep_observations, 
                red_locations=red_locations,
                )

    print(detect_rates)

def collect_time(epsilon, num_runs, 
                    starting_seed, 
                    random_cameras, 
                    folder_name,
                    heuristic_type,
                    blue_type,
                    env_path,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """
    detect_rates = []

    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        goal_sample_rate = 0
        # goal_sample_rate = break_goal_sample_rate + (0.6 - break_goal_sample_rate) / num_runs * (seed - starting_seed)
        print("goal_sample_rate: ", goal_sample_rate)

        hideout_observations = []
        timestep_observations = []

        red_locations = []

        env = load_environment(env_path)
        
        # env.seed(seed)
        print("Running with seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)

        if heuristic_type == 'AStar_only':
            red_policy = AStarOnly(env, max_speed=env.fugitive_speed_limit, cost_coeff=1000)
            path = f"datasets/{folder_name}/time_collection/AStar_only/"
        elif heuristic_type == 'diffusion':
            path = f"datasets/{folder_name}/time_collection/diffusion/"
            diffusion_path = "./saved_models/diffusions/diffusion.pth"
            red_policy = DiffusionStateOnlyGlobalPlanner(env, diffusion_path, plot=False, traj_grader_path=None, sel=False) 
        elif heuristic_type == 'RRTStarOnly':
            red_policy = RRTStarOnly(env, max_speed=env.fugitive_speed_limit, terrain_cost_coef=0, n_iter=2000, goal_sample_rate=goal_sample_rate, step_len=50, search_radius=50)
            path = f"datasets/{folder_name}/time_collection/RRTStarOnly/"            
        else:
            raise NotImplementedError

        if random_cameras:
            path += "_random_cameras"

        
        # MDN_filter = load_filter(filtering_model_config="./configs/IROS_2023/sel_mlp.yaml", filtering_model_path="./blue_bc/saved_models/high_speed_corner_fromVel_combined_success/best.pth", device="cuda")
        # env.set_filter(filter_model=MDN_filter) 
        if not os.path.exists(path):
            os.makedirs(path)

        
        if blue_type == "heuristic":
            blue_policy = BlueHeuristic(env, debug=False)
        elif blue_type == "random":
            blue_policy = BlueRandom(env, debug=False)
        elif blue_type == "RL":
            blue_policy = RLWrapper(env, path="/home/wu/GatechResearch/Zixuan/PrisonerEscape/logs/marl/20221205-001146/regular_a_star_filtering/")
        elif blue_type == "quasiEED":
            blue_policy = QuasiEED(env, debug=False)
        else:
            raise NotImplementedError

        env = PrisonerRedEnv(env, blue_policy)
        env.reset()

        downsample_ratio, waypt_num = 4, 10
        exp_traj_set = [1, 10, 20, 30, 40, 50]
        repeat_exp_num = 5
        for traj_num in exp_traj_set:
            repeat_time = []
            for repeat in tqdm(range(repeat_exp_num)):
                env.reset()
                accumulative_time = 0
                if heuristic_type == 'RRTStarOnly' or heuristic_type == 'AStar_only':
                    for traj_idx in range(traj_num):
                        start = time.time()
                        if heuristic_type == 'RRTStarOnly':
                            red_locs = red_policy.global_plan(downsample_ratio, waypt_num, env.get_fugitive_observation())
                        elif heuristic_type == 'AStar_only':
                            red_locs = red_policy.get_scaled_path(plot=False)
                        else:
                            raise NotImplementedError
                        end = time.time()
                        one_rrt_time = end - start
                        # print("rrt outer loop time: ", one_rrt_time)
                        accumulative_time = accumulative_time + one_rrt_time
                elif heuristic_type == 'diffusion':
                    start = time.time()
                    red_locs = red_policy.get_scaled_path(seed=None, hideout_division=[traj_num//3, traj_num//3, traj_num-2*(traj_num//3)])
                    end = time.time()
                    diffusion_rrt_time = end - start
                    # print("diffusion outer loop time: ", diffusion_rrt_time)
                    accumulative_time = accumulative_time + diffusion_rrt_time
                repeat_time.append(accumulative_time)
            np.savez(path + "traj_num_%d_time_%s" % (traj_num, heuristic_type), repeat_time=repeat_time)

            #     if traj_idx % 15 == 0:
            #         print("Planning time for %d trajectory is %f" % (traj_idx, accumulative_time))
            #     plt.plot(red_locs[:,0], red_locs[:,1])
            # plt.xlim(0, 2428)
            # plt.ylim(0, 2428)
            # plt.show()
    print(detect_rates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num

    store_last_k_fugitive_detections = True
    store_last_two_fugitive_detections = True

    starting_seed=0; num_runs=10000; epsilon=0; folder_name = "balance_game"

    heuristic_type = "RRTStarOnly" # "AStar_only, diffusion, RRTStarOnly, RRTStar" 
    blue_type = "heuristic" # "heuristic", "quaisiEED"
    random_cameras=False
    observation_step_type = "Blue"

    env_path = "simulator/configs/balance_game.yaml"

    collect_waypoints(epsilon, num_runs, starting_seed, random_cameras, folder_name, heuristic_type, blue_type, env_path, show=False)
    # collect_time(epsilon, num_runs, starting_seed, random_cameras, folder_name, heuristic_type, blue_type, env_path, show=False)