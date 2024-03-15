import argparse
import torch
import time
import imageio
import json
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import seaborn as sns
from pathlib import Path
from torch.autograd import Variable
from config_loader import config_loader
from tensorboard.backend.event_processing import event_accumulator

def add_event_file(dir_name, path_list):
    for file_name in os.listdir(dir_name.absolute()):
        if file_name.startswith("events"):
            path_list.append(dir_name.absolute() / file_name) 
    return path_list

def run_para_cmp(config):
    root_dir = Path(config["environment"]["root"])
    base_folders = [root_dir/folder for folder in os.listdir(root_dir) if folder != "SUMMARY"]
    agent_num = config["train"]["agent_num"]
    reward_binary_paths = []
    types_agents_eventPaths = []
    for base_folder in base_folders:
        log_folder = base_folder / "log" 
        """ list to contain the reward for each method reward """
        reward_binary_paths = add_event_file(log_folder, reward_binary_paths) 
        def find_grandparent(elem):
            return elem.parent.parent.name
        reward_binary_paths = sorted(reward_binary_paths, key=find_grandparent)
        " list to contain the vf_loss and policy_loss for each method reward "
        type_agents_eventPaths = []
        for agent_ind in range(agent_num):
            type_agent_eventPaths = []
            policy_loss_folder = log_folder / ("agent%d"%agent_ind) / "losses" / "pol_loss"
            vf_loss_folder = log_folder / ("agent%d"%agent_ind) / "losses" / "vf_loss"
            type_agent_eventPaths = add_event_file(policy_loss_folder, type_agent_eventPaths)
            type_agent_eventPaths = add_event_file(vf_loss_folder, type_agent_eventPaths)
            type_agents_eventPaths.append(type_agent_eventPaths)
        types_agents_eventPaths.append(type_agents_eventPaths)

    blue_agent_idx = range(agent_num)

    step_size = 100
    reward_means = []
    reward_stds = []
    vf_loss_means = []
    item_names = ["reward", "TD_MSE", "policy_loss"]
    type_names = ["Baseline", "PER"]
    types_agents_reward, types_agents_vfLoss, types_agents_policyLoss = result_analysis(reward_binary_paths, types_agents_eventPaths, blue_agent_idx)
    # min_len = np.min(np.array([len(agents_reward_mean[i]) for i in range(len(agents_reward_mean))]))
    # agents_reward_mean = [agents_reward_mean[i][:min_len] for i in range(len(agents_reward_mean))]



    # reward_means structure: [[[[folder0_ag0_rew], [folder0_ag1_rew], [folder0_ag2_rew], [folder0_ag3_rew], [folder0_ag4_rew], [folder0_ag5_rew]]], [[]], [[]], [[]], [[]]]  
    # folder0_ag0_rew = reward_means[folder_id=0][0][ag_id=0]
    # plot_mean_std_para(log_folder, [types_agents_reward, types_agents_vfLoss, types_agents_policyLoss], item_names, type_names, blue_agent_idx, step_size)
    # plot_mean_std(vf_loss_means, agent_num)
    plot_reward_mean_std(reward_binary_paths, types_agents_reward, reward_stds, agent_num, step_size)
    
    # env.close()
    return 

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_final_detect_dist(evaluate_config):
    root = evaluate_config["environment"]["root"]
    evaluate_root = root + "/SUMMARY/"
    evaluate_base_dir_appendices = sorted(os.listdir(evaluate_root))   
    model_num = len(evaluate_base_dir_appendices)
    metrics_num = 3
    fig, axes = plt.subplots(1, metrics_num, figsize=(20, 7))
    colors = ['brown', 'g', 'gray', 'c', 'b', 'y', 'pink', 'orange', 'r', 'm']
    metrics_total_names = ["Detection Rate"+r"$\uparrow$", "Closest Distance"+r"$\downarrow$", "Reward"+r"$\uparrow$"]
    ticks_in_metric = ['EED', 'Heuristic', 'Detections+MADDPG', 'FC(Heu)+MADDPG', 'FC(Random)+MADDPG', 'BaseObs+MADDPG', 'PMC+Search', 'PMC+Highest-Prob', '[Ours] PMC(Random)+MADDPG', '[Ours] PMC(Heu)+MADDPG']
    ylabels = ["dr", "Evader to closest searcher", "gg"]
    filter_total_names = []
    
    for model_idx, evaluate_base_dir_appendix in enumerate(evaluate_base_dir_appendices):
        boxprops = dict(linewidth=2.5, facecolor = colors[model_idx])
        medianprops = dict(linewidth=2.5, color='k')
        detection_filename = evaluate_root + evaluate_base_dir_appendix + "/log/detection_rate.txt"
        dist_filename = evaluate_root + evaluate_base_dir_appendix + "/log/closest_dist.txt"
        rew_filename = evaluate_root + evaluate_base_dir_appendix + "/log/reward_data.txt"
        episodes_detection_rate = np.loadtxt(detection_filename)
        # print("The averge detection rate of method %s is %f" % (evaluate_base_dir_appendix, np.mean(episodes_detection_rate)))
        # print("The std detection rate of method %s is %f" % (evaluate_base_dir_appendix, episodes_detection_rate.std()))
        episodes_closest_dist = np.loadtxt(dist_filename) / 2428.0
        # print("The averge closest distance of method %s is %f" % (evaluate_base_dir_appendix, np.mean(episodes_closest_dist)))
        # print("The std closest distance of method %s is %f" % (evaluate_base_dir_appendix, episodes_closest_dist.std()))
        episodes_agents_reward = np.loadtxt(rew_filename)
        episodes_reward = episodes_agents_reward.mean(axis=-1)
        print("The averge reward of method %s is %f" % (evaluate_base_dir_appendix, np.mean(episodes_reward)))
        print("The std reward of method %s is %f" % (evaluate_base_dir_appendix, episodes_reward.std()))
        bp_detection_rate = axes[0].boxplot([episodes_detection_rate], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=True, showfliers=False)
        bp_closest_dist = axes[1].boxplot([episodes_closest_dist], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=True, showfliers=False)
        bp_reward = axes[2].boxplot([episodes_reward], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=True, showfliers=False)

        # set_box_color(bp_detection_rate, colors[model_idx])
        # set_box_color(bp_closest_dist, colors[model_idx])
        # set_box_color(bp_reward, colors[model_idx])

    for metric_idx in range(metrics_num):
        # axes[metric_idx].set_title(metrics_total_names[metric_idx], fontsize=20)
        axes[metric_idx].set_ylabel(metrics_total_names[metric_idx], fontsize=20)
        # ticks_in_metric = model_ticks[metric_idx]
        axes[metric_idx].set_xlim(-0.2, model_num*0.2)
        axes[metric_idx].set_xticks((0.2*np.arange(model_num)).tolist())
        axes[metric_idx].set_xticklabels(ticks_in_metric, rotation=45, ha='right', color="k", size=15)  
        axes[metric_idx].tick_params(axis='y', labelsize=15)
        # axes[metric_idx].tick_params(labelbottom = False)
        axes[metric_idx].grid(visible=True)

    # plt.plot([], c='g', label='Heuristic',linewidth=2.5)
    # plt.plot([], c='r', label='Detections+MADDPG',linewidth=2.5)
    # # plt.plot([], c='c', label='LFM+MADDPG',linewidth=2.5)
    # plt.plot([], c='c', label='FC(Heu)+MADDPG',linewidth=2.5)
    # plt.plot([], c='b', label='FC(Random)+MADDPG',linewidth=2.5)
    # plt.plot([], c='k', label='BaseObs+MADDPG',linewidth=2.5)
    # plt.plot([], c='y', label='PMC+Policy(Heu)',linewidth=2.5)
    # plt.plot([], c='m', label='PMC(Random)+Policy',linewidth=2.5)
    # plt.plot([], c='orange', label='PMC(Heu)+MADDPG',linewidth=2.5)
    # plt.plot([], c='grey', label='PMC+MADDPG(Share)',linewidth=2.5)





    # fig.legend(fontsize=15,loc='lower center', mode = "expand", ncol = 3)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.42)
    plt.savefig('marl_metric.png', dpi=100)



def plot_evaluation(evaluate_config):
    root = evaluate_config["environment"]["root"]
    evaluate_root = root + "/SUMMARY/"
    evaluate_base_dir_appendices = os.listdir(evaluate_root)



    plt.figure()
    for evaluate_base_dir_appendix in evaluate_base_dir_appendices:
        detection_filename = evaluate_root + evaluate_base_dir_appendix + "/log/detection_data.txt"
        agents_models_episodes_detection = np.loadtxt(detection_filename)
        meanAgents_models_episodes_detection = agents_models_episodes_detection.mean(axis=1)

        agent_num = agents_models_episodes_detection.shape[1]
        start_episode = evaluate_config["train"]["start_episode"]
        end_episode = evaluate_config["train"]["end_episode"]
        increment_episode = evaluate_config["train"]["increment_episode"]
        episode_axis = np.arange(start_episode, end_episode, increment_episode)

        each_agent_fig_title = "Total_detections"
        plt.title(each_agent_fig_title)
        plt.plot(episode_axis, meanAgents_models_episodes_detection, linewidth=2)
    lg = plt.legend(evaluate_base_dir_appendices, bbox_to_anchor=(1.05, 0.8, 0.8, 0.2), loc='upper left')
    plt.xlabel("Episode")
    plt.ylabel("Average Detection Num.")
    plt.grid()
    plt.savefig(evaluate_root + (each_agent_fig_title + ".png"), bbox_extra_artists=(lg,), bbox_inches='tight')

    plt.figure()
    for evaluate_base_dir_appendix in evaluate_base_dir_appendices:
        reward_filename = evaluate_root + evaluate_base_dir_appendix + "/log/reward_data.txt"
        agents_models_episodes_reward = np.loadtxt(reward_filename)
        meanAgents_models_episodes_reward = agents_models_episodes_reward.mean(axis=1)

        agent_num = agents_models_episodes_reward.shape[1]
        start_episode = evaluate_config["train"]["start_episode"]
        end_episode = evaluate_config["train"]["end_episode"]
        increment_episode = evaluate_config["train"]["increment_episode"]
        episode_axis = np.arange(start_episode, end_episode, increment_episode)

        each_agent_fig_title = "Total_reward"
        plt.title(each_agent_fig_title)
        plt.plot(episode_axis, meanAgents_models_episodes_reward, linewidth=2)
    lg = plt.legend(evaluate_base_dir_appendices, bbox_to_anchor=(1.05, 0.8, 0.8, 0.2), loc='upper left')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.savefig(evaluate_root + (each_agent_fig_title + ".png"), bbox_extra_artists=(lg,), bbox_inches='tight')
    
    for ag_i in range(agent_num):
        plt.figure()
        for evaluate_base_dir_appendix in evaluate_base_dir_appendices:
            detection_filename = evaluate_root + evaluate_base_dir_appendix + "/log/detection_data.txt"
            agents_models_episodes_detection = np.loadtxt(detection_filename)
            """plot the detections for comparison (baseline and per)"""
            agent_models_episodes_detection = agents_models_episodes_detection[:,ag_i]
            each_agent_fig_title = "agent_%i_detections" % (ag_i)
            plt.title(each_agent_fig_title)
            plt.plot(episode_axis, agent_models_episodes_detection, linewidth=2)
        lg = plt.legend(evaluate_base_dir_appendices, bbox_to_anchor=(1.05, 0.8, 0.8, 0.2), loc='upper left')
        plt.xlabel("Episode")
        plt.ylabel("Average Detection Num.")
        plt.grid()
        plt.savefig(evaluate_root + (each_agent_fig_title + ".png"), bbox_extra_artists=(lg,), bbox_inches='tight')

    
    for ag_i in range(agent_num):
        plt.figure()
        for evaluate_base_dir_appendix in evaluate_base_dir_appendices:
            reward_filename = evaluate_root + evaluate_base_dir_appendix + "/log/reward_data.txt"
            agents_models_episodes_reward = np.loadtxt(reward_filename)
            """plot the detections for comparison (baseline and per)"""
            agent_models_episodes_reward = agents_models_episodes_reward[:,ag_i]
            each_agent_fig_title = "agent_%i_reward" % (ag_i)
            plt.title(each_agent_fig_title)
            plt.plot(episode_axis, agent_models_episodes_reward, linewidth=2)
        lg = plt.legend(evaluate_base_dir_appendices, bbox_to_anchor=(1.05, 0.8, 0.8, 0.2), loc='upper left')
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid()
        plt.savefig(evaluate_root + (each_agent_fig_title + ".png"), bbox_extra_artists=(lg,), bbox_inches='tight')

            # """plot the reward for comparison (baseline and per)"""
            # agent_models_episodes_reward_bl = agents_models_episodes_reward[:,ag_i]
            # agent_models_episodes_reward_per = agents_models_episodes_reward[:,ag_i]
            # plt.figure()
            # each_agent_fig_title = "agent_%i_reward" % (ag_i)
            # plt.title(each_agent_fig_title)
            # plt.plot(episode_axis, agent_models_episodes_reward_bl, linewidth=2)
            # plt.plot(episode_axis, agent_models_episodes_reward_per, linewidth=2)
            # plt.legend(["Baseline", "PER"])
            # plt.xlabel("Episode")
            # plt.ylabel("Average Reward")
            # plt.grid()
            # plt.savefig(config["environment"]["base_path"][0] + "/logs/" + (each_agent_fig_title + ".png"))
            # plt.savefig(config["environment"]["base_path"][1] + "/logs/" + (each_agent_fig_title + ".png"))
    plt.show()


def plot_mean_std_para(log_folder, types_agents_items, item_names, type_names, blue_agent_idx, step_size):
    weight_smooth = 0.600
    types_num = len(types_agents_items[0])
    agents_num = len(types_agents_items[0][0])

    for item_idx, types_agents_item in enumerate(types_agents_items):
        for agent_idx in range(agents_num):
            plt.figure()
            each_agent_fig_title = "agent_%i_%s" % (agent_idx, item_names[item_idx])
            plt.title(each_agent_fig_title)
            for type_idx in range(types_num):
                agent_type_item = types_agents_item[type_idx][agent_idx]
                type_agent_item = smooth(np.array(agent_type_item), weight=weight_smooth)
                episodes = np.arange(type_agent_item.shape[-1])[0::step_size]
                type_agent_item = type_agent_item[0::step_size]
                plt.plot(episodes, type_agent_item, linewidth=2)
            plt.legend(type_names)
            plt.xlabel("Episode")
            plt.ylabel(item_names[item_idx])
            plt.grid()
            plt.savefig(log_folder.absolute() / (each_agent_fig_title + ".png"))

        types_agents_item = np.array(types_agents_item)
        types_item = types_agents_item.mean(axis=1)
        types_item = smooth_paras(types_item, weight=weight_smooth)
        plt.figure()
        average_fig_title = "%s_Average" % (item_names[item_idx])
        plt.title(average_fig_title)
        for type_idx in range(types_num):
            type_item = types_item[type_idx]
            episodes = np.arange(type_item.shape[-1])[0::step_size]
            type_item = type_item[0::step_size]  
            plt.plot(episodes, type_item, linewidth=2)
        plt.legend(type_names)
        plt.xlabel("Episode")
        plt.ylabel(item_names[item_idx])
        plt.grid()
        plt.savefig(log_folder.absolute() / (average_fig_title + ".png"))


    # plt.show()
                
            # # paras_reward_means = np.sum(paras_agents_reward_means, axis=0)
            # # episode_num = paras_reward_means.shape[1]
            # # episodes = np.arange(0, episode_num, step_size)
            # # paras_reward_means = paras_reward_means
            # # config_num = paras_reward_means.shape[0]
            
            # plt.figure()
            # plt.plot(episodes, paras_reward_means.transpose(), label='total_reward')
            # plt.title("Total Reward")
            # # plt.legend(legend_on_plot)
            # plt.savefig(log_folder.absolute() / ("Total_Reward.png"))

            # for idx, a_i in enumerate(blue_agent_idx):
            #     plt.figure()
            #     plt.plot(episodes, paras_agents_reward_means[idx,:].transpose())
            #     plt.title("agent_%i_reward" % a_i)
            #     # plt.legend(legend_on_plot)
            #     plt.savefig(log_folder.absolute() / ("agent_%i_reward.png" % a_i))
   
def plot_mean_std(binary_path, reward_means, reward_stds, agent_num, step_size):
    for (reward_means_bl_es, reward_stds_bl_es) in zip(reward_means, reward_stds):
        total_rew0 = 0
        total_rew1 = 0
        for a_i in range(agent_num):
            window_size = 20
            weight_smooth = 0.999
            
            
            agent0_reward_mean = np.array(reward_means_bl_es[0][a_i])
            if a_i != 3:
                total_rew0 = total_rew0 + agent0_reward_mean
            episode_axis0 = np.arange(len(agent0_reward_mean))[0:-1:step_size]
            agent0_reward_std = window_std(agent0_reward_mean, window_size)
            agent0_reward_mean = smooth(agent0_reward_mean, weight_smooth)[0:-1:step_size]
            agent0_reward_std = smooth(agent0_reward_std, weight_smooth)[0:-1:step_size]
            
            
            
            agent1_reward_mean =  np.array(reward_means_bl_es[1][a_i])
            if a_i != 3:
                 total_rew1 = total_rew1 + agent1_reward_mean
            episode_axis1 = np.arange(len(agent1_reward_mean))[0:-1:step_size]
            agent1_reward_std = window_std(agent1_reward_mean, window_size)
            agent1_reward_mean = smooth(agent1_reward_mean, weight_smooth)[0:-1:step_size]
            agent1_reward_std = smooth(agent1_reward_std, weight_smooth)[0:-1:step_size]
            
            plot_and_save(a_i, episode_axis0, agent0_reward_mean, agent0_reward_std, agent1_reward_mean, agent1_reward_std, binary_path)

        total0_reward_std = window_std(total_rew0, window_size)
        total1_reward_std = window_std(total_rew1, window_size) 
        total0_reward_std = smooth(total0_reward_std, weight_smooth)[0:-1:step_size]
        total1_reward_std = smooth(total1_reward_std, weight_smooth)[0:-1:step_size]
        total_rew0 = smooth(total_rew0, weight_smooth)[0:-1:step_size]
        total_rew1 = smooth(total_rew1, weight_smooth)[0:-1:step_size]
        plot_and_save("Total", episode_axis0, total_rew0, total0_reward_std, total_rew1, total1_reward_std, binary_path)     
    plt.show()

def plot_reward_mean_std(binary_path, reward_means, reward_stds, agent_num, step_size):
    filterMethods_agents_episode_axis = []
    filterMethods_agents_smoothed_reward_mean = []
    filterMethods_agents_smoothed_reward_std = []
    filterMethods_total_smoothed_reward_mean = []
    filterMethods_total_smoothed_reward_std = []
    for (reward_means_bl_es) in (reward_means):
        total_rew = 0
        filterMethod_agents_smoothed_reward_mean = []
        filterMethod_agents_smoothed_reward_std = []
        for a_i in range(agent_num):
            window_size = 15
            weight_smooth = 0.9
            agent_reward_mean = reward_means_bl_es[a_i]
            total_rew = total_rew + agent_reward_mean
            episode_axis = np.arange(len(agent_reward_mean))[0:-1:step_size]
            agent_reward_std = window_std(agent_reward_mean, window_size)
            agent_reward_mean = smooth(agent_reward_mean, weight_smooth)[0:-1:step_size]
            agent_reward_std = smooth(agent_reward_std, weight_smooth)[0:-1:step_size]  
            filterMethods_agents_episode_axis.append(episode_axis)
            filterMethod_agents_smoothed_reward_mean.append(agent_reward_mean)
            filterMethod_agents_smoothed_reward_std.append(agent_reward_std)
        filterMethods_agents_smoothed_reward_mean.append(filterMethod_agents_smoothed_reward_mean)
        filterMethods_agents_smoothed_reward_std.append(filterMethod_agents_smoothed_reward_std)
            # plot_and_save(a_i, episode_axis, agent_reward_mean, agent_reward_std, agent1_reward_mean, agent1_reward_std, binary_path)

        total_reward_std = window_std(total_rew, window_size)
        total_reward_std = smooth(total_reward_std, weight_smooth)[0:-1:step_size]
        total_rew = smooth(total_rew, weight_smooth)[0:-1:step_size]
        filterMethods_total_smoothed_reward_mean.append(total_rew)
        filterMethods_total_smoothed_reward_std.append(total_reward_std)
        # plot_and_save("Total", episode_axis0, total_rew0, total0_reward_std, total_rew1, total1_reward_std, binary_path)     
    # plt.show()
    plot_and_save_single_bufferType(binary_path, filterMethods_agents_episode_axis, filterMethods_agents_smoothed_reward_mean, 
                                    filterMethods_agents_smoothed_reward_std, filterMethods_total_smoothed_reward_mean, filterMethods_total_smoothed_reward_std)
    print("Done.")

def plot_and_save_para(episode_axis, agent0_reward_mean, agent0_reward_std, binary_path, a_i, option):
    if option == "total":
        plt.figure()
        plt.plot(episode_axis, agent0_reward_mean, label='mean_0', color='b')
        plt.legend(["baseline", "exp_sharing"], loc = 'best')
        plt.fill_between(episode_axis, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
        plt.title("{a_i} Reward v.s. Episode".format(a_i=a_i))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(binary_path[0].parent / ('%s_rew.png'%a_i))
        plt.savefig(binary_path[1].parent / ('%s_rew.png'%a_i))
    elif option == "agent":
        plt.figure()
        plt.plot(episode_axis, agent0_reward_mean, label='mean_0', color='b')
        # plt.plot(episode_axis5, agent5_reward_mean, 'b-', label='mean_5')
        plt.legend(["baseline", "exp_sharing"])
        plt.fill_between(episode_axis, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
        plt.title("Agent {a_i} Reward v.s. Episode".format(a_i=a_i))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(binary_path[0].parent / ('ag%i_rew.png'%a_i))
        plt.savefig(binary_path[1].parent / ('ag%i_rew.png'%a_i))
    else:
        raise NotImplementedError

def plot_and_save_single_bufferType(binary_path, fs_axis, fs_mean, fs_std, tfs_mean, tfs_std):
    filter_num = len(binary_path)
    colors = ['c', 'm']
    # label_names = ["FC(Heu)+MADDPG", "FC(Random)+MADDPG", "PMC(Heu)+MADDPG", "PMC(Random)+MADDPG"]
    # label_names = ["FC(Random)+MADDPG", "[Ours] PMC(Random)+MADDPG"]
    label_names = ["FC(Heu)+MADDPG", "[Ours] PMC(Heu)+MADDPG"]
    plt.figure(figsize=(10,5))
    for filter_idx in range(filter_num):
        plt.plot(fs_axis[filter_idx], tfs_mean[filter_idx], label=label_names[filter_idx], color=colors[filter_idx], linewidth=2)
        # plt.legend(["baseline", "exp_sharing"], loc = 'best')
        plt.fill_between(fs_axis[filter_idx], tfs_mean[filter_idx] - tfs_std[filter_idx], tfs_mean[filter_idx] + tfs_std[filter_idx], color=colors[filter_idx], alpha=0.2)
        # plt.hold(True)

    plt.title("Searching Agents Total Reward v.s. Episode", fontsize=20)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Reward", fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend(loc="upper left", fontsize=15)
    plt.grid(True)
    # plt.savefig(binary_path[0].parent / ('total_rew_rand.png'))
    plt.savefig(binary_path[0].parent / ('total_rew_heu.png'))
    plt.show()
    

    # else:
    #     plt.figure()
    #     plt.plot(episode_axis, agent0_reward_mean, label='mean_0', color='b')
    #     plt.plot(episode_axis, agent1_reward_mean, label='mean_0', color='r')
    #     # plt.plot(episode_axis5, agent5_reward_mean, 'b-', label='mean_5')
    #     plt.legend(["baseline", "exp_sharing"])
    #     plt.fill_between(episode_axis, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
    #     plt.fill_between(episode_axis, agent1_reward_mean - agent1_reward_std, agent1_reward_mean + agent1_reward_std, color='r', alpha=0.2)
    #     plt.title("Agent {a_i} Reward v.s. Episode".format(a_i=a_i))
    #     plt.xlabel("Episode")
    #     plt.ylabel("Reward")
    #     plt.savefig(binary_path[0].parent / ('ag%i_rew.png'%a_i))
    #     plt.savefig(binary_path[1].parent / ('ag%i_rew.png'%a_i))

def plot_and_save(a_i, episode_axis, agent0_reward_mean, agent0_reward_std, agent1_reward_mean, agent1_reward_std, binary_path):
    if a_i == "Total":
        plt.figure()
        plt.plot(episode_axis, agent0_reward_mean, label='mean_0', color='b')
        plt.plot(episode_axis, agent1_reward_mean, label='mean_0', color='r')
        # plt.plot(episode_axis5, agent5_reward_mean, 'b-', label='mean_5')
        plt.legend(["baseline", "exp_sharing"], loc = 'best')
        plt.fill_between(episode_axis, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
        plt.fill_between(episode_axis, agent1_reward_mean - agent1_reward_std, agent1_reward_mean + agent1_reward_std, color='r', alpha=0.2)
        plt.title("{a_i} Reward v.s. Episode".format(a_i=a_i))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(binary_path[0].parent / ('%s_rew.png'%a_i))
        plt.savefig(binary_path[1].parent / ('%s_rew.png'%a_i))
    else:
        plt.figure()
        plt.plot(episode_axis, agent0_reward_mean, label='mean_0', color='b')
        plt.plot(episode_axis, agent1_reward_mean, label='mean_0', color='r')
        # plt.plot(episode_axis5, agent5_reward_mean, 'b-', label='mean_5')
        plt.legend(["baseline", "exp_sharing"])
        plt.fill_between(episode_axis, agent0_reward_mean - agent0_reward_std, agent0_reward_mean + agent0_reward_std, color='b', alpha=0.2)
        plt.fill_between(episode_axis, agent1_reward_mean - agent1_reward_std, agent1_reward_mean + agent1_reward_std, color='r', alpha=0.2)
        plt.title("Agent {a_i} Reward v.s. Episode".format(a_i=a_i))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(binary_path[0].parent / ('ag%i_rew.png'%a_i))
        plt.savefig(binary_path[1].parent / ('ag%i_rew.png'%a_i))
    
def window_std(np_data, window_size):
    std_out = np.zeros(window_size)
    data_num = np_data.shape[0]
    start_idx = 0
    end_idx = start_idx + window_size
    while end_idx < data_num:
        std_out = np.hstack((std_out, np_data[start_idx:end_idx].std()))
        start_idx = start_idx + 1
        end_idx = start_idx + window_size
    return std_out


def window_std_para(np_data, window_size):
    std_out = np.zeros(window_size)
    data_num = np_data.shape[0]
    start_idx = 0
    end_idx = start_idx + window_size
    while end_idx < data_num:
        std_out = np.hstack((std_out, np_data[start_idx:end_idx].std()))
        start_idx = start_idx + 1
        end_idx = start_idx + window_size
    return std_out


def result_analysis(reward_binary_paths, types_agents_eventPaths, adversary_idx):
    # --------------------------------------------Draw Plot Begin Here------------------------------------------------
    # Calculate average reward, loss
    types_agents_reward = []
    for reward_binary_path in reward_binary_paths:
        reward_binary_path = str(reward_binary_path)
        ea=event_accumulator.EventAccumulator(reward_binary_path, size_guidance={'scalars': 0})
        ea.Reload()
        print(ea.scalars.Keys())
        # weight_smooth = [1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10] # convolution coefficient, sum up to 1 and first coefficient means weight for the most recent element
        weight_smooth = 0.99
        agents_reward_mean_agg = []
        for i in adversary_idx:
            rew_agi = ea.scalars.Items('agent{a_i}/mean_episode_rewards'.format(a_i=i))
            # rew_agi = ea.scalars.Items('agent{a_i}/losses/curr_filtering_loss'.format(a_i=i))
            agenti_reward_mean = np.array([event.value for event in rew_agi])
            # agenti_reward_mean = smooth(agenti_reward_mean, weight_smooth)
            # agenti_reward_std = smooth(agenti_reward_std, weight_smooth)
            episode_axisi = np.array(range(len(agenti_reward_mean)))
            agents_reward_mean_agg.append(agenti_reward_mean)
        types_agents_reward.append(agents_reward_mean_agg)

    types_agents_vfLoss = []
    types_agents_policyLoss = []
    for type_agents_eventPaths in types_agents_eventPaths:
        type_agents_vfLoss = []
        type_agents_policyLoss = []
        for i, type_agent_eventPaths in enumerate(type_agents_eventPaths):
            type_agent_vfPath = str(type_agent_eventPaths[1])
            vf_ea=event_accumulator.EventAccumulator(type_agent_vfPath, size_guidance={'scalars': 0})
            vf_ea.Reload()
            print(vf_ea.scalars.Keys())               
            vf_loss_agi = vf_ea.scalars.Items('agent{a_i}/losses'.format(a_i=i))
            agenti_vf_loss_mean = np.array([event.value for event in vf_loss_agi])
            # agenti_vf_loss_mean = smooth(agenti_vf_loss_mean, weight_smooth)
            episode_axisi = np.array(range(len(agenti_vf_loss_mean)))
            type_agents_vfLoss.append(agenti_vf_loss_mean)

            type_agent_PolicyPath = str(type_agent_eventPaths[0])
            policy_ea=event_accumulator.EventAccumulator(type_agent_PolicyPath, size_guidance={'scalars': 0})
            policy_ea.Reload()
            print(policy_ea.scalars.Keys())               
            policy_loss_agi = policy_ea.scalars.Items('agent{a_i}/losses'.format(a_i=i))
            agenti_policy_loss_mean = np.array([event.value for event in policy_loss_agi])
            # agenti_policy_loss_mean = smooth(agenti_policy_loss_mean, weight_smooth)
            episode_axisi = np.array(range(len(agenti_policy_loss_mean)))
            type_agents_policyLoss.append(agenti_policy_loss_mean)
        types_agents_vfLoss.append(type_agents_vfLoss)
        types_agents_policyLoss.append(type_agents_policyLoss)

    return types_agents_reward, types_agents_vfLoss, types_agents_policyLoss


# Smooth the reward data
def smooth_paras(scalar, weight=0.6):
    # weight = np.array(weight)
    # smoothed = np.convolve(scalar, weight, 'same')
    if weight == 0:
        smoothed = scalar
        return smoothed

    smoothed = scalar
    last = scalar[:,0]
    reward_num = scalar.shape[-1]
    for i in range(reward_num):
        point = scalar[:,i]
        smoothed_val = last * weight + (1 - weight) * point
        smoothed[:,i] = smoothed_val
        last = smoothed_val
    return smoothed

def smooth(scalar, weight=[0.75, 0.15, 0.1]):
    # weight = np.array(weight)
    # smoothed = np.convolve(scalar, weight, 'same')
    if weight == 0:
        smoothed = scalar
        return smoothed
    last = scalar[0]
    smoothed = []
    for ind, point in enumerate(scalar):
        if ind >= 0:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
    return np.array(smoothed)

if __name__ == '__main__':
    evaluate_config = config_loader(path="./blue_bc/parameters_evaluate.yaml")  # load model configuration
    root_dir_name = evaluate_config["environment"]["root"]
    """from training data"""
    # run_para_cmp(evaluate_config)
    """from evaluation data"""
    # plot_evaluation(evaluate_config)
    plot_final_detect_dist(evaluate_config)