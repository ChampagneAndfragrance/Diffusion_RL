import argparse
import torch
import time
import imageio
import json
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import seaborn as sns
from pathlib import Path
from torch.autograd import Variable
from config_loader import config_loader
from tensorboard.backend.event_processing import event_accumulator

def plot_final_detect_dist(env_name):
    root = "./logs/RAL2024"
    evaluate_root = root + "/benchmark_results/"
    evaluate_base_dir_appendices = ["A_Star(escape)", "RRT_Star(escape)", "VO", "DDPG", "SAC", "Diffusion", "Diffusion_RL", "Sel_Diffusion_RL"]
    
    model_num = len(evaluate_base_dir_appendices)
    metrics_num = 3
    fig, axes = plt.subplots(1, metrics_num, figsize=(20, 7))
    colors = ['brown', 'g', 'gray', 'c', 'b', 'y', 'pink', 'orange', 'r', 'm']
    # metrics_total_names = ["Detection Rate"+r"$\downarrow$", "Success Rate"+r"$\uparrow$", "Score"+r"$\uparrow$", "Timesteps"+r"$\downarrow$"]
    metrics_total_names = ["Score", "Detection", "Goal-Reaching Rate ", "Timesteps"]
    ticks_in_metric = ["A* Heuristic", "RRT* Heuristic", "VO Heuristic", "DDPG", "SAC", "Diffusion Only", "Diffusion-RL [Ours]", "Diffusion-RL-Map [Ours]"]
    color_in_metric = ['k', "k", "k", "k", "k", "k", "blue", "blue"]
    ylabels = ["dr", "Evader to closest searcher", "gg"]
    filter_total_names = []
    models_successRate = []
    models_unSuccessRate = []

    min_detection, min_score, min_success, min_time, min_dist = np.inf, np.inf, np.inf, np.inf, np.inf
    max_detection, max_score, max_sucess, max_time, max_dist = -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

    for model_idx, evaluate_base_dir_appendix in enumerate(evaluate_base_dir_appendices):
        boxprops = dict(linewidth=2.5, facecolor = colors[model_idx])
        medianprops = dict(linewidth=2.5, color='k')

        # INFO: prepare the file names
        detection_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/detections.txt"
        success_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/success.txt"
        score_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/scores.txt"
        time_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/time.txt"
        dist_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/closest_dist.txt"

        # INFO: get min and max value of each metric across all the models
        episodes_detection = np.loadtxt(detection_filename)
        min_detection = np.minimum(min_detection, np.min(episodes_detection))
        max_detection = np.maximum(max_detection, np.max(episodes_detection))

        episodes_success = np.loadtxt(success_filename)
        min_success = np.minimum(min_success, np.min(episodes_success))
        max_sucess = np.maximum(max_sucess, np.max(episodes_success))

        episodes_score = np.loadtxt(score_filename)
        min_score = np.minimum(min_score, np.min(episodes_score))
        max_score = np.maximum(max_score, np.max(episodes_score))

        episodes_time = np.loadtxt(time_filename)
        min_time = np.minimum(min_time, np.min(episodes_time))
        max_time = np.maximum(max_time, np.max(episodes_time))

        episodes_dist = np.loadtxt(dist_filename)
        min_dist = np.minimum(min_dist, np.min(episodes_dist))
        max_dist = np.maximum(max_dist, np.max(episodes_dist))

    for model_idx, evaluate_base_dir_appendix in enumerate(evaluate_base_dir_appendices):
        boxprops = dict(linewidth=2.5, facecolor = colors[model_idx])
        medianprops = dict(linewidth=2.5, color='k')

        # INFO: prepare the file names
        detection_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/detections.txt"
        success_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/success.txt"
        score_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/scores.txt"
        time_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/time.txt"
        dist_filename = evaluate_root + evaluate_base_dir_appendix + "/" + evaluate_base_dir_appendix + "_" + env_name + "/log/closest_dist.txt"

        # INFO: load the data in files
        episodes_score = np.loadtxt(score_filename)
        normalized_episodes_score = (episodes_score - min_score) / (max_score - min_score)
        print("The averge score of method %s is %.3f" % (evaluate_base_dir_appendix, np.mean(normalized_episodes_score)))
        print("The std score of method %s is %.3f" % (evaluate_base_dir_appendix, normalized_episodes_score.std()))

        episodes_detection = np.loadtxt(detection_filename)
        normalized_episodes_detection = (episodes_detection - min_detection) / (max_detection - min_detection)
        print("The averge detection rate of method %s is %.3f" % (evaluate_base_dir_appendix, np.mean(normalized_episodes_detection)))
        print("The std detection rate of method %s is %.3f" % (evaluate_base_dir_appendix, normalized_episodes_detection.std()))

        episodes_success = np.loadtxt(success_filename)
        normalized_episodes_success = (episodes_success - min_success) / (max_sucess - min_success)
        models_successRate.append(np.mean(episodes_success))
        models_unSuccessRate.append(1-np.mean(episodes_success))
        print("The averge success rate of method %s is %.3f" % (evaluate_base_dir_appendix, np.mean(normalized_episodes_success)))
        print("The std success rate of method %s is %.3f \n" % (evaluate_base_dir_appendix, normalized_episodes_success.std()))



        episodes_time = np.loadtxt(time_filename)
        normalized_episodes_time = (episodes_time - min_time) / (max_time - min_time)
        # print("The averge episode time of method %s is %f" % (evaluate_base_dir_appendix, np.mean(episodes_time)))
        # print("The std episode time of method %s is %f" % (evaluate_base_dir_appendix, episodes_time.std()))
        # episodes_reward = episodes_agents_reward.mean(axis=-1)

        episodes_dist = np.loadtxt(dist_filename)
        normalized_episodes_dist = (episodes_dist - min_dist) / (max_dist - min_dist)
        # print("The averge closest distance of method %s is %f" % (evaluate_base_dir_appendix, np.mean(episodes_dist)))
        # print("The std closest distance of method %s is %f \n" % (evaluate_base_dir_appendix, episodes_dist.std()))

        # INFO: draw box plots for the metrics
        bp_detection = axes[0].boxplot([normalized_episodes_score], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=False, showfliers=True)
        bp_dist = axes[1].boxplot([normalized_episodes_detection], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=False, showfliers=True)
        # bp_success = axes[1].boxplot([episodes_success], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=True, showfliers=False)
        # bp_score = axes[2].boxplot([episodes_score], positions=np.array([0.2*model_idx]), sym='b+', widths=0.06, patch_artist = True, boxprops=boxprops, medianprops=medianprops, notch=True, showfliers=True)

        # set_box_color(bp_detection_rate, colors[model_idx])
        # set_box_color(bp_closest_dist, colors[model_idx])
        # set_box_color(bp_reward, colors[model_idx])

    bottom = np.zeros([len(models_successRate)])
    bar_ticks = 0.2*np.arange(len(models_successRate))
    width = 0.1
    p = axes[2].bar(bar_ticks, models_successRate, width, label="Reached", bottom=bottom, color=colors[:len(models_successRate)], edgecolor='k', linewidth=2)
    # bottom = bottom + np.array(models_successRate)
    # p = axes[2].bar(bar_ticks, models_unSuccessRate, width, label="Non-Reached", bottom=bottom)
    # Draw a horizontal line at y=2
    axes[2].axhline(y=1, color='r', linestyle='--')
    axes[2].set_ylim(0, 1.1)
    # axes[2].legend(loc="upper right", fontsize=17)

    # for boolean, weight_count in models_successRate:
    #     p = axes[2].bar(ticks_in_metric, weight_count, width, label=boolean, bottom=bottom)
    #     bottom += weight_count

    for metric_idx in range(metrics_num):
        # axes[metric_idx].set_title(metrics_total_names[metric_idx], fontsize=20)
        axes[metric_idx].set_ylabel(metrics_total_names[metric_idx], fontsize=20)
        # ticks_in_metric = model_ticks[metric_idx]
        axes[metric_idx].set_xlim(-0.2, model_num*0.2)
        axes[metric_idx].set_xticks((0.2*np.arange(model_num)).tolist())
        axes[metric_idx].set_xticklabels(ticks_in_metric, rotation=45, ha='right', color='k', size=17)  
        for xtick, color in zip(axes[metric_idx].get_xticklabels(), color_in_metric):
            xtick.set_color(color)
        axes[metric_idx].tick_params(axis='y', labelsize=15)
        axes[metric_idx].set_facecolor('whitesmoke')
        # axes[metric_idx].tick_params(labelbottom = False)
        if metric_idx < 2:
            axes[metric_idx].grid(visible=True)

    # fig.legend(fontsize=15,loc='lower center', mode = "expand", ncol = 3)
    fig.tight_layout()
    # plt.subplots_adjust(bottom=0.42)
    plt.savefig('red_rl_metric.png', dpi=100)

if __name__ == '__main__':
    plot_final_detect_dist(env_name="prisoner")