""" This script collects a triple of agent observations, hideout location, and timestep. """
# import wandb
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from simulator.load_environment import load_environment
from simulator import PrisonerRedEnv
from diffuser.datasets.multipath import StateOnlyDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.single_state import SingleStateNet
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import argparse
import random
import time

def _select_device():
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def to_device(x, device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, (list, tuple)): return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict): return {k: to_device(v, device) for k, v in x.items()}
    return x

global_device_name = _select_device()
global_device = torch.device(global_device_name)
print(f"[Device] Using {global_device_name} (MPS available: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}, CUDA available: {torch.cuda.is_available()})")


mse_loss_func = torch.nn.MSELoss()

def red_diffusion_train(epsilon, num_runs, 
                    starting_seed, 
                    random_cameras, 
                    folder_name,
                    heuristic_type,
                    blue_type,
                    env_path,
                    continue_training_estimator_flag,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """

    diffusion_train_dataset = StateOnlyDataset(folder_path="./datasets/balance_game/gnn_map_0_run_1_RRTStarOnly/train", 
                                         horizon=10,
                                         dataset_type = "prisoner_globe",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=10,)
    diffusion_valid_dataset = StateOnlyDataset(folder_path="./datasets/balance_game/gnn_map_0_run_1_RRTStarOnly/valid", 
                                         horizon=10,
                                         dataset_type = "prisoner_globe",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=10,)

    if heuristic_type == 'diffusion':
        path = f"./RAL_2024_logs/{folder_name}/start_seed_{starting_seed}_run_{num_runs}_{heuristic_type}"
        os.makedirs(path, exist_ok=True)
        writer = SummaryWriter(path+"/logs")
        # also keep a copy in a stable location
        public_ckpt_dir = "./saved_models/diffusions"
        os.makedirs(public_ckpt_dir, exist_ok=True)
        # INFO: Red-only diffusions
        diffusion_net = SingleStateNet(transition_dim=2, horizon=10, global_cond_dim=8, lstm_out_dim=None, num_agents=1).to(global_device)
        diffusion_model = GaussianDiffusion(model=diffusion_net, horizon=10, observation_dim=2, action_dim=0, n_timesteps=10, predict_epsilon=False).to(global_device)
        diffusion_optimizer = Adam(diffusion_model.parameters(), lr=0.00002, weight_decay=0.0005)

        diffusion_train_dataloader = (torch.utils.data.DataLoader(diffusion_train_dataset, batch_size=128, num_workers=0, shuffle=True, pin_memory=False, collate_fn=diffusion_train_dataset.collate_fn))
        diffusion_valid_dataloader = (torch.utils.data.DataLoader(diffusion_valid_dataset, batch_size=128, num_workers=0, shuffle=True, pin_memory=False, collate_fn=diffusion_valid_dataset.collate_fn))

        epoch_num = 100
        validation_epoch_period = 2
        min_valid_loss = np.inf

        for ep_i in range(epoch_num):
            diffusion_train_losses = []

            for _, batch in tqdm(enumerate(diffusion_train_dataloader), total=len(diffusion_train_dataloader)):
                # move batch to selected device (mps/cuda/cpu)
                batch = to_device(batch, global_device)
                # forward + backward + step
                diffusion_loss, infos = diffusion_model.loss(*batch)
                diffusion_optimizer.zero_grad()
                diffusion_loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 0.1)
                diffusion_optimizer.step()
                # store detached loss to avoid holding graph refs
                diffusion_train_losses.append(diffusion_loss.detach().cpu())

            # default valid to train so logging is defined every epoch
            diffusion_valid_losses = diffusion_train_losses

            if ep_i % validation_epoch_period == 0:
                diffusion_valid_losses = []
                with torch.no_grad():
                    for _, batch in tqdm(enumerate(diffusion_valid_dataloader), total=len(diffusion_valid_dataloader)):
                        batch = to_device(batch, global_device)
                        diffusion_loss, infos = diffusion_model.loss(*batch)
                        diffusion_valid_losses.append(diffusion_loss.detach().cpu())

            # compute means safely
            mean_train = torch.stack(diffusion_train_losses).mean().item()
            mean_valid = torch.stack(diffusion_valid_losses).mean().item()

            # save best model
            if mean_valid < min_valid_loss:
                min_valid_loss = mean_valid
                # save into the run folder
                torch.save(diffusion_model, path + "/diffusion.pth")
                # and also save/update a public checkpoint for inference scripts
                torch.save(diffusion_model, os.path.join(public_ckpt_dir, "diffusion.pth"))

            # log
            writer.add_scalars('loss', {'train_loss': mean_train, 'valid_loss': mean_valid}, ep_i)
            print("Trajectory eps:", ep_i)
            print("min_valid_loss:", min_valid_loss)

        

def red_diffusion_test():
    def interpolate_paths(paths, total_dense_path_num):
        batchsize, waypt_num, coordinates = paths.shape
        new_paths = np.zeros((batchsize, total_dense_path_num, coordinates))

        for i in range(batchsize):
            for j in range(coordinates):
                new_paths[i, :, j] = np.interp(
                    np.linspace(0, 1, total_dense_path_num),
                    np.linspace(0, 1, waypt_num),
                    paths[i, :, j]
                )

        return new_paths

    diffusion_dataset = StateOnlyDataset(folder_path="./datasets/balance_game/gnn_map_0_run_1_RRTStarOnly/test", 
                                         horizon=10,
                                         dataset_type = "prisoner_globe",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=10,)

    diffusion_model = torch.load("./saved_models/diffusions/diffusion.pth", map_location=global_device)
    diffusion_model.to(global_device).eval()
    
    # INFO: draw samples from current diffusion model
    env = load_environment(env_path)
    env = PrisonerRedEnv(env, blue_policy=None)
    env.reset(seed=3) # 3, 6
    hideout_division = [50, 50, 50] # [0, 0, 1], [25, 25, 25]
    global_cond, local_cond = env.construct_diffusion_conditions(cond_on_hideout_num=hideout_division)
    global_cond = to_device(global_cond, global_device)  
    local_cond  = to_device(local_cond,  global_device)

    # INFO: set random seed for the diffusion denoise process
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    diffusion_model.n_timesteps = 10

    start_time = time.time()
    sample = diffusion_model.conditional_sample(global_cond=global_cond, cond=local_cond, sample_type="constrained")
    end_time = time.time()
    print("Diffusion takes %f seconds." % (end_time-start_time))
    sample = diffusion_dataset.unnormalize(sample).detach().cpu().numpy()
    dense_path = interpolate_paths(sample, total_dense_path_num=100)
    hideouts = diffusion_dataset.unnormalize(global_cond["hideouts"].reshape(-1, 3, 2)).detach().cpu().numpy()
    starts = diffusion_dataset.unnormalize(global_cond["red_start"]).detach().cpu().numpy()
    figure, axes = plt.subplots()
    for sample_idx in range(sum(hideout_division)):
        if all(np.linalg.norm(dense_path[sample_idx] - np.array([[1600, 1800]]), axis=1) > 140):
            axes.plot(sample[sample_idx,:,0], sample[sample_idx,:,1])
            # axes.scatter(sample[sample_idx,:,0], sample[sample_idx,:,1], s=50, c='r')
        axes.scatter(hideouts[sample_idx,:,0], hideouts[sample_idx,:,1], s=300, c='gold', marker='*')
        axes.scatter(starts[sample_idx,0], starts[sample_idx,1], c='gold')
        axes.axis("off")
        # plt.scatter(1600, 1800, s=150)
    # circle = plt.Circle(( 1600 , 1800 ), 150 )
    # axes.add_artist( circle )
    axes.imshow(env.custom_render_canvas(option=["terrain"], show=False, large_icons=False), extent=(0, 2428, 0, 2428))
    plt.axis('square')
    plt.xlim(0, 2428)
    plt.ylim(0, 2428)
    # plt.show()
    plt.savefig("diverse_diffusion_paths_10.png", bbox_inches='tight')

def red_diffusion_rrt_time_cmp():
    heuristic_type_list = ["RRTStarOnly", "diffusion"] # "AStar_only", 
    labels = ["RRT*", "Diffusion [Ours]"] # "A*", 
    # Create the main axis
    fig, ax1 = plt.subplots(figsize=(8, 6))
    for heuristic_id, heuristic_type in enumerate(heuristic_type_list):
        exp_traj_set = np.array([1, 10, 20, 30, 40, 50])
        time_means = []
        time_stds = []
        for traj_num in exp_traj_set:
            time_mean = np.load("./datasets/balance_game/time_collection/%s/traj_num_%d_time_%s.npz" % (heuristic_type, traj_num, heuristic_type))["repeat_time"].mean()
            time_std = np.load("./datasets/balance_game/time_collection/%s/traj_num_%d_time_%s.npz" % (heuristic_type, traj_num, heuristic_type))["repeat_time"].std()
            time_means.append(time_mean)
            time_stds.append(time_std)
            print("The mean time of method %s is %.2f" % (labels[heuristic_id], time_mean))
            print("The std time of method %s is %.2f" % (labels[heuristic_id], time_std))
        # Plot the first dataset on the left axis
        ax1.plot(exp_traj_set, time_means, label=labels[heuristic_id], linewidth=3, marker='.', markersize=32)
        # ax1.errorbar(exp_traj_set, time_means, yerr=time_std, fmt='o', label='Standard Deviation')
        ax1.set_yscale('log')

    ax1.set_xlabel('Planned Trajectory Number', fontdict={"weight": "bold", "size": 22})
    ax1.set_ylabel('Time (second)', fontdict={"weight": "bold", "size": 22})

    ax1.legend(fontsize=22)

    # Set the tick label size for both x-axis and y-axis
    plt.tick_params(axis='x', labelsize=22)  # Set x-axis tick label size
    plt.tick_params(axis='y', labelsize=22)  # Set y-axis tick label size

    plt.grid(True)
    plt.title("Path Generation Time Comparison", fontsize=25, fontweight='bold')

    plt.savefig("time_cmp.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num
    continue_training_estimator_flag = False

    starting_seed=1; num_runs=1; epsilon=0; folder_name = "online_training" 

    heuristic_type = "diffusion" # "AStar_only, diffusion" 
    blue_type = "heuristic"
    random_cameras=False

    env_path = "simulator/configs/balance_game.yaml"
    print("[Run] Starting diffusion training on device:", global_device)

    # run training (this will also save a copy of the best model to ./saved_models/diffusions/diffusion.pth)
    red_diffusion_train(
        epsilon,
        num_runs,
        starting_seed,
        random_cameras,
        folder_name,
        heuristic_type,
        blue_type,
        env_path,
        continue_training_estimator_flag,
        show=False
    )
    # To evaluate after training, you can comment the above and uncomment the next line:
    # red_diffusion_test()