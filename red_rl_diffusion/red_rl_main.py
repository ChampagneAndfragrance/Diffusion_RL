import os
import time
from pathlib import Path

from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
import sys
import yaml

project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator.forest_coverage.autoencoder import train
from simulator.prisoner_perspective_envs import PrisonerRedEnv
import matplotlib
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from blue_bc.maddpg import BaseMADDPG
from SAC.sac import SAC
from red_bc.heuristic import BlueHeuristic


matplotlib.use("Agg")
import matplotlib.pylab
from red_bc.utils import save_video
from config_loader import config_loader
import random
from simulator.load_environment import load_environment
from blue_bc.buffer import ReplayBuffer, Buffer

# from prioritized_memory import Memory
from diffuser.datasets.multipath import NAgentsIncrementalDataset
from fugitive_policies.diffusion_policy import DiffusionStateOnlyGlobalPlanner

from enum import Enum, auto


# --- Device selection helper and globals ---
from utils.device import global_device, global_device_name, to_device, print_device
print_device()


# Move nested tensors/arrays to the selected device
def to_device(x, device):
    import torch, numpy as np

    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def _doc_to_device():
    """(Documentation helper) to_device

    The actual device-migration helper above moves nested torch tensors or
    numpy arrays to the provided `device`. This small placeholder docstring
    provides quick developer reference in IDEs without altering runtime
    behavior. The functional implementation is `to_device(x, device)`.

    Note: this helper is intentionally a no-op at runtime and exists solely
    to keep module-level documentation discoverable.
    """


def red_rl_baseline(config, env_config):
    """Train a MADDPG baseline for the red (fugitive) agent.

    This function sets up logging directories, initializes the environment and
    the MADDPG agent, runs episodes, collects experience into a replay buffer,
    and performs periodic updates/checkpointing. It is intended for
    subpolicy-level training with the project's MADDPG implementation.

    Args:
        config (dict): training and environment configuration dictionary
            (typically loaded via `config_loader`).
        env_config (dict): environment-specific configuration dictionary
            referenced by `load_environment`.

    Returns:
        None. Checkpoints and logs are written to disk under the configured
        `config['environment']['dir_path']`.
    """
    # INFO: set up file and folder structure
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)

    # INFO: Save the config into the para dir
    with open(parameter_dir / "parameters_network.yaml", "w") as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", "w") as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)

    # --- Device policy ---
    # IMPORTANT: Keep models and tensors on the SAME device to avoid slow transfers.
    # Your BaseMADDPG only switches between 'gpu' (CUDA) and 'cpu'. Since CUDA isn't
    # available on Apple Silicon, we keep everything on CPU (fastest consistent path).
    # If you later add real CUDA, this will automatically use it.
    use_cuda = torch.cuda.is_available()
    lib_dev = "gpu" if use_cuda else "cpu"  # for maddpg.prep_*()
    tensor_device = torch.device("cuda" if use_cuda else "cpu")  # for torch tensors

    # Optional: if env.step is Python-heavy, letting PyTorch grab many threads can hurt.
    try:
        torch.set_num_threads(int(config["train"].get("torch_num_threads", 4)))
    except Exception:
        pass

    # INFO: Load the environment
    epsilon = 0.1
    variation = 0
    print(
        "Loaded environment variation %d with seed %d"
        % (variation, config["environment"]["seed"])
    )
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])

    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)
    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())

    # INFO: Load the maddpg model
    agent_num = 1  # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"]
    filtering_input_dims = [
        [0] for _ in range(agent_num)
    ]  # no filter input from the fugitive perspective
    obs_dims = [red_observation[i].shape[0] for i in range(agent_num)]
    ac_dims = [action_dim_per_agent for _ in range(agent_num)]
    loc_dims = [len(prisoner_loc) for _ in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]

    maddpg = BaseMADDPG(
        agent_num=agent_num,
        num_in_pol=red_observation[0].shape[0],
        num_out_pol=action_dim_per_agent,
        num_in_critic=(red_observation[0].shape[0] + action_dim_per_agent) * agent_num,
        discrete_action=False,
        gamma=config["train"]["gamma"],
        tau=config["train"]["tau"],
        critic_lr=config["train"]["critic_lr"],
        policy_lr=config["train"]["policy_lr"],
        hidden_dim=config["train"]["hidden_dim"],
        device=("cuda" if use_cuda else "cpu"),
    )

    if config["train"]["continue"]:
        maddpg.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split(".")
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(
        config["train"]["buffer_size"],
        agent_num,
        buffer_dims=obs_ac_dims,
        is_cuda=use_cuda,
    )

    # --- Training-time knobs (faster defaults if not provided in YAML) ---
    max_steps_per_episode = int(
        config["train"].get("max_steps_per_episode", 500)
    )  # cap long episodes
    render_every_step = int(
        config["train"].get("render_every_step", 10)
    )  # throttle in-episode frames
    video_step = int(config["train"].get("video_step", 999999))  # rare video by default
    save_interval = int(
        config["train"].get("save_interval", 100)
    )  # checkpoint less often
    batch_size = int(config["train"].get("batch_size", 128))  # avoid tiny batches

    pbar_maddpg = trange(recent_episode, config["train"]["episode_num"], desc="MADDPG Episodes", unit="ep")
    for ep in pbar_maddpg:
        # Keep model on the chosen device for rollout
        maddpg.prep_rollouts(device=lib_dev)
        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep) / max(
            1, config["train"]["n_exploration_eps"]
        )
        maddpg.scale_noise(
            config["train"]["final_noise_scale"]
            + (
                config["train"]["init_noise_scale"]
                - config["train"]["final_noise_scale"]
            )
            * explr_pct_remaining
        )
        maddpg.reset_noise()

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset(seed=ep)
        # (Removed) incremental_dataset = NAgentsIncrementalDataset(env)  # Unused and adds overhead

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False

        while not done and t < max_steps_per_episode:
            t += 1

            # Convert observations to tensors ON THE SAME DEVICE AS THE MODEL/TENSORS WE USE
            torch_red_observation = [
                torch.as_tensor(
                    red_observation[i], dtype=torch.float32, device=tensor_device
                )
                for i in range(maddpg.nagents)
            ]

            # Policy step (exploration enabled)
            with torch.no_grad():
                torch_agent_actions = maddpg.step(torch_red_observation, explore=True)

            # Move actions to numpy for env.step; detach avoids autograd overhead, and .cpu() ensures numpy conversion
            agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]

            next_red_observation, rewards, done, i, _, red_detected_flag = env.step(
                split_red_directions_to_direction_speed((np.concatenate(agent_actions)))
            )

            # Only push to buffer when not on video-episode or regardless? (Keeping your original logic)
            if ep % video_step != 0:
                replay_buffer.push(
                    red_observation, agent_actions, rewards, next_red_observation, done
                )

            red_observation = next_red_observation
            prisoner_loc = copy.deepcopy(env.get_prisoner_location())

            # Throttled rendering inside video episodes
            if ep % video_step == 0 and (t % render_every_step == 0):
                game_img = env.render("Policy", show=False, fast=True)
                imgs.append(game_img)


        # Save video rarely
        if ep % video_step == 0 and len(imgs) > 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)

        # Checkpoint less frequently to reduce I/O stalls
        if ep % save_interval == 0:
            maddpg.save(model_dir / (str(ep) + ".pth"))
            maddpg.save(base_dir / ("model.pth"))

        # Training updates: keep model and sampled batch on the SAME device
        if (
            len(replay_buffer) >= 2 * batch_size
        ):  # update every time you have enough data
            maddpg.prep_training(device=("gpu" if use_cuda else "cpu"))

            for a_i in range(maddpg.nagents):
                sample = replay_buffer.sample(
                    batch_size, to_gpu=use_cuda, norm_rews=False
                )
                # Move the (obs, act, rew, next_obs, done) batch to the exact tensor device we use elsewhere
                sample = to_device(sample, tensor_device)
                maddpg.update(sample, a_i, train_option="regular", logger=logger)

            maddpg.update_all_targets()

        # Empty MPS cache if needed (kept; harmless on CPU/CUDA)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar("agent%i/mean_episode_rewards" % a_i, a_ep_rew, ep)

        # Print progress in a tqdm-safe way
        print_every_ep = int(config["train"].get("print_every_ep", 100))
        if ((ep + 1) % print_every_ep == 0) or (ep + 1 == int(config["train"]["episode_num"])):
            pbar_maddpg.write("complete %f of the training" % ((ep + 1) / float(config["train"]["episode_num"])))

    return


def red_rl_baseline_sac(config, env_config):
    """Train a SAC baseline for the red (fugitive) agent.

    Similar to `red_rl_baseline` but using the SAC implementation. It
    initializes SAC, runs environment rollouts, populates a ReplayBuffer,
    performs updates, and writes checkpoints and optional videos.

    Args:
        config (dict): training and environment configuration dictionary.
        env_config (dict): environment configuration for `load_environment`.

    Returns:
        None. Artifacts are saved to the configured directories.
    """
    # INFO: set up file and folder structure
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)
    # INFO: Save the config into the para dir
    with open(parameter_dir / "parameters_network.yaml", "w") as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", "w") as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)
    # INFO: Load the environment
    device = global_device_name
    epsilon = 0.1
    variation = 0
    print(
        "Loaded environment variation %d with seed %d"
        % (variation, config["environment"]["seed"])
    )
    # set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset()
    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    # INFO: Load the sac model
    agent_num = 1  # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"]
    filtering_input_dims = [
        [0] for i in range(agent_num)
    ]  # no filter input for from the fugitive perspective
    obs_dims = [red_observation[i].shape[0] for i in range(agent_num)]
    ac_dims = [action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    sac = SAC(
        num_in_pol=red_observation[0].shape[0],
        num_out_pol=action_dim_per_agent,
        num_in_critic=(red_observation[0].shape[0] + action_dim_per_agent) * agent_num,
        discrete_action=False,
        gamma=config["train"]["gamma"],
        tau=config["train"]["tau"],
        critic_lr=config["train"]["critic_lr"],
        policy_lr=config["train"]["policy_lr"],
        entropy_lr=config["train"]["entropy_lr"],
        hidden_dim=config["train"]["hidden_dim"],
        policy_type=config["train"]["policy_type"],
        device=device,
        constrained=False,
    )
    if config["train"]["continue"]:
        sac.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split(".")
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(
        config["train"]["buffer_size"],
        agent_num,
        buffer_dims=obs_ac_dims,
        is_cuda=torch.cuda.is_available(),
    )
    # --- Training-time knobs (align with baseline) ---
    max_steps_per_episode = int(config["train"].get("max_steps_per_episode", 500))
    render_every_step = int(config["train"].get("render_every_step", 10))
    video_step = int(config["train"].get("video_step", 999999))
    save_interval = int(config["train"].get("save_interval", 100))
    batch_size = int(config["train"].get("batch_size", 128))

    pbar_sac = trange(recent_episode, config["train"]["episode_num"], desc="SAC Episodes", unit="ep")
    for ep in pbar_sac:

        # INFO: Start a new episode
        red_observation, red_partial_observation = env.reset()
        # incremental_dataset = NAgentsIncrementalDataset(env)  # Removed unused dataset creation

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False
        while not done and t < max_steps_per_episode:
            # INFO: run episode
            t = t + 1
            if ep % video_step == 0:
                # INFO: Use the same policy to explore
                torch_red_observation = [
                    torch.as_tensor(
                        red_observation[i], dtype=torch.float32, device=global_device
                    )
                    for i in range(agent_num)
                ]
                with torch.no_grad():
                    torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [
                    ac.detach().cpu().numpy() for ac in torch_agent_actions
                ]
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(
                    split_red_directions_to_direction_speed(
                        (np.concatenate(agent_actions))
                    )
                )
            else:
                # INFO: Use the same policy to explore
                torch_red_observation = [
                    torch.as_tensor(
                        red_observation[i], dtype=torch.float32, device=global_device
                    )
                    for i in range(agent_num)
                ]
                with torch.no_grad():
                    torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [
                    ac.detach().cpu().numpy() for ac in torch_agent_actions
                ]
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(
                    split_red_directions_to_direction_speed(
                        (np.concatenate(agent_actions))
                    )
                )
                replay_buffer.push(
                    red_observation, agent_actions, rewards, next_red_observation, done
                )
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % video_step == 0 and (t % render_every_step == 0):
                game_img = env.render("Policy", show=False, fast=True)
                imgs.append(game_img)

        if ep % video_step == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % save_interval == 0:
            sac.save(model_dir / (str(ep) + ".pth"))
            sac.save(base_dir / ("model.pth"))

        if (
            len(replay_buffer) >= 2 * batch_size
        ):  # update every config["train"]["steps_per_update"] steps
            for a_i in range(agent_num):
                sample = replay_buffer.sample(
                    batch_size,
                    to_gpu=torch.cuda.is_available(),
                    norm_rews=False,
                )
                sample = to_device(sample, global_device)
                sac.update_bl(sample, a_i, train_option="regular", logger=logger)

        # Empty MPS cache if needed
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar("agent%i/mean_episode_rewards" % a_i, a_ep_rew, ep)

        # Print progress in a tqdm-safe way
        print_every_ep = int(config["train"].get("print_every_ep", 100))
        if ((ep + 1) % print_every_ep == 0) or (ep + 1 == int(config["train"]["episode_num"])):
            pbar_sac.write("complete %f of the training" % ((ep + 1) / float(config["train"]["episode_num"])))

    return


def red_rl_piece_sac(config, env_config):
    """Train a piecewise (waypoint) SAC policy guided by a diffusion planner.

    This routine composes a diffusion-based global planner with a SAC low-level
    controller. It initializes the DiffusionStateOnlyGlobalPlanner, creates
    the SAC agent, and runs episodes where the diffusion planner provides
    guidance for waypoint-level actions. Standard logging, checkpointing, and
    buffer updates are performed.

    Args:
        config (dict): training and environment configuration dictionary.
        env_config (dict): environment-specific options for environment loader.

    Returns:
        None. Models and logs are stored under the configured `dir_path`.
    """
    # INFO: set up file and folder structure
    base_dir = Path(config["environment"]["dir_path"])
    log_dir = base_dir / "log"
    video_dir = base_dir / "video"
    model_dir = base_dir / "model"
    dataset_dir = base_dir / "data"
    parameter_dir = base_dir / "parameter"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(parameter_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # INFO: specify the writer
    logger = SummaryWriter(log_dir=log_dir)

    # INFO: Save the config into the para dir
    with open(parameter_dir / "parameters_network.yaml", "w") as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    with open(parameter_dir / "parameters_env.yaml", "w") as para_yaml:
        yaml.dump(env_config, para_yaml, default_flow_style=False)

    # INFO: Load the environment
    device = global_device_name
    epsilon = 0.1
    variation = 0
    print(
        "Loaded environment variation %d with seed %d"
        % (variation, config["environment"]["seed"])
    )
    # INFO: set seeds
    np.random.seed(config["environment"]["seed"])
    random.seed(config["environment"]["seed"])
    env = load_environment(env_config)
    env.gnn_agent_last_detect = config["environment"]["gnn_agent_last_detect"]

    blue_policy = BlueHeuristic(env, debug=False)

    diffusion_path = "./saved_models/diffusions/diffusion.pth"
    red_policy = DiffusionStateOnlyGlobalPlanner(
        env, diffusion_path, plot=False, traj_grader_path=None
    )

    env = PrisonerRedEnv(env, blue_policy)

    # INFO: Reset the environment
    red_observation, red_partial_observation = env.reset(
        seed=None, reset_type=None, red_policy=red_policy
    )

    prisoner_loc = copy.deepcopy(env.get_prisoner_location())
    # INFO: Load the sac model
    agent_num = 1  # there is only one fugitive
    action_dim_per_agent = 2 + env_config["comm_dim"]
    filtering_input_dims = [
        [0] for i in range(agent_num)
    ]  # no filter input for from the fugitive perspective
    obs_dims = [red_observation[i].shape[0] for i in range(agent_num)]
    ac_dims = [action_dim_per_agent for i in range(agent_num)]
    loc_dims = [len(prisoner_loc) for i in range(agent_num)]
    obs_ac_dims = [obs_dims, ac_dims]
    obs_ac_filter_loc_dims = [obs_dims, ac_dims, filtering_input_dims, loc_dims]
    sac = SAC(
        num_in_pol=red_observation[0].shape[0],
        num_out_pol=action_dim_per_agent,
        num_in_critic=(red_observation[0].shape[0] + action_dim_per_agent) * agent_num,
        discrete_action=False,
        gamma=config["train"]["gamma"],
        tau=config["train"]["tau"],
        critic_lr=config["train"]["critic_lr"],
        policy_lr=config["train"]["policy_lr"],
        entropy_lr=config["train"]["entropy_lr"],
        hidden_dim=config["train"]["hidden_dim"],
        policy_type=config["train"]["policy_type"],
        device=device,
        constrained=False,
    )
    if config["train"]["continue"]:
        sac.init_from_save(config["train"]["para_file"])
        pth_files = os.listdir(Path(config["train"]["para_file"]).parent / "model")
        recent_episode = 0
        for pth_file in pth_files:
            episode_pth = pth_file.split(".")
            episode = int(episode_pth[0])
            if episode > recent_episode:
                recent_episode = episode
    else:
        recent_episode = 0

    # INFO: Initialize the buffer
    replay_buffer = ReplayBuffer(
        config["train"]["buffer_size"],
        agent_num,
        buffer_dims=obs_ac_dims,
        is_cuda=torch.cuda.is_available(),
    )
    # --- Training-time knobs (align with baseline) ---
    max_steps_per_episode = int(config["train"].get("max_steps_per_episode", 500))
    render_every_step = int(config["train"].get("render_every_step", 10))
    video_step = int(config["train"].get("video_step", 999999))
    save_interval = int(config["train"].get("save_interval", 100))
    batch_size = int(config["train"].get("batch_size", 128))

    pbar_dsac = trange(recent_episode, config["train"]["episode_num"], desc="Diffusion+SAC Episodes", unit="ep")
    for ep in pbar_dsac:

        # INFO: Set the dist penalty coefficient
        env.set_dist_coeff(ep, config["train"]["dist_coeff_episode_num"], 0.05)

        # INFO: Start a new episode

        red_observation, red_partial_observation = env.reset(
            seed=ep, reset_type=None, red_policy=red_policy, waypt_seed=ep
        )
        # incremental_dataset = NAgentsIncrementalDataset(env)  # Removed unused dataset creation

        prisoner_loc = copy.deepcopy(env.get_prisoner_location())
        t = 0
        imgs = []
        done = False

        while not done and t < max_steps_per_episode:
            # INFO: run episode
            t = t + 1

            if ep % video_step == 0:
                # INFO: Use diffusion guidance
                torch_red_observation = [
                    torch.as_tensor(
                        red_observation[i], dtype=torch.float32, device=global_device
                    )
                    for i in range(agent_num)
                ]
                with torch.no_grad():
                    torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [
                    ac.detach().cpu().numpy() for ac in torch_agent_actions
                ]
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(
                    split_red_directions_to_direction_speed(
                        (np.concatenate(agent_actions))
                    )
                )
            else:
                # INFO: Use diffusion guidance
                torch_red_observation = [
                    torch.as_tensor(
                        red_observation[i], dtype=torch.float32, device=global_device
                    )
                    for i in range(agent_num)
                ]
                with torch.no_grad():
                    torch_agent_actions = sac.select_action(torch_red_observation)
                agent_actions = [
                    ac.detach().cpu().numpy() for ac in torch_agent_actions
                ]
                next_red_observation, rewards, done, i, _, red_detected_flag = env.step(
                    split_red_directions_to_direction_speed(
                        (np.concatenate(agent_actions))
                    )
                )
                replay_buffer.push(
                    red_observation, agent_actions, rewards, next_red_observation, done
                )
            # next_torch_red_observation = [Variable(torch.Tensor(next_red_observation[i]), requires_grad=False).to(device) for i in range(maddpg.nagents)]
            next_prisoner_loc = copy.deepcopy(env.get_prisoner_location())

            red_observation = next_red_observation
            prisoner_loc = next_prisoner_loc

            if ep % video_step == 0 and (t % render_every_step == 0):
                game_img = env.render("Policy", show=False, fast=True)
                imgs.append(game_img)

        if ep % video_step == 0:
            video_path = video_dir / (str(ep) + ".mp4")
            save_video(imgs, str(video_path), fps=10)
        if ep % save_interval == 0:
            sac.save(model_dir / (str(ep) + ".pth"))
            sac.save(base_dir / ("model.pth"))

        if len(replay_buffer) >= 2 * batch_size:
            for a_i in range(agent_num):
                sample = replay_buffer.sample(
                    batch_size,
                    to_gpu=torch.cuda.is_available(),
                    norm_rews=False,
                )
                sample = to_device(sample, global_device)
                sac.update_bl(sample, a_i, train_option="regular", logger=logger)

        # Empty MPS cache if needed
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        ep_rews = replay_buffer.get_average_rewards(t)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar("agent%i/mean_episode_rewards" % a_i, a_ep_rew, ep)

        # Print progress in a tqdm-safe way
        print_every_ep = int(config["train"].get("print_every_ep", 100))
        if ((ep + 1) % print_every_ep == 0) or (ep + 1 == int(config["train"]["episode_num"])):
            pbar_dsac.write("complete %f of the training" % ((ep + 1) / float(config["train"]["episode_num"])))

    return


def split_red_directions_to_direction_speed(directions):
    """Convert a raw direction vector into [speed, angle] action format.

    The project uses an action representation where the first element is the
    desired speed (scaled by `fugitive_v_limit`) and the second element is
    the heading angle in radians. This helper normalizes the input direction
    vector if necessary and returns a numpy array [speed, angle].

    Args:
        directions (array-like): flattened direction vector(s) for the red agent.

    Returns:
        numpy.ndarray: a 2-element array [speed, angle] suitable for environment.step.
    """
    red_actions_norm_angle_vel = []
    red_actions_directions = np.split(directions, 1)
    fugitive_v_limit = 15
    for idx in range(len(red_actions_directions)):
        fugitive_direction = red_actions_directions[idx]
        if np.linalg.norm(fugitive_direction) > 1:
            fugitive_direction = fugitive_direction / np.linalg.norm(fugitive_direction)
        fugitive_speed = (
            np.minimum(np.linalg.norm(fugitive_direction), 1.0) * fugitive_v_limit
        )
        red_actions_norm_angle_vel.append(
            np.array(
                [
                    fugitive_speed,
                    np.arctan2(fugitive_direction[1], fugitive_direction[0]),
                ]
            )
        )
    return red_actions_norm_angle_vel[0]


if __name__ == "__main__":
    config = config_loader(
        path="./red_rl_diffusion/configs/parameters_training_combine.yaml"
    )  # load model configuration
    env_config = config_loader(path=config["environment"]["env_config_file"])
    """create base dir"""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("./logs/marl") / timestr
    os.makedirs(base_dir, exist_ok=True)
    """Benchmark Starts Here"""
    # INFO: Specify the benchmarking parameters: random seeds, learning rates
    seeds = [0]
    critic_lrs = [0.003]
    policy_lrs = [0.003]
    entropy_lrs = [0.003]
    threat_lrs = [0.003]
    load_checkpoint = bool(config["train"].get("continue", False))
    start_episode = 0
    if load_checkpoint:
        print("\033[33mYou are loading checkpoint\033[33m")
    else:
        print("\033[33mYou are NOT loading checkpoint\033[33m")

    for seed in seeds:
        for c_lr in critic_lrs:
            for p_lr in policy_lrs:
                for e_lr in entropy_lrs:
                    """Modify the config"""
                    config["environment"]["seed"] = seed
                    config["train"]["critic_lr"] = c_lr
                    config["train"]["policy_lr"] = p_lr
                    config["train"]["entropy_lr"] = e_lr
                    config["train"]["continue"] = load_checkpoint
                    config["train"]["start_episode"] = start_episode

                    """create base dir name for each setting"""
                    base_dir = (
                        Path("./logs/marl")
                        / timestr
                        / (
                            config["train"]["policy_type"]
                            + "_"
                            + config["train"]["path_type"]
                        )
                    )
                    config["environment"]["dir_path"] = str(base_dir)
                    if (
                        config["train"]["policy_level"] == "subpolicy"
                        and config["train"]["policy_type"] == "ddpg"
                    ):
                        # INFO: DDPG baseline training
                        red_rl_baseline(config, env_config)
                    elif (
                        config["train"]["policy_level"] == "subpolicy"
                        and config["train"]["policy_type"] == "sac"
                        and config["train"]["model_type"] == "free"
                        and config["train"]["path_type"] == "whole"
                    ):
                        # INFO: SAC baseline training
                        red_rl_baseline_sac(config, env_config)
                    elif (
                        config["train"]["policy_level"] == "subpolicy"
                        and config["train"]["policy_type"] == "sac"
                        and config["train"]["model_type"] == "free"
                        and config["train"]["path_type"] == "piece"
                    ):
                        # INFO: Diffusion + RL training
                        red_rl_piece_sac(config, env_config)
                    else:
                        raise NotImplementedError
