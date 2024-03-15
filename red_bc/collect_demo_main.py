import os
import argparse
import numpy as np
import random
import torch
from pathlib import Path
from collect_demo_utils import collect_demo
import sys
project_path = os.getcwd()
sys.path.append(str(project_path))
from simulator import PrisonerBothEnv, PrisonerBlueEnv, PrisonerEnv, PrisonerRedEnv
from fugitive_policies.heuristic import HeuristicPolicy
from red_bc.heuristic import BlueHeuristic, SimplifiedBlueHeuristic
from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from simulator.load_environment import load_environment

def run(args):

    print(f"Loaded environment with seed {args.seed}")

    env = load_environment('simulator/configs/red_bc.yaml')
    
    # set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    epsilon = 0.1
    
    """Environment for red team is constructed here"""
    env = PrisonerRedEnv(env, blue_policy=None)
    blue_policy = SimplifiedBlueHeuristic(env, debug=False)
    env.blue_policy = blue_policy
    """Blue heuristic is used to generate expert trajectory"""
    red_heuristic = RRTStarAdversarialAvoid(env, n_iter=1500, max_speed=7.5)
    # red_heuristic = HeuristicPolicy(env, epsilon=epsilon)
    env.red_policy = red_heuristic


    buffer = collect_demo(
        env=env,
        blue_heuristic=blue_policy,
        red_policy=red_heuristic,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='RedFixNormal')
    p.add_argument('--buffer_size', type=int, default=1000)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    run(args)
