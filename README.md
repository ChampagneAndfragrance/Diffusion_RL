# Diffusion_RL
This repo has the code and suplementary materials of our 2024 RAL submission "Diffusion-Reinforcement Learning Hierarchical Motion Planning in
Adversarial Multi-agent Games".
## Code Introduction
* Our code is mainly in the folder `red_rl_diffusion`.
* You can first collect the RRT* dataset with the file `collect_demonstrations_balance_game.py` in the folder `datasets`.
* Then the diffusion model can be trained with the function `red_diffusion_train` in the file `red_diffusion_main.py`.
* Next, our diffusion-RL model and DDPG, SAC baselines can be trained from `red_rl_main.py` file with the functions in it.
* You can use the file `red_rl_test.py` to generate the benchmarking results including learning based methods and heuristics.
* The results can be visualized with the file `red_rl_analysis.py`. It will print out the benchmarking results and generate boxplots.
* Remember all the log files and checkpoints of the model you trained will be in the folder `logs`.
## Conda Environment
* Please check the file `environment.yml` for the package versions we are using.
## Video
<img src="https://github.com/ChampagneAndfragrance/Diffusion_RL/blob/main/logs/RAL2024/benchmark_results/Diffusion_RL/Diffusion_RL_prisoner/video/demo0.gif" width="100" height="100" />
## Citations
* We adapt our diffusion model training code from [`Diffuser`](https://github.com/jannerm/diffuser) with the following citation:
```c
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


