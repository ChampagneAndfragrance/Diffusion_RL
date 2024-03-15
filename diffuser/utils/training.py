from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import copy
import numpy as np
import torch
import einops
import pdb
from diffuser.datasets.prisoner import pad_collate, pad_collate_global, pad_collate_detections, pad_collate_detections_repeat, wrap_prisoner_a_star
from diffuser.utils.stats import get_stats_batch

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        test_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=wrap_prisoner_a_star
        ))
        self.test_dataset = test_dataset

        self.n_test_samples = 1
        self.n_test_batch_size = 10
        batch_collate_fn = lambda batch: pad_collate_detections_repeat(batch, self.n_test_samples)
        self.test_dataloader = cycle(torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.n_test_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=wrap_prisoner_a_star))

        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True, collate_fn=wrap_prisoner_a_star
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.writer = SummaryWriter(self.logdir)

        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            running_loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                # batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                running_loss += loss.item()

            # print(f'{self.logdir}/training_loss.txt')
            # with open(f'{self.logdir}/training_loss.txt', 'a') as f:
            #     f.write(f'{self.step},{running_loss}\n')
            self.writer.add_scalar('training_loss', running_loss, self.step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            if self.sample_freq and self.step % self.sample_freq == 0:
                gt_list = []
                sample_list = []
                # add test dataset here
                with torch.no_grad():
                    # running_test_loss = 0.0
                    # for _ in range(self.gradient_accumulate_every):
                    #     batch = next(self.test_dataloader)
                    #     loss, infos = self.model.loss(*batch)
                    #     running_test_loss += loss / self.gradient_accumulate_every
                    # self.writer.add_scalar('test_loss', running_test_loss, self.step)
                    for i in range(25):
                        # record ade
                        batch = self.test_dataloader.__next__()
                        data, global_cond, conditions = batch
                        sample_type = 'original'
                        samples = self.model.conditional_sample(global_cond, conditions, sample_type=sample_type)
                        samples = to_np(samples)

                        ## [ n_samples x horizon x observation_dim ]
                        normed_observations = samples

                        ## [ n_samples x (horizon + 1) x observation_dim ]
                        observations = self.test_dataset.unnormalize(normed_observations)

                        gt_path = self.test_dataset.unnormalize(batch[0])
                        gt_list.append(gt_path)
                        sample_list.append(observations)

                    dist_min, dist_averages = get_stats_batch(gt_list, sample_list, None, self.n_test_batch_size, self.n_test_samples)
                    self.writer.add_scalars(f'ade_min', {
                        '0': np.mean(dist_min[:, 0])/2428,
                        '30': np.mean(dist_min[:, 29])/2428,
                        '60': np.mean(dist_min[:, 59])/2428,
                    }, self.step)

                    self.writer.add_scalars(f'ade_average', {
                        '0': np.mean(dist_averages[:, 0])/2428,
                        '30': np.mean(dist_averages[:, 29])/2428,
                        '60': np.mean(dist_averages[:, 59])/2428,
                    }, self.step)


            self.step += 1

            # wandb.log({'step': self.step,
            #             'train_loss': running_loss,
            #             'test_ade': np.mean(dist_averages)})

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load_model(self, loadpath):
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True, collate_fn=wrap_prisoner_a_star
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch[0])
        # conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        # normed_observations = trajectories[:, :, self.dataset.action_dim:]
        normed_observations = trajectories
        observations = self.dataset.unnormalize(normed_observations)

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            # conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            # conditions = apply_dict(
            #     einops.repeat,
            #     conditions,
            #     'b d -> (repeat b) d', repeat=n_samples,
            # )
            # global_cond = batch[1].repeat(n_samples, 1)
            # global_cond = {k: v.repeat(n_samples, 1) for k, v in batch[1].items()}
            repeated_detects = batch[1]['detections'].data.repeat(n_samples, 1)
            global_cond = {'detections': repeated_detects, 'hideouts': batch[1]['hideouts'].repeat(n_samples, 1)}

            conditions = batch[2] * n_samples

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(global_cond, conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples

            # [ 1 x 1 x observation_dim ]
            # normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # normed_observations = np.concatenate([
            #     np.repeat(normed_conditions, n_samples, axis=0),
            #     normed_observations
            # ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # print(normed_observations.shape)
            observations = self.dataset.unnormalize(normed_observations)

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
