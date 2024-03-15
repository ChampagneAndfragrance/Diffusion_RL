import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from .utils import clip_theta, distance, c_str
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import torch
import copy
import random
import skimage.measure
from fugitive_policies.custom_queue import QueueFIFO
# from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial, Plotter
from fugitive_policies.a_star_policy import AStarPolicy
from fugitive_policies.a_star.gridmap import OccupancyGridMap
from fugitive_policies.a_star.a_star import a_star
from fugitive_policies.a_star.utils import plot_path, plot_multiple_paths, plot_both_paths

global_device_name = "cuda"
global_device = torch.device("cuda")

def cycle(dl):
    while True:
        if len(dl.dataset) == 0:
            yield None
        else:
            for data in dl:
                yield data

# def smoothness_loss(traj):
#     smoothness_out = (traj[:, :-1, :] - traj[:, 1:, :])**2
#     smoothness_out = torch.sqrt(smoothness_out.sum(dim=-1))
#     return smoothness_out.mean()

# def constraint_loss(traj, conditions):
#     c_loss = 0
#     for b, conditions in enumerate(conditions):
#         if len(conditions[0]) > 0: # if there are conditions
#             # traj[b, conditions[0]] = torch.tensor(conditions[1], dtype=x.dtype).to(x.device)
#             d = (traj[b, conditions[0]] - torch.tensor(conditions[1], dtype=traj.dtype).to(traj.device)) ** 2
#             d = torch.sqrt(d.sum(dim=-1))
#             c_loss += d.mean()
#     return c_loss

# def mountain_loss(traj):
#     m_center = np.array([[1800, 1600]])/ 2428. 
#     m_center = m_center * 2 - 1
#     m_center = torch.tensor(m_center, dtype=traj.dtype).to(traj.device)

#     m_center = m_center.unsqueeze(0)
#     m_center = m_center.repeat((traj.shape[0], traj.shape[1], 1))
#     m_radius = (75 / 2428.) * 2

#     d = torch.sqrt(((traj - m_center) ** 2).sum(dim=-1))
#     loss = - d / m_radius + 1
#     loss = torch.clamp(loss, min=0)

#     return loss.mean()

# def apply_conditioning(x, conditions, action_dim):
#     for b, conditions in enumerate(conditions):
#         if len(conditions[0]) > 0: # if there are conditions
#             x[b, conditions[0]] = torch.tensor(conditions[1], dtype=x.dtype).to(x.device)
#     return x

# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = np.linspace(0, steps, steps)
#     alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
#     return torch.tensor(betas_clipped, dtype=dtype)

def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, list):
        x = np.array(x)
    return x

# class DiffusionModel(torch.nn.Module):
#     def __init__(self, env, diffusion_path, ema_path, n_timesteps, clip_denoised=True, predict_epsilon=False) -> None:
#         self.model = torch.load(diffusion_path)
#         self.ema_model = torch.load(ema_path)
#         betas = cosine_beta_schedule(n_timesteps)
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

#         self.n_timesteps = int(n_timesteps)
#         self.clip_denoised = clip_denoised
#         self.predict_epsilon = predict_epsilon

#         self.register_buffer('betas', betas)
#         self.register_buffer('alphas_cumprod', alphas_cumprod)
#         self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
#         self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
#         self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
#         self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#         self.register_buffer('posterior_variance', posterior_variance)

#         ## log calculation clipped because the posterior variance
#         ## is 0 at the beginning of the diffusion chain
#         self.register_buffer('posterior_log_variance_clipped',
#             torch.log(torch.clamp(posterior_variance, min=1e-20)))
#         self.register_buffer('posterior_mean_coef1',
#             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         self.register_buffer('posterior_mean_coef2',
#             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

#     def predict_start_from_noise(self, x_t, t, noise):
#         '''
#             if self.predict_epsilon, model output is (scaled) noise;
#             otherwise, model predicts x0 directly
#         '''
#         if self.predict_epsilon:
#             return (
#                 extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#                 extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
#             )
#         else:
#             return noise

#     def q_posterior(self, x_start, x_t, t):
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped

#     def p_mean_variance(self, x, cond, t, global_cond):
#         x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, global_cond=global_cond))

#         if self.clip_denoised:
#             x_recon.clamp_(-1., 1.)
#         else:
#             assert RuntimeError()

#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
#                 x_start=x_recon, x_t=x, t=t)
#         return model_mean, posterior_variance, posterior_log_variance


#     @torch.no_grad()
#     def p_sample(self, x, global_cond, cond, t):
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
#         noise = torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     @torch.no_grad()
#     def p_sample_repaint(self, x, global_cond, cond, t, noise=None, device='cuda'):
#         """ Use the repaint sampler to condition on the known timesteps """
#         # global cond are {detections and hideouts}
#         # cond are the detections that are within the current plan
#         cond_noised = copy.deepcopy(cond)

#         # add noise to the ground truth detections
#         for i in range(len(cond_noised)):
#             if isinstance(cond_noised[i][1], (np.ndarray, np.generic)):
#                 data = torch.from_numpy(cond_noised[i][1]).to(device)
#             else:
#                 data = cond_noised[i][1].to(device)
#             # t_full = torch.ones(data.size(0)).to() * t[i]
#             t_full = torch.full((data.size(0),), t[i], device=device, dtype=torch.long)
#             if noise is None:
#                 noise = torch.randn_like(data).to(device)
#             data_noised =  (
#                 extract(self.sqrt_alphas_cumprod, t_full, data.shape) * data +
#                 extract(self.sqrt_one_minus_alphas_cumprod, t_full, data.shape) * noise
#             )
#             cond_noised[i][1] = data_noised

#         # apply the new conditioning to the trajectory
#         # this essentially masks our x
#         x = apply_conditioning(x, cond_noised, self.action_dim) 

#         # Now we denoise it again
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
#         noise = torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     def p_sample_constrained(self, x, global_cond, cond, t, constraint_scale = 30):
#         b, *_, device = *x.shape, x.device
#         noise= torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)

#         model_mean = model_mean.clone().detach()
#         model_mean.requires_grad_()
#         optimizer = torch.optim.Adam([model_mean], lr=1e-2)
#         losses = []
#         for i in range(10):
#             optimizer.zero_grad()
#             l = 5*mountain_loss(model_mean) + 50 * smoothness_loss(model_mean)
#             # print(d)
#             losses.append(l.item())
#             l.backward()
#             optimizer.step()
#             # model_mean = apply_conditioning(model_mean, cond, self.action_dim)

#         # adjust_mean = self.compute_constraint_gradient(x, cond) * constraint_scale
#         # adjust_mean = adjust_mean * constraint_scale * (nonzero_mask * (0.5 * model_log_variance).exp())

#         # try alternating between sampling from the model and sampling from the constraint
#         # if i % 2 != 0:
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise    


#     @torch.no_grad()
#     def move_towards_constraint(self, x, global_cond, cond, t, constraint_scale=15):
#         b, *_, device = *x.shape, x.device
#         noise= torch.randn_like(x)
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) 
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=global_cond)
#         adjust_mean = self.compute_constraint_gradient(x, cond) 
#         # adjust_mean = adjust_mean * constraint_scale * (nonzero_mask * (0.5 * model_log_variance).exp())
#         adjust_mean = adjust_mean * constraint_scale
#         # return x + adjust_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
#         return x + adjust_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     @torch.no_grad()
#     def p_sample_loop_original(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
#         device = self.betas.device

#         batch_size = shape[0]
#         x = torch.randn(shape, device=device)
#         x = apply_conditioning(x, cond, self.action_dim)

#         if return_diffusion: diffusion = [x]

#         # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
#             x = self.p_sample(x, global_cond, cond, timesteps)
#             x = apply_conditioning(x, cond, self.action_dim)

#             # progress.update({'t': i})

#             if return_diffusion: diffusion.append(x)

#         # progress.close()

#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x
        
#     # @torch.no_grad()
#     def p_sample_loop_constrained(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
#         device = self.betas.device

#         batch_size = shape[0]
#         x = torch.randn(shape, device=device)
#         x = apply_conditioning(x, cond, self.action_dim)

#         if return_diffusion: diffusion = [x]

#         # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
#             x = self.p_sample_constrained(x, global_cond, cond, timesteps)
#             x = apply_conditioning(x, cond, self.action_dim)

#             # progress.update({'t': i})

#             if return_diffusion: diffusion.append(x)

#         # progress.close()

#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x

#     @torch.no_grad()
#     def p_sample_loop_repaint(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
#         for sample in self.p_sample_loop_progressive(shape, global_cond, cond, verbose, return_diffusion):
#             final = sample
#         return final

#     # @torch.no_grad()
#     def p_sample_loop(self, shape, global_cond, verbose=True, return_diffusion=False, **kwargs):
#         sample_type = kwargs.get('sample_type', 'original')
#         if sample_type == 'repaint':
#             return self.p_sample_loop_repaint(shape, global_cond, verbose, return_diffusion)
#         elif sample_type == 'constrained':
#             return self.p_sample_loop_constrained(shape, global_cond, verbose, return_diffusion)
#         elif sample_type == 'original':
#             return self.p_sample_loop_original(shape, global_cond, verbose, return_diffusion)
#         else:
#             raise NotImplementedError

#     @torch.no_grad()
#     def p_sample_loop_progressive(self, shape, global_cond, cond, verbose=True, return_diffusion=False):
#         device = self.betas.device

#         batch_size = shape[0]

#         times = get_schedule_jump()
#         time_pairs = list(zip(times[:-1], times[1:]))

#         image_after_step = torch.randn(shape, device=device)

#         for t_last, t_cur in time_pairs: 
#             if t_cur < t_last:
#                 # x = reverse_diffusion x_known
#                 timesteps = torch.full((batch_size,), t_last, device=device, dtype=torch.long)
#                 image_after_step = self.p_sample_repaint(image_after_step, global_cond, cond, timesteps)
#                 yield image_after_step
#             else:
#                 # x = forward diffusion x
#                 # renoise everything to get better yummy samples
#                 t_shift = 1
#                 # can't use q_sample because we are noising from the current step, not from the start
#                 image_before_step = image_after_step.clone()
#                 image_after_step = self.undo(image_after_step, t=t_last + t_shift, debug=False)

#     def undo(self, img_after_model, t, debug=False):
#         return self._undo(img_after_model, t)    

#     def _undo(self, x, t, device="cuda"):
#         # need to make this correct
#         # beta = _extract_into_tensor(self.betas, t, x.shape)
#         # x_noisy = torch.sqrt(1-beta) * x + torch.sqrt(beta) * torch.randn_like(x)
#         # return x_noisy
    
#         # self.register_buffer('sqrt_betas', torch.sqrt(betas))
#         sqrt_betas = torch.sqrt(self.betas)
#         sqrt_one_minus_betas = torch.sqrt(1 - self.betas)

#         t_full = torch.full((x.size(0),), t, device=device, dtype=torch.long)
#         noise = torch.randn_like(x).to(device)
#         x_noisy =  (
#             extract(sqrt_one_minus_betas, t_full, x.shape) * x +
#             extract(sqrt_betas, t_full, x.shape) * noise
#         )
#         return x_noisy


#     def compute_smoothness(self, traj):
#         """ Currently not used on REPAINT branch !!! """
#         smoothness_out = (traj[:, :-1, :] - traj[:, 1:, :])**2
#         smoothness_out = torch.sqrt(smoothness_out.sum(dim=-1))

#         return smoothness_out.mean()

#     def compute_constraint_gradient(self, traj, cond):
#         """ Compute the gradient to make the constraint satisfied.
#          Currently not used on REPAINT branch !!! 
#         One issue is how to ignore the gradient in areas of -1?
#         """
#         assert cond is not None
#         with torch.enable_grad():
#             traj = traj.detach().requires_grad_(True)
#             out = self.compute_smoothness(traj)
#             ret_val = torch.autograd.grad(out, traj)[0]
        
#         # for all the indices in cond, set the gradient to 0
#         for b, c in enumerate(cond):start_loc
#         return ret_val

#     # @torch.no_grad()
#     def conditional_sample(self, global_cond, cond, *args, horizon=None, **kwargs):
#         '''
#             conditions : [ (time, state), ... ]
#         '''
#         device = self.betas.device
#         batch_size = len(cond)
#         horizon = horizon or self.horizon
#         shape = (batch_size, horizon, self.transition_dim)
#         # global_cond = global_cond.to(device)
#         global_cond = {k: v.to(device) for k, v in global_cond.items()}

#         return self.p_sample_loop(shape, global_cond, cond, *args, **kwargs)

class DiffusionGlobalPlanner(object):
    def __init__(self, env, diffusion, ema, max_speed, plot, plot_traj_num=None) -> None:
        self.model = diffusion
        self.ema_model = ema
        self.env = env
        self.max_speed = max_speed
        self.actions = []
        self.max_x, self.max_y = (self.env.dim_x, self.env.dim_y)
        self.min_x, self.min_y = (0, 0)
        self.plot_flag = plot
        self.plot_traj_num = plot_traj_num

    def predict(self, observation, deterministic=True):
        return (self.get_desired_action(observation), None)

    def get_desired_action(self, observation):
        # self.observations.process_observation(observation)

        desired_action = self.generate_path_samples()
        # desired_action = self.action_to_random_hideout()
        # import time
        # time.sleep(1)
        return desired_action

    def generate_path_samples(self, n_samples=10, plot=True):
        '''
            generate samples from (ema) diffusion model
        '''
        if len(self.actions) == 0:
            start_loc = torch.cat((torch.Tensor([0]), self.normalize(torch.Tensor(self.env.get_prisoner_location())))).unsqueeze(0).repeat(n_samples, 1)
            # start_loc = torch.cat((torch.Tensor([0]), torch.Tensor(self.env.get_prisoner_location()) / self.env.dim_x)).unsqueeze(0).repeat(n_samples, 1)
            hideout_locs = torch.Tensor(np.concatenate(self.env.hideout_locations) / self.env.dim_x).unsqueeze(0).repeat(n_samples, 1)
            # repeated_detects = batch[1]['detections'].data.repeat(n_samples, 1)
            global_cond = {'detections': start_loc, 'hideouts': hideout_locs}
            # conditions = self.construct_conditions(start_loc[0:1,1:], hideout_locs[0:1]*self.env.dim_x, n_samples)
            conditions = [[[np.array([0]), start_loc[0:1,1:]]]] * n_samples
            # conditions = [[[[], []]]] * n_samples

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.model.conditional_sample(global_cond, conditions, sample_type="constrained")
            samples = self.unnormalize(to_np(samples))
            downsampled_paths = np.clip(samples[:, ::3, :], a_min=0, a_max=self.env.dim_x-1).astype(int)
            if plot:
                plot_multiple_paths(downsampled_paths)
            path_sample = downsampled_paths[np.random.randint(n_samples)]
            self.actions = self.convert_path_to_actions(path_sample)
        return self.actions.pop(0)

    def construct_conditions(self, normalized_start_loc, unnormalized_hideout_locs, n_samples):
        hideoutID_hideoutLoc = unnormalized_hideout_locs.view(-1, 2).clone()
        normalized_hideout_locs = self.normalize(hideoutID_hideoutLoc)
        # INFO: conditioned on random hideout
        conditions = [[[np.array([0]), normalized_start_loc], [np.array([239]), normalized_hideout_locs[np.random.randint(normalized_hideout_locs.shape[0])]]] for _ in range(n_samples)]
        # INFO: conditioned on all hideout
        # conditions = [[[np.array([0]), normalized_start_loc], [np.array([239]), normalized_hideout_locs]] for _ in range(n_samples)]
        return conditions

    def convert_path_to_actions(self, path):
        """ Converts list of points on path to list of actions (speed, thetas)
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        actions = []
        currentpos = path[0]
        for nextpos in path[1:]:
            a = self.get_actions_between_two_points(currentpos, nextpos)
            currentpos = nextpos
            actions.extend(a)
        return actions

    def convert_traj_to_action(self, curr_loc, next_loc):
        dist = (np.linalg.norm(np.asarray(curr_loc) - np.asarray(next_loc)))
        speed = min(dist, self.max_speed)
        theta = np.arctan2(next_loc[1] - curr_loc[1], next_loc[0] - curr_loc[0])
        action = np.array([speed, theta], dtype=np.float32)
        return action

    def get_actions_between_two_points(self, startpos, endpos):
        """ Returns list of actions (speed, thetas) to traverse between two points.
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        currentpos = startpos
        actions = []
        if np.array_equal(currentpos, endpos) == True:
            action = np.array([0, 0], dtype=np.float32)
            actions.append(action)
        while np.array_equal(currentpos, endpos) == False:
            dist = (np.linalg.norm(np.asarray(currentpos) - np.asarray(endpos)))
            speed = min(dist, self.max_speed)
            try:
                # currentpos = np.clip(currentpos, -1e10, 2428)
                # endpos = np.clip(endpos, -1e10, 2428)
                # y_diff = endpos[1] - currentpos[1]
                # x_diff = endpos[0] - currentpos[0]
                # x_diff_clipped = np.clip(x_diff, -1e10, 1e10)
                # y_diff_clipped = np.clip(y_diff, -1e10, 1e10)
                theta = np.arctan2(endpos[1] - currentpos[1], endpos[0] - currentpos[0])
            except RuntimeWarning as rw:
                theta = 0.0
            except Exception as e:
                theta = 0.0
            action = np.array([speed, theta], dtype=np.float32)
            actions.append(action)
            currentpos = self.simulate_action(currentpos, action)

            # if self.terrain.world_representation[0, currentpos[0], currentpos[1]] == False:
                # print("In mountain!!")

        return actions

    def simulate_action(self, start_location, action):
        direction = np.array([np.cos(action[1]), np.sin(action[1])])
        speed = action[0]
        new_location = np.round(start_location + direction * speed)
        new_location[0] = np.clip(new_location[0], 0, self.env.dim_x - 1)
        new_location[1] = np.clip(new_location[1], 0, self.env.dim_y - 1)
        new_location = new_location.astype(np.int)
        return new_location

    def unnormalize(self, sample):
            x = sample[..., 0]
            sample[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

            y = sample[..., 1]
            sample[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
            return sample

    def normalize(self, arr):
            x = arr[..., 0]
            arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

            y = arr[..., 1]
            arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
            return arr

class DiffusionStateOnlyGlobalPlanner(object):
    def __init__(self, env, diffusion_path, plot, traj_grader_path, costmap=None, res=None, sel=False) -> None:
        self.model = torch.load(diffusion_path)
        self.env = env
        self.actions = []
        self.max_x, self.max_y = (self.env.dim_x, self.env.dim_y)
        self.min_x, self.min_y = (0, 0)
        self.plot_flag = plot
        self.costmap = costmap
        self.res = res
        self.sel = sel
        if traj_grader_path is not None:
            self.traj_grader = torch.load(traj_grader_path)
        else:
            self.traj_grader = None

    def get_scaled_path(self, seed=None, hideout_division=None):
        valid_path = False

        # INFO: process the current red observation
        # self.observations.process_observation(observation)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        if hideout_division is None:
            if not self.sel:
                # INFO: randomly select one global path to track
                target_hideout_id = np.random.randint(low=0, high=3)
                hideout_division = np.zeros((len(self.env.hideout_list))).astype(int)
                hideout_division[target_hideout_id] = 1
            else:
                # INFO: generate lots of global path at the same time
                hideout_division = [10, 10, 10]

        global_cond, local_cond = self.env.construct_diffusion_conditions(cond_on_hideout_num=hideout_division)
        # global_cond: {'hideouts': tensor([[-0.8072, -0... 0.8731]]), 'red_start': tensor([[0.0000, 0.8... 0.9811]])}, dim: [batch_size, feat_dim]
        while valid_path is False:
            start = time.time()
            sample = self.model.conditional_sample(global_cond=global_cond, cond=local_cond, sample_type="constrained", estimator=self.traj_grader)
            end = time.time()
            one_rrt_time = end - start
            # print("inner loop time: ", one_rrt_time)
            sample = self.unnormalize(sample)
            dense_path = self.interpolate_paths(sample, total_dense_path_num=100)

            # INFO: select best trajectory
            if self.costmap is not None:
                best_traj_id = self.select_traj(dense_path, self.costmap, self.res)
            else:
                best_traj_id = 0

            # INFO: if candidate sample collide with mountain?
            if all(np.linalg.norm(dense_path[best_traj_id] - np.array([[1600, 1800]]), axis=1) > 140):
                valid_path = True

        self.env.waypoints = sample[best_traj_id]

        if self.plot_flag:
            # INFO: Define custom colorbar
            boundaries = [0, 0.2, 0.5, 1.0]  # Adjust the boundaries as needed
            colors = ['black', 'brown', 'orange', 'oldlace']  # Adjust the colors as needed
            cmap2 = mcolors.LinearSegmentedColormap.from_list('cmap2', list(zip(boundaries, colors)))

            hideouts = self.unnormalize(global_cond["hideouts"].reshape(-1, 3, 2)).detach().cpu().numpy()
            starts = self.unnormalize(global_cond["red_start"]).detach().cpu().numpy()
            figure, axes = plt.subplots()
            for sample_idx in range(sum(hideout_division)):
                if all(np.linalg.norm(dense_path[sample_idx] - np.array([[1600, 1800]]), axis=1) > 140):
                    axes.plot(sample[sample_idx,:,0], sample[sample_idx,:,1], c='cyan', alpha=0.5)
                axes.scatter(hideouts[sample_idx,:,0], hideouts[sample_idx,:,1], c='violet', s=80)
                axes.scatter(starts[sample_idx,0], starts[sample_idx,1], c='g', s=80)
                # plt.scatter(1600, 1800, s=150)
            axes.plot(sample[best_traj_id,:,0], sample[best_traj_id,:,1], 'y', linewidth=3)
            axes.imshow(self.costmap, extent=(0, 2428, 0, 2428), cmap=cmap2, interpolation='nearest')
            plt.axis('off')
            
            # circle = plt.Circle(( 1600 , 1800 ), 150 )
            # axes.add_artist( circle )
            plt.axis('square')
            plt.xlim(0, 2428)
            plt.ylim(0, 2428)
            plt.savefig('candidates_nozone_%d.png' % seed, dpi=100, bbox_inches='tight')
            # plt.show()

            # plt.figure()
            # plt.imshow(self.costmap, cmap='magma', interpolation='nearest')
            # plt.colorbar()
            # plt.show()
        return sample.tolist()

    def interpolate_paths(self, paths, total_dense_path_num):
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

    def select_traj(self, dense_path, costmap, res):
        grading_path = (dense_path // res).astype(int)
        grading_path_len = np.sum(np.linalg.norm(grading_path[:,1:,:]-grading_path[:,:-1,:], axis=-1), axis=-1)
        costs = costmap[costmap.shape[0]-1-grading_path[:, :, 1], grading_path[:, :, 0]]
        # INFO: Calculate the average cost along the trajectory axis
        sum_costs = np.sum(costs, axis=1) * grading_path_len
        # INFO: Get the trajectory index with the lowest cost
        best_traj_idx = np.argmin(sum_costs)
        return best_traj_idx



    def unnormalize(self, sample):

        sample = copy.deepcopy(sample)

        x = sample[..., 0]
        sample[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = sample[..., 1]
        sample[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return sample

    def normalize(self, arr):

        arr = copy.deepcopy(arr)

        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

class DiffusionGlobalPlannerHideout(DiffusionGlobalPlanner):
    def __init__(self, env, diffusion, ema, estimator, max_speed, plot, plot_traj_num) -> None:
        super().__init__(env, diffusion, ema, max_speed, plot, plot_traj_num)
        self.scale = (20, 20)
        pooled = self.convert_map_for_astar()
        self.gmap = OccupancyGridMap.from_terrain(pooled, 1)
        self.traj_id = 0
        self.traj_grader = estimator
        self.model.n_timesteps = 1

    def update_diffusion_grader(self, new_diffusion, new_ema, new_estimator):
        self.model = new_diffusion
        self.ema_model = new_ema
        self.traj_grader = new_estimator

    def reset(self, incremental_dataset):
        self.actions = []
        self.traj_id = 0
        # self.predict(self.env.get_fugitive_observation(), incremental_dataset)

    def get_normalized_raw_red_downsampled_traj(self):
        return self.raw_red_traj

    def get_raw_red_downsampled_traj(self):
        return (self.raw_red_traj + 1) / 2

    def update_raw_red_downsampled_traj(self, new_red_downsampled_traj):
        red_loc = ((torch.Tensor(self.env.prisoner.location).to(global_device_name) / self.env.dim_x) * 2 - 1).unsqueeze(0)
        self.raw_red_traj = new_red_downsampled_traj * 2 - 1 
        min_dist_pt_idx = torch.argmin(torch.norm(self.raw_red_traj-red_loc, dim=-1))
        self.raw_red_traj = torch.cat((red_loc, self.raw_red_traj[min_dist_pt_idx:, :]), dim=0)
        self.gmap.plot()
        plot_path(self.unnormalize(to_np(self.raw_red_traj)), self.env.hideout_locations, point_period=1)
        return

    def update_red_actions(self):
        digit_traj = self.unnormalize(to_np(self.raw_red_traj))
        digit_traj = np.clip(digit_traj, a_min=0, a_max=self.env.dim_x-1).astype(int)
        self.actions = self.convert_path_to_actions(digit_traj)
        return 

    def predict(self, observation, dataset, deterministic=True):
        return self.get_desired_action(observation, dataset)

    def get_desired_action(self, observation, dataset):
        # self.observations.process_observation(observation)

        desired_action, hideout = self.generate_path_samples(dataset, plot=self.plot_flag)
        # desired_action = self.action_to_random_hideout()
        # import time
        # time.sleep(1)
        return desired_action, hideout

    def generate_path_only(self, dataset, n_samples=10, plot=True):
        repeat_collate_fn = dataset.collate_fn_repeat()
        # def multiHideout_collate_fn(batch):
        #     return repeat_collate_fn(batch, n_samples)
        multiHideout_collate_fn = lambda batch: repeat_collate_fn(batch, n_samples)
        self.dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=multiHideout_collate_fn))
        batch = self.dataloader.__next__()
        _, global_cond, conditions = batch
        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = self.model.conditional_sample(global_cond, conditions, sample_type="constrained", horizon=240, estimator=self.traj_grader)
        unnormalized_samples = self.unnormalize(to_np(samples))
        downsampled_paths = np.clip(unnormalized_samples[:, ::1, :], a_min=0, a_max=self.env.dim_x-1).astype(int)
        # if plot:
        #     self.gmap.plot()
        #     plot_multiple_paths(downsampled_paths, self.env.hideout_locations)
        sel_traj_idx = np.random.randint(n_samples)
        self.path_sample = downsampled_paths[sel_traj_idx]
        self.hideout = global_cond["hideouts"][sel_traj_idx]
        self.raw_red_traj = samples[sel_traj_idx,::4,:2]
        print("Reward Est. = ", self.traj_grader(samples[:,::4,:2])[sel_traj_idx])
        if plot:
            # plot_path(path_sample[...,:2])
            self.gmap.plot()
            # INFO: Plot red+blue paths
            # plot_both_paths(np.expand_dims(self.path_sample, axis=0), self.env.hideout_locations)
            # INFO: Plot red path only
            plot_path(self.path_sample, self.env.hideout_locations)

        return self.path_sample[::4,...,0:2], self.path_sample[:,...,2:4], self.path_sample[:,...,4:6], self.hideout

    def generate_path_samples(self, dataset, n_samples=10, refresh_period=239, plot=True):
        '''
            generate samples from (ema) diffusion model
        '''
        # self.model.n_timesteps = 30
        # self.ema_model.n_timesteps = 30
        if self.traj_id % refresh_period == 0 or len(self.actions) == 0:
            self.traj_id = 0
            repeat_collate_fn = dataset.collate_fn_repeat()
            # def multiHideout_collate_fn(batch):
            #     return repeat_collate_fn(batch, n_samples)
            multiHideout_collate_fn = lambda batch: repeat_collate_fn(batch, n_samples)
            self.dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=multiHideout_collate_fn))
            batch = self.dataloader.__next__()
            _, global_cond, conditions = batch
            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.model.conditional_sample(global_cond, conditions, sample_type="constrained", horizon=240, estimator=self.traj_grader)
            unnormalized_samples = self.unnormalize(to_np(samples))
            downsampled_paths = np.clip(unnormalized_samples[:, ::1, :], a_min=0, a_max=self.env.dim_x-1).astype(int)
            # if plot:
            #     self.gmap.plot()
            #     plot_multiple_paths(downsampled_paths, self.env.hideout_locations)
            sel_traj_idx = np.random.randint(n_samples)
            self.path_sample = downsampled_paths[sel_traj_idx]
            self.hideout = global_cond["hideouts"][sel_traj_idx]
            uniform_path = generate_uniform_path(self.path_sample, self.hideout, total_points=60, normalized_path=False, normalized_hideout=True)
            self.raw_red_traj = samples[sel_traj_idx,::4,:2]
            print("Reward Est. = ", self.traj_grader(samples[:,::1,:2])[sel_traj_idx])
            if plot:
                # plot_path(path_sample[...,:2])
                self.gmap.plot()
                # INFO: Plot red+blue paths
                # plot_both_paths(np.expand_dims(self.path_sample, axis=0), self.env.hideout_locations)
                # INFO: Plot red path only
                # plot_path(self.path_sample, self.env.hideout_locations)
                # INFO: Plot UNIFORM red path only
                plot_path(uniform_path, self.env.hideout_locations, point_period=1)

            # INFO: This is for path->actions
            # self.actions = self.convert_path_to_actions(self.path_sample[:refresh_period:4,...,0:2])
            # INFO: This is for UNIFORM path->actions
            self.actions = self.convert_path_to_actions(uniform_path)
        # INFO: This is for traj->actions
        # action = self.convert_traj_to_action(self.env.prisoner.location, self.path_sample[...,self.traj_id+1,:2])
        self.traj_id = self.traj_id + 1
        # INFO: This is for path->actions
        return self.actions.pop(0), self.hideout
        # INFO: This is for traj->actions
        # return action, self.hideout

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)

        last_dim = obs.shape[-1]
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = obs[..., evens]
        obs[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., odds]
        obs[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return obs

    def convert_map_for_astar(self):
        """ Reduce the size of the map for the Astar algorithm """
        mountains = copy.deepcopy(self.env.terrain.world_representation[0, :, :])
        terrain_map = copy.deepcopy(self.env.terrain.world_representation[1, :, :])

        mask = np.where(mountains == 1)
        terrain_map[mask] = -1

        # x_remainder = terrain_map.shape[0] % self.scale[0]
        # y_remainder = terrain_map.shape[1] % self.scale[1]
        # pooled = skimage.measure.block_reduce(terrain_map, (self.scale[0], self.scale[1]), np.min)
        # pooled = pooled[:-x_remainder, :-y_remainder]
        # pooled = np.flipud(np.rot90(pooled, k=1))

        # self.x_scale = self.env.terrain.dim_x / pooled.shape[0]
        # self.y_scale = self.env.terrain.dim_y / pooled.shape[1]

        terrain_map = np.flipud(np.rot90(terrain_map, k=1))
        return terrain_map

    
class DiffusionGlobalPlannerSelHideouts(DiffusionGlobalPlannerHideout):
    def __init__(self, env, diffusion, ema, estimator, max_speed, plot, plot_traj_num) -> None:
        super().__init__(env, diffusion, ema, estimator, max_speed, plot, plot_traj_num)

    def generate_path_samples(self, dataset, n_samples=10, refresh_period=239, plot=True):
        '''
            generate samples from (ema) diffusion model
        '''
        # self.model.n_timesteps = 30
        # self.ema_model.n_timesteps = 30
        if self.traj_id % refresh_period == 0 or len(self.actions) == 0:
            self.traj_id = 0
            repeat_collate_fn = dataset.sel_collate_fn()
            # def multiHideout_collate_fn(batch):
            #     return repeat_collate_fn(batch, n_samples)
            selHideout_collate_fn = lambda batch: repeat_collate_fn(batch, n_samples)
            self.dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=selHideout_collate_fn))
            batch = self.dataloader.__next__()
            _, global_cond, conditions = batch
            print("hideout = ", global_cond["hideouts"][0], global_cond["hideouts"][11], global_cond["hideouts"][21])
            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.model.conditional_sample(global_cond, conditions, sample_type="constrained", horizon=240, estimator=self.traj_grader)
            unnormalized_samples = self.unnormalize(to_np(samples))
            downsampled_paths = np.clip(unnormalized_samples[:, ::1, :], a_min=0, a_max=self.env.dim_x-1).astype(int)
            # if plot:
            #     self.gmap.plot()
            #     plot_multiple_paths(downsampled_paths, self.env.hideout_locations)
            traj_grades = self.traj_grader(samples[:,::4,0:2])
            traj_grades_mean = [traj_grades[i*n_samples:(i+1)*n_samples].mean() for i in range(traj_grades.shape[0]//n_samples)]
            # sorted_indices = sorted(range(len(traj_grades)), key=lambda idx: -traj_grades_mean[idx].item())
            _, sorted_indices = torch.sort(traj_grades.view(-1), descending=True)
            sel_traj_idx = sorted_indices[0] # sorted_indices[0], np.random.randint(n_samples)
            self.path_sample = downsampled_paths[sel_traj_idx]
            self.hideout = global_cond["hideouts"][sel_traj_idx]
            self.raw_red_traj = samples[sel_traj_idx,::4,0:2]
            print("Reward Est. = ", traj_grades_mean)
            if plot:
                # plot_path(path_sample[...,:2])
                
                self.gmap.plot()
                # INFO: Plot red+blue paths
                plot_both_paths(np.expand_dims(self.path_sample, axis=0), self.env.hideout_locations, self.plot_traj_num)
                # INFO: Plot red path only
                # plot_path(self.path_sample, self.env.hideout_locations)

            # INFO: This is for path->actions
            self.actions = self.convert_path_to_actions(self.path_sample[:refresh_period:4,...,0:2])
            print("The length of action is: ", len(self.actions))


        # INFO: This is for traj->actions
        # action = self.convert_traj_to_action(self.env.prisoner.location, self.path_sample[...,self.traj_id+1,:2])
        self.traj_id = self.traj_id + 1
        # INFO: This is for path->actions
        return self.actions.pop(0), self.hideout
        # INFO: This is for traj->actions
        # return action, self.hideout

    def generate_path_only(self, dataset, n_samples=10, plot=True):
        repeat_collate_fn = dataset.sel_collate_fn()
        # def multiHideout_collate_fn(batch):
        #     return repeat_collate_fn(batch, n_samples)
        selHideout_collate_fn = lambda batch: repeat_collate_fn(batch, n_samples)
        self.dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=selHideout_collate_fn))
        batch = self.dataloader.__next__()
        _, global_cond, conditions = batch
        print("hideout = ", global_cond["hideouts"][0], global_cond["hideouts"][11], global_cond["hideouts"][21])
        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = self.model.conditional_sample(global_cond, conditions, sample_type="constrained", horizon=240, estimator=self.traj_grader)
        unnormalized_samples = self.unnormalize(to_np(samples))
        downsampled_paths = np.clip(unnormalized_samples[:, ::1, :], a_min=0, a_max=self.env.dim_x-1).astype(int)
        # if plot:
        #     self.gmap.plot()
        #     plot_multiple_paths(downsampled_paths, self.env.hideout_locations)
        traj_grades = self.traj_grader(samples[:,::4,0:2])
        traj_grades_mean = [traj_grades[i*n_samples:(i+1)*n_samples].mean() for i in range(traj_grades.shape[0]//n_samples)]
        # sorted_indices = sorted(range(len(traj_grades)), key=lambda idx: -traj_grades_mean[idx].item())
        _, sorted_indices = torch.sort(traj_grades.view(-1), descending=True)
        sel_traj_idx = sorted_indices[0] # sorted_indices[0], np.random.randint(n_samples)
        self.path_sample = downsampled_paths[sel_traj_idx]
        self.hideout = global_cond["hideouts"][sel_traj_idx]
        self.raw_red_traj = samples[sel_traj_idx,::4,0:2]
        print("Reward Est. = ", traj_grades_mean)
        if plot:
            # plot_path(path_sample[...,:2])
            
            self.gmap.plot()
            # INFO: Plot red+blue paths
            plot_both_paths(np.expand_dims(self.path_sample, axis=0), self.env.hideout_locations, self.plot_traj_num)
            # INFO: Plot red path only
            # plot_path(self.path_sample, self.env.hideout_locations)

        return self.path_sample[::4,...,0:2], self.path_sample[:,...,2:4], self.path_sample[:,...,4:6], self.hideout

def generate_uniform_path(path_sample, hideout, total_points, normalized_path=True, normalized_hideout=True):
    if normalized_path:
        path_sample = to_np(path_sample) * 2428
    else:
        path_sample = to_np(path_sample)

    if normalized_hideout:
        hideout = to_np(hideout) * 2428
    else:
        hideout = to_np(hideout)

    prisoner_traj = path_sample[:,...,0:2]
    prisoner_to_hideout_dist = np.linalg.norm(hideout - prisoner_traj, axis=-1)
    idx_of_traj_pt_closest_to_hideout = np.argmin(prisoner_to_hideout_dist)

    prisoner_traj_cut = np.concatenate((prisoner_traj[:idx_of_traj_pt_closest_to_hideout+1,:], np.expand_dims(hideout, axis=0)), axis=0)

    line = LineString(prisoner_traj_cut)
    distances = np.linspace(0, line.length, total_points)
    points = [line.interpolate(distance) for distance in distances]
    path = np.floor(np.array([[points[i].x, points[i].y] for i in (range(len(points)))]))
    # path = unary_union(points)
    return path
    # # Calculate the cumulative distances between consecutive points
    # distances = np.linalg.norm(np.diff(prisoner_traj_cut, axis=0), axis=1)
    # total_distance = np.sum(distances)

    # # Calculate the desired distance between dividing points
    # desired_distance = total_distance / (total_points - 1)

    # # Initialize variables
    # dividing_points = [prisoner_traj_cut[0]]
    # current_distance = 0

    # # Iterate through path segments and insert points based on desired distance
    # for i in range(len(prisoner_traj_cut) - 1):
    #     segment = prisoner_traj_cut[i:i + 2]
    #     segment_distance = distances[i]

    #     if current_distance < desired_distance:
    #         current_distance = current_distance + segment_distance
    #     while current_distance + desired_distance < segment_distance:
    #         t = (current_distance + desired_distance) / segment_distance
    #         new_point = (1 - t) * segment[0] + t * segment[1]
    #         dividing_points.append(new_point)
    #         current_distance += desired_distance

    # # Add the last point from the path
    # dividing_points.append(prisoner_traj_cut[-1])

    # # Convert the result to a NumPy array
    # dividing_points = np.array(dividing_points)

    # # Print the dividing points
    # print(dividing_points)