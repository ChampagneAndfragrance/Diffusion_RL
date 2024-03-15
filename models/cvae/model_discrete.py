import torch.nn as nn
# import torch.nn.functional as F
import torch
# import torch.optim as optim
import sys, os
sys.path.append(os.getcwd())
# import numpy as np
import torch
import torch.distributions as td
# from models.cvae.gmm2D import GMM2D
from models.encoders import EncoderRNN
# import random
# import shutil
# from models.encoders import EncoderRNN
# from models.decoder import MixtureDensityDecoder

class CVAE(nn.Module):

    def __init__(self, x_dim, encoder_hidden_dim, future_hidden_dim, z_dim):
        super(CVAE, self).__init__()
        self.z_logit_clip = None
        self.x_encoder = EncoderRNN(x_dim, encoder_hidden_dim)
        self.y_encoder = EncoderRNN(2, future_hidden_dim)
        self.x_to_z_layer = nn.Linear(encoder_hidden_dim, z_dim)
        self.xy_to_z_layer = nn.Linear(encoder_hidden_dim + future_hidden_dim, z_dim)

        self.initial_h_layer = nn.Linear(z_dim + x_dim, 32)
        rnn_cell = nn.GRUCell(32, 32) # input, hidden dim

        self.z_dim = z_dim
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, o_b, hidden_state=None):
        p_theta = self.o_b_encoder(o_b)

    def q_z_xy(self, o_b_encoded, s_r_encoded):
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([o_b_encoded, s_r_encoded], dim=1)
        h = xy
        z = self.xy_to_z_layer(h)
        return self.create_distribution(z)

    def p_z_x(self, o_b_encoded):
        h = o_b_encoded
        z = self.x_to_z_layer(h)
        return self.create_distribution(z)

    def create_distribution(self, logits):
        # logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits - torch.mean(logits, dim=-1, keepdim=True)
        # if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
        if self.z_logit_clip:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero

        return td.OneHotCategorical(logits=logits)

    def p_y_xz(self, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False, predict=False):
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            # if mode == ModeKeys.PREDICT and gmm_mode:
            #     a_t = gmm.mode()
            # else:
            #     a_t = gmm.rsample()

            if num_components > 1:
                if predict:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic'][self.node_type]['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist

    def sample_p(self, num_samples, p_dist):
        bs = p_dist.probs.size()[0]
        # num_components = self.N * self.K
        z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
        return torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))

    def compute_loss(self, o_b, s_r):
        """ o_b: the blue observation
            s_r: the red observation
        """ 
        o_b_encoded = self.o_b_encoder(o_b) # B x H
        s_r_encoded = self.s_r_encoder(s_r) # B x H
        
        q_dist = self.q_z_xy(o_b_encoded, s_r_encoded)
        p_dist = self.p_z_x(o_b_encoded)

        z = self.sample_latent(q_dist)

        p_y_xz = self.p_y_xz(o_b_encoded, z) # gmm values

        recon_loss = self.gmm_decoder.compute_loss(p_y_xz, s_r)
        kl_loss = self.kl_loss(q_dist, p_dist)
        # recon_loss = self.recon_loss(gmm_output, o_b)
        # recon_loss = self.decoder.compute_loss(gmm_output, s_r)

        loss = kl_loss + recon_loss
        return loss

