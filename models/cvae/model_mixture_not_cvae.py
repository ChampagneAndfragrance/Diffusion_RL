import torch.nn as nn
# import torch.nn.functional as F
import torch
# import torch.optim as optim
import sys, os
sys.path.append(os.getcwd())
# import numpy as np
import torch
import torch.distributions as td
# import random
# import shutil
# from models.encoders import EncoderRNN
# from models.decoder import MixtureDensityDecoder
from models.multi_head_mog import MixureDecoderMultiHead
from models.encoders import EncoderRNN

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


def kl_gaussian_mixtures_vectorized(mix_1, mix_2):
    pi_a, mu_a, sigma_a = mix_1 # B x n_gaussians x 2
    pi_b, mu_b, sigma_b = mix_2

    pi_a = pi_a.view(pi_a.shape[0], -1)
    mu_a = mu_a.view(mu_a.shape[0], -1, mu_a.shape[3])
    sigma_a = sigma_a.view(sigma_a.shape[0], -1, sigma_a.shape[3])

    pi_b = pi_b.view(pi_b.shape[0], -1)
    mu_b = mu_b.view(mu_b.shape[0], -1, mu_b.shape[3])
    sigma_b = sigma_b.view(sigma_b.shape[0], -1, sigma_b.shape[3])

    n_gaussians = mu_b.shape[1]

    # outer
    mu_a_tog = torch.repeat_interleave(mu_a, n_gaussians, dim=1) # [a, b, c] --> [a, a, b, b, c, c]
    sigma_a_tog = torch.repeat_interleave(sigma_a, n_gaussians, dim=1)

    pi_a_p = pi_a.repeat(1, n_gaussians) # B x n_gaussians**2 [a, b, c] --> [a, b, c, a, b, c]
    mu_a_p = mu_a.repeat(1, n_gaussians, 1) # B x n_gaussians**2 x 2
    sigma_a_p = sigma_a.repeat(1, n_gaussians, 1) # B x n_gaussians**2 x 2

    pi_b_p = pi_b.repeat(1, n_gaussians) # B x n_gaussians**2
    mu_b_p = mu_b.repeat(1, n_gaussians, 1) # B x n_gaussians**2 x 2
    sigma_b_p = sigma_b.repeat(1, n_gaussians, 1) # B x n_gaussians**2 x 2

    p_num = Normal(mu_a_tog, sigma_a_tog)
    q_num = Normal(mu_a_p, sigma_a_p)
    kl_num = kl_divergence(p_num, q_num)

    kls_num = torch.einsum('bxd,bx->bxd', torch.exp(-kl_num), pi_a_p) # B x n_gaussians**2
    kls_num_reshaped = kls_num.reshape(kl_num.shape[0], n_gaussians, n_gaussians, 2) # B x n_gaussians x n_gaussians x 2
    num = kls_num_reshaped.sum(dim=2) # B x n_gaussians x 2

    p_den = Normal(mu_a_tog, sigma_a_tog)
    q_den = Normal(mu_b_p, sigma_b_p)
    kl_den = kl_divergence(p_den, q_den)
    kls_den = torch.einsum('bxd,bx->bxd', torch.exp(-kl_den), pi_b_p) # B x n_gaussians**2
    kls_den_reshaped = kls_den.reshape(kl_den.shape[0], n_gaussians, n_gaussians, 2) # B x n_gaussians x n_gaussians x 2
    den = kls_den_reshaped.sum(dim=2) # B x n_gaussians x 2

    divided = num / den # B x num_gaussians x 2
    # print(divided.shape)
    res = torch.einsum('bgd,bg->bgd', torch.log(divided), pi_a)
    return res.sum(dim=1)

# def kl_gaussian_mixtures_vectorized(mix_1, mix_2):

#     pi_a, mu_a, sigma_a = mix_1 # B x n_gaussians x 2
#     pi_b, mu_b, sigma_b = mix_2

#     pi_a = pi_a.view(pi_a.shape[0], -1)
#     mu_a = mu_a.view(mu_a.shape[0], -1, mu_a.shape[3])
#     sigma_a = sigma_a.view(sigma_a.shape[0], -1, sigma_a.shape[3])

#     pi_b = pi_b.view(pi_b.shape[0], -1)
#     mu_b = mu_b.view(mu_b.shape[0], -1, mu_b.shape[3])
#     sigma_b = sigma_b.view(sigma_b.shape[0], -1, sigma_b.shape[3])

#     n_gaussians = mu_b.shape[1]
#     kl_all = []
#     for a in range(n_gaussians):

#         kl_num = []
#         for a_p in range(n_gaussians):
#             p_dist = Normal(mu_a[:, a, :], sigma_a[:, a, :])
#             q_dist = Normal(mu_a[:, a_p, :], sigma_a[:, a_p, :])
#             kl_n = kl_divergence(p_dist, q_dist)
#             pi = pi_a[:, a_p]
#             kl_num.append(torch.einsum('b, bd -> bd', pi, torch.exp(-kl_n)))

#         kl_num = torch.stack(kl_num) # num_gaussians x 2
#         kl_num = kl_num.sum(dim=0)
#         # print(kl_num)
#         kl_den = []
#         for b_p in range(n_gaussians):
#             p_dist = Normal(mu_a[:, a, :], sigma_a[:, a, :]) # b x 2
#             q_dist = Normal(mu_b[:, b_p, :], sigma_b[:, b_p, :])
#             kl_d = kl_divergence(p_dist, q_dist)
#             pi = pi_b[:, b_p]
#             kl_den.append(torch.einsum('b, bd -> bd', pi, torch.exp(-kl_d)))

#         kl_den = torch.stack(kl_den) # num_gaussians x 2
#         kl_den = kl_den.sum(dim=0)
#         kl_all.append(kl_num / kl_den)
#         # print(kl_num/kl_den)

#     # print(kl_all)
#     kl_all = torch.log(torch.stack(kl_all)) # num_gaussians x b x dimension
#     # print(kl_all, pi_a)
#     res = torch.einsum('gbd,bg->gbd', kl_all, pi_a)
#     # print(res)
#     return torch.sum(res, dim=0)

class CVAEContinuous(nn.Module):

    def __init__(self, x_dim, encoder_hidden_dim, future_hidden_dim):
        super(CVAEContinuous, self).__init__()
        self.z_logit_clip = None
        self.o_b_encoder = EncoderRNN(x_dim, encoder_hidden_dim)
        self.s_r_encoder = EncoderRNN(2, future_hidden_dim)
        # self.x_to_z_mu = nn.Linear(encoder_hidden_dim, z_dim)
        # self.x_to_z_log_var = nn.Linear(encoder_hidden_dim, z_dim)

        # self.xy_to_z_mu = nn.Linear(encoder_hidden_dim + future_hidden_dim, z_dim)
        # self.xy_to_z_log_var = nn.Linear(encoder_hidden_dim + future_hidden_dim, z_dim)

        self.x_to_gmm = MixureDecoderMultiHead(encoder_hidden_dim, output_dim=2, num_gaussians=5, num_heads=1)
        self.xy_to_gmm = MixureDecoderMultiHead(encoder_hidden_dim + future_hidden_dim, num_gaussians=5, num_heads=1)

        # if gmm_decoder:
        #     self.gmm = MixureDecoderMultiHead(encoder_hidden_dim + z_dim, output_dim=2, num_gaussians=2, num_heads=13)
        # else:
        #     self.initial_h_layer = nn.Linear(z_dim + encoder_hidden_dim, 32)
        #     self.rnn_cell = nn.GRUCell(encoder_hidden_dim + z_dim + 2, 32) # input, hidden dim
        #     self.rnn_to_state = nn.Linear(32, 2)

        # self.gmm_decoder_bool = gmm_decoder        
        
        self.relu = nn.ReLU()

        self.writer = None
    @property
    def device(self):
        return next(self.parameters()).device

    def predict(self, o_b, num_samples):
        o_b = o_b.to(self.device).float()
        o_b_encoded = self.o_b_encoder(o_b) # B x H
        p_dist = self.p_z_x(o_b_encoded)
        print("mean", p_dist.mean)
        print("var", p_dist.stddev)
        total_outputs = []
        for sample in range(num_samples):
            z = p_dist.rsample()
            
            # z = torch.ones((o_b.shape[0], self.z_dim), device=self.device)

            if self.gmm_decoder_bool:
                gmm_output = self.gmm_decoder(o_b_encoded, z)
                total_outputs.append(gmm_output)
                gmm_output = self.gmm_decoder(o_b_encoded, z)
                total_outputs.append(gmm_output)
            else:
                s_predicted = self.mse_decoder(o_b_encoded, z, horizon=13)
                total_outputs.append(s_predicted)
        if self.gmm_decoder_bool:
            return total_outputs
        else:
            return torch.stack(total_outputs)

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
        mu = self.xy_to_z_mu(h)
        log_var = self.xy_to_z_log_var(h)

        return self.create_distribution(mu, log_var)

    def p_z_x(self, o_b_encoded):
        h = o_b_encoded
        mu = self.x_to_z_mu(h)
        log_var = self.x_to_z_log_var(h)

        return self.create_distribution(mu, log_var)

    def create_distribution(self, mu, log_var):
        dist = td.normal.Normal(mu, log_var.exp())
        return dist

    def mse_decoder(self, x_encoded, z, horizon):
        zx = torch.cat([x_encoded, z], dim=1)
        initial_h = self.relu(self.initial_h_layer(zx))

        # B x seq_len x 2
        init_state = -1 * torch.ones((x_encoded.shape[0], 2), device=self.device)
        input_ = torch.cat([zx, init_state], dim=1)
        h = initial_h

        res = []
        for _ in range(horizon):
            h = self.rnn_cell(input_, h)
            next_state = self.rnn_to_state(h)
            res.append(next_state)
            input_ = torch.cat([zx, next_state], dim=1)
        
        res = torch.stack(res, dim=1)
        return res

    def gmm_decoder(self, x, z):
        zx = torch.cat([x, z], dim=1)
        gmm_output = self.gmm(zx)
        return gmm_output
    
    def eval_loss(self, o_b, s_r, global_step=None):
        o_b = o_b.to(self.device).float()
        s_r = s_r.to(self.device).float()

        o_b_encoded = self.o_b_encoder(o_b) # B x H
        o_b_encoded = self.o_b_encoder(o_b) # B x H

        x_gmm = self.x_to_gmm(o_b_encoded)

        recon_loss = self.x_to_gmm.compute_loss(x_gmm, s_r)
        return recon_loss.mean()


    def compute_loss(self, o_b, s_r, global_step=None):
        """ o_b: the blue observation
            s_r: the red observation
        """ 

        o_b = o_b.to(self.device).float()
        s_r = s_r.to(self.device).float()

        o_b_encoded = self.o_b_encoder(o_b) # B x H
        s_r_encoded = self.s_r_encoder(s_r) # B x H

        x_gmm = self.x_to_gmm(o_b_encoded)
        xy = torch.cat([o_b_encoded, s_r_encoded], dim=1)
        z_gmm = self.xy_to_gmm(xy)

        kl_loss = kl_gaussian_mixtures_vectorized(x_gmm, z_gmm)
        recon_loss = self.xy_to_gmm.compute_loss(z_gmm, s_r)

        # q_dist = self.q_z_xy(o_b_encoded, s_r_encoded)
        # p_dist = self.p_z_x(o_b_encoded)

        # p_entropy = p_dist.entropy()
        # q_entropy = q_dist.entropy()

        # z = q_dist.rsample()

        # if self.gmm_decoder_bool:
        #     gmm_output = self.gmm_decoder(o_b_encoded, z)
        #     recon_loss = self.gmm.compute_loss(gmm_output, s_r)
        # else:
        #     horizon = s_r.shape[1]
        #     s_predicted = self.mse_decoder(o_b_encoded, z, horizon=horizon)
        #     recon_loss = torch.nn.MSELoss()(s_r, s_predicted)
        
        # kl_loss = td.kl.kl_divergence(q_dist, p_dist)

        loss = 40 * kl_loss.sum(dim=1) + recon_loss
        if self.writer is not None and global_step is not None:
            self.writer.add_scalar('loss/kl_loss', kl_loss.mean(), global_step)
            self.writer.add_scalar('loss/recon_loss', recon_loss.mean(), global_step)
            # self.writer.add_scalar('loss/p_entropy', p_entropy.mean(), global_step)
            # self.writer.add_scalar('loss/q_entropy', q_entropy.mean(), global_step)
            # self.writer.add_scalar('loss/loss', loss.mean(), global_step)
        return loss.mean()

