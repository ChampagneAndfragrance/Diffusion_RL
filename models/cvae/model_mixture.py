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
from models.decoder import MixtureDensityDecoder

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.gumbel import Gumbel

def sample_mixture_gumbel(pi, mu, sigma, temp=0.1):
    """ 
    
    Given a mixture of gaussians, sample from the mixture in a way that we can backpropagate through

    pi: (B, G)
    mu: (B, G, D)
    sigma: (B, G, D)

    First, sample categorically from the mixture pi with gumbel softmax.
    Then, sample from the corresponding gaussian by multiplying and adding with mean and std.

    Returns shape of (B, D) where we have batch size and dimension of gaussian

    """
    # ensure all the dimensions are correct
    assert pi.size(0) == mu.size(0) == sigma.size(0)
    assert pi.size(1) == mu.size(1) == sigma.size(1)
    assert mu.size(2) == sigma.size(2)

    # sample from gumbel softmax
    m = Gumbel(torch.zeros_like(pi), torch.ones_like(pi))
    g = m.sample()
    gumbel_softmax = torch.softmax((torch.log(pi) + g)/temp, dim=-1) # (B, num_gaussians)

    # reparamaterize the gaussians
    eps = torch.randn_like(sigma)
    samples = mu + (eps * sigma)

    gumbel_weighted = torch.einsum('bgd,bg->bd', [samples, gumbel_softmax])
    return gumbel_weighted
    

def kl_gaussian_mixtures_vectorized(mix_1, mix_2):
    pi_a, mu_a, sigma_a = mix_1 # B x n_gaussians x 2
    pi_b, mu_b, sigma_b = mix_2

    distribution_dim = mu_a.size(2)

    # pi_a = pi_a.view(pi_a.shape[0], -1)
    # mu_a = mu_a.view(mu_a.shape[0], -1, mu_a.shape[3])
    # sigma_a = sigma_a.view(sigma_a.shape[0], -1, sigma_a.shape[3])

    # pi_b = pi_b.view(pi_b.shape[0], -1)
    # mu_b = mu_b.view(mu_b.shape[0], -1, mu_b.shape[3])
    # sigma_b = sigma_b.view(sigma_b.shape[0], -1, sigma_b.shape[3])

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
    kls_num_reshaped = kls_num.reshape(kl_num.shape[0], n_gaussians, n_gaussians, distribution_dim) # B x n_gaussians x n_gaussians x 2
    num = kls_num_reshaped.sum(dim=2) # B x n_gaussians x 2

    p_den = Normal(mu_a_tog, sigma_a_tog)
    q_den = Normal(mu_b_p, sigma_b_p)
    kl_den = kl_divergence(p_den, q_den)
    kls_den = torch.einsum('bxd,bx->bxd', torch.exp(-kl_den), pi_b_p) # B x n_gaussians**2
    kls_den_reshaped = kls_den.reshape(kl_den.shape[0], n_gaussians, n_gaussians, distribution_dim) # B x n_gaussians x n_gaussians x 2
    den = kls_den_reshaped.sum(dim=2) # B x n_gaussians x 2

    divided = num / den # B x num_gaussians x 2
    # print(divided.shape)
    res = torch.einsum('bgd,bg->bgd', torch.log(divided), pi_a)
    return res.sum(dim=1)

class CVAEMixture(nn.Module):

    def __init__(self, x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, num_heads, decoder_type="mse"):
        super(CVAEMixture, self).__init__()
        self.decoder_type = decoder_type
        self.num_heads = num_heads
        self.z_dim = z_dim
        self.z_logit_clip = None
        self.o_b_encoder = EncoderRNN(x_dim, encoder_hidden_dim)
        self.s_r_encoder = EncoderRNN(2, future_hidden_dim)

        self.x_to_gmm = MixtureDensityDecoder(encoder_hidden_dim, num_gaussians=2, output_dim=z_dim)
        self.xy_to_gmm = MixtureDensityDecoder(encoder_hidden_dim + future_hidden_dim, num_gaussians=2, output_dim=z_dim)

        if decoder_type == "mixture":
            self.gmm = MixureDecoderMultiHead(encoder_hidden_dim + z_dim, output_dim=2, num_gaussians=2, num_heads=num_heads)
        elif decoder_type == "mse":
            self.initial_h_layer = nn.Linear(z_dim + encoder_hidden_dim, 32)
            self.rnn_cell = nn.GRUCell(encoder_hidden_dim + z_dim + 2, 32) # input, hidden dim
            self.rnn_to_state = nn.Linear(32, 2)
        elif decoder_type == "gaussian":
            self.zx_to_y_mu = nn.Linear(encoder_hidden_dim + z_dim, self.num_heads * 2)
            nn.init.xavier_normal_(self.zx_to_y_mu.weight)
            nn.init.zeros_(self.zx_to_y_mu.bias)

            self.zx_to_y_log_var = nn.Linear(encoder_hidden_dim + z_dim, self.num_heads * 2)
            nn.init.zeros_(self.zx_to_y_log_var.weight)
            nn.init.zeros_(self.zx_to_y_log_var.bias)

        
        self.relu = nn.ReLU()

        self.writer = None
    @property
    def device(self):
        return next(self.parameters()).device

    def predict(self, o_b, num_samples):
        o_b = o_b.to(self.device).float()
        o_b_encoded = self.o_b_encoder(o_b) # B x H
        x_gmm = self.x_to_gmm(o_b_encoded)

        total_outputs = []
        for sample in range(num_samples):
            z = sample_mixture_gumbel(*x_gmm)
            # z = torch.randn(o_b.shape[0], self.z_dim).to(self.device)

            if self.decoder_type == "gmm":
                gmm_output = self.gmm_decoder(o_b_encoded, z)
                total_outputs.append(gmm_output)
                gmm_output = self.gmm_decoder(o_b_encoded, z)
                total_outputs.append(gmm_output)
            elif self.decoder_type == "mse":
                s_predicted = self.mse_decoder(o_b_encoded, z, horizon=13)
                total_outputs.append(s_predicted)
            elif self.decoder_type == "gaussian":
                zx = torch.cat([o_b_encoded, z], dim=1)
                pos_mu = self.zx_to_y_mu(zx)
                pos_log_var = self.zx_to_y_log_var(zx)
                out_dist = td.normal.Normal(pos_mu, pos_log_var.exp())
                total_outputs.append(out_dist)
        
        if self.decoder_type == "gmm":
            return total_outputs
        elif self.decoder_type == "mse":
            return torch.stack(total_outputs)
        else:
            return total_outputs

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

        z = sample_mixture_gumbel(*x_gmm)

        if self.decoder_type == "gmm":
            gmm_output = self.gmm_decoder(o_b_encoded, z)
            recon_loss = self.gmm.compute_loss(gmm_output, s_r)
        elif self.decoder_type == "mse":
            horizon = s_r.shape[1]
            s_predicted = self.mse_decoder(o_b_encoded, z, horizon=horizon)
            recon_loss = torch.nn.MSELoss()(s_r, s_predicted)
        elif self.decoder_type == "gaussian":
            zx = torch.cat([o_b_encoded, z], dim=1)
            pos_mu = self.zx_to_y_mu(zx)
            pos_log_var = self.zx_to_y_log_var(zx)
            out_dist = td.normal.Normal(pos_mu, pos_log_var.exp())
            recon_loss = -out_dist.log_prob(s_r.view(s_r.shape[0], -1)).sum(dim=1)

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

        z = sample_mixture_gumbel(*z_gmm)

        if self.decoder_type == "gmm":
            gmm_output = self.gmm_decoder(o_b_encoded, z)
            recon_loss = self.gmm.compute_loss(gmm_output, s_r)
        elif self.decoder_type == "mse":
            horizon = s_r.shape[1]
            s_predicted = self.mse_decoder(o_b_encoded, z, horizon=horizon)
            recon_loss = torch.nn.MSELoss()(s_r, s_predicted)
        elif self.decoder_type == "gaussian":
            zx = torch.cat([o_b_encoded, z], dim=1)
            # pos_mu = (torch.tanh(self.zx_to_y_mu(zx)) + 1)
            pos_mu = self.zx_to_y_mu(zx)
            pos_log_var = self.zx_to_y_log_var(zx)
            out_dist = td.normal.Normal(pos_mu, pos_log_var.exp())
            recon_loss = -out_dist.log_prob(s_r.view(s_r.shape[0], -1)).sum(dim=1)
        
        loss = 100*kl_loss.sum(dim=1) + recon_loss
        if self.writer is not None and global_step is not None:
            self.writer.add_scalar('loss/kl_loss', kl_loss.mean(), global_step)
            self.writer.add_scalar('loss/recon_loss', recon_loss.mean(), global_step)
            # self.writer.add_scalar('loss/p_entropy', p_entropy.mean(), global_step)
            # self.writer.add_scalar('loss/q_entropy', q_entropy.mean(), global_step)
            # self.writer.add_scalar('loss/loss', loss.mean(), global_step)
        return loss.mean()

