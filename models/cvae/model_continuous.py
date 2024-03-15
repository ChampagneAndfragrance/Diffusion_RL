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
from models.gnn.gnn import GNNLSTM

class CVAEContinuous(nn.Module):

    def __init__(self, x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, gmm_decoder=False, input_type="vector"):
        super(CVAEContinuous, self).__init__()
        self.z_logit_clip = None

        if input_type == "vector":
            self.o_b_encoder = EncoderRNN(x_dim, encoder_hidden_dim)
        else:
            # input_dim, hidden_dim, gnn_hidden, 
            self.o_b_encoder = GNNLSTM(x_dim, 8, encoder_hidden_dim)
            encoder_hidden_dim = encoder_hidden_dim + 3
            
        self.s_r_encoder = EncoderRNN(2, future_hidden_dim)
        self.x_to_z_mu = nn.Linear(encoder_hidden_dim, z_dim)
        self.x_to_z_log_var = nn.Linear(encoder_hidden_dim, z_dim)

        self.xy_to_z_mu = nn.Linear(encoder_hidden_dim + future_hidden_dim, z_dim)
        self.xy_to_z_log_var = nn.Linear(encoder_hidden_dim + future_hidden_dim, z_dim)

        if gmm_decoder:
            self.gmm = MixureDecoderMultiHead(encoder_hidden_dim + z_dim, output_dim=2, num_gaussians=2, num_heads=13)
        else:
            self.initial_h_layer = nn.Linear(z_dim + encoder_hidden_dim, 32)
            self.rnn_cell = nn.GRUCell(encoder_hidden_dim + z_dim + 2, 32) # input, hidden dim
            self.rnn_to_state = nn.Linear(32, 2)

        self.gmm_decoder_bool = gmm_decoder        
        
        self.relu = nn.ReLU()

        self.writer = None

        self.z_dim = z_dim
    @property
    def device(self):
        return next(self.parameters()).device

    def predict(self, o_b, num_samples):
        # o_b = o_b.to(self.device).float()
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
        # o_b = o_b.to(self.device).float()
        s_r = s_r.to(self.device).float()

        o_b_encoded = self.o_b_encoder(o_b) # B x H
        p_dist = self.p_z_x(o_b_encoded)

        z = p_dist.rsample()

        if self.gmm_decoder_bool:
            gmm_output = self.gmm_decoder(o_b_encoded, z)
            recon_loss = self.gmm.compute_loss(gmm_output, s_r)
        else:
            horizon = s_r.shape[1]
            s_predicted = self.mse_decoder(o_b_encoded, z, horizon=horizon)
            recon_loss = torch.nn.MSELoss()(s_r, s_predicted)            

        return recon_loss.mean()

    def compute_loss(self, o_b, s_r, global_step=None):
        """ o_b: the blue observation
            s_r: the red observation
        """ 

        # o_b = o_b.to(self.device).float()
        s_r = s_r.to(self.device).float()

        o_b_encoded = self.o_b_encoder(o_b) # B x H
        s_r_encoded = self.s_r_encoder(s_r) # B x H
        
        q_dist = self.q_z_xy(o_b_encoded, s_r_encoded)
        p_dist = self.p_z_x(o_b_encoded)

        p_entropy = p_dist.entropy()
        q_entropy = q_dist.entropy()

        z = q_dist.rsample()

        if self.gmm_decoder_bool:
            gmm_output = self.gmm_decoder(o_b_encoded, z)
            recon_loss = self.gmm.compute_loss(gmm_output, s_r)
        else:
            horizon = s_r.shape[1]
            s_predicted = self.mse_decoder(o_b_encoded, z, horizon=horizon)
            recon_loss = torch.nn.MSELoss()(s_r, s_predicted)
        
        kl_loss = td.kl.kl_divergence(q_dist, p_dist)

        loss = 100 * kl_loss + recon_loss
        if self.writer is not None and global_step is not None:
            self.writer.add_scalar('loss/kl_loss', kl_loss.mean(), global_step)
            self.writer.add_scalar('loss/recon_loss', recon_loss.mean(), global_step)
            self.writer.add_scalar('loss/p_entropy', p_entropy.mean(), global_step)
            self.writer.add_scalar('loss/q_entropy', q_entropy.mean(), global_step)
            # self.writer.add_scalar('loss/loss', loss.mean(), global_step)
        return loss.mean()

