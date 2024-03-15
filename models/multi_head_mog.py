import sys, os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.distributions import Normal
import math
import torch.distributions as D
# from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch, insert_dim, NoNorm


import math
import torch
from torch.distributions import Categorical
from scipy.stats import multivariate_normal
from models.utils import log_prob


class MixureDecoderMultiHead(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, log_std_init=0.0):
        super(MixureDecoderMultiHead, self).__init__()

        self.fc = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)

        input_dim = 32

        # self.batch_size = batch_size
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Linear(input_dim, num_gaussians*num_heads)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=2)

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.num_heads = num_heads

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):

        x = self.fc(x)
        x = self.dropout(x)

        batch_size = x.size(0)
        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
        pi = self.softmax(pi)

        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(batch_size, self.num_heads,  self.num_gaussians, self.output_dim)

        mu = self.mu(x)
        mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        sigma = nn.ELU()(sigma) + 1e-15
        # sigma = torch.clamp(mu, min=0.00001)
        # sigma = self.relu(sigma)
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)

        losses = self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        loss = torch.sum(losses, dim=1).mean()
        return loss

    def get_stats(self, nn_output, red_locs):
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)
        log_prob = -self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        ade = self.average_displacement_error_from_mode(nn_output, red_locs)
        dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps = self.distance_threshold_metric_with_single_mean(
            nn_output, red_locs)
        return log_prob, ade, dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps
        

    def mdn_negative_log_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training 
        
        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        target = target.unsqueeze(2).expand_as(sigma)
        neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
            ((target - mu) / sigma)**2 / 2
        
        # (B, num_heads, num_gaussians)
        inner = torch.log(pi) + torch.sum(neg_logprob, 3) # Sum the log probabilities of (x, y) for each 2D Gaussian
        return -torch.logsumexp(inner, dim=2)

    def mdn_negative_log_likelihood_loss(self, nn_output, target):
        """
        Compute the negative log likelihood loss for a MoG model.
        """
        pi, mu, sigma = nn_output
        return self.mdn_negative_log_likelihood(pi, mu, sigma, target)
    def average_displacement_error_from_mode(self, nn_output, target):
        pi, mu, sigma = nn_output
        num_mixtures = pi.shape[-1]
        target = target.unsqueeze(dim=2).repeat(1, 1, num_mixtures, 1)
        mse_error = torch.linalg.norm((mu - target), dim=-1)
        mse_error = torch.sum(pi * mse_error, axis=-1)  # Calculate weighted average of mse errors
        return mse_error

    def distance_threshold_metric(self, nn_output, target, dist_threshold=0.025):
        """
        Given mean, and logstd predicted from the filtering module, calculate the likelihood
        of the fugitive's ground truth location from the predicted distribution
        :param mean: (np.array) Mean of the predicted distribution from the filtering module
        :param logstd: (np.array) Logstd of the predicted distribution from the filtering module
        :param true_location: (np.array) Ground Truth location of the fugitive (x,y)
        :return:
            prob: The probability of the fugitive being at the ground truth location
                  as predicted by the filtering module's distribution
        """
        pi, mu, sigma = nn_output
        bs, seq_len, num_mixtures, n_dim = sigma.shape
        # var = np.exp(logstd) ** 2
        var = torch.clamp(sigma ** 2, min=1e-5, max=1)

        var = var.repeat(1, 1, 1, n_dim)
        var = var.reshape(bs, seq_len, num_mixtures, n_dim, n_dim)

        cov = torch.ones(bs, seq_len, num_mixtures, n_dim, n_dim).to(self.device)
        probs = torch.empty(pi.shape).to('cpu')
        mahalanobis_dist = torch.empty(pi.shape).to('cpu').detach().numpy()

        num_steps_conf_thresh = np.zeros((bs, seq_len))

        target = target.unsqueeze(dim=2).repeat(1, 1, num_mixtures, 1)

        # Torch does not have CDF for multi-variate normal.
        # Hence converting to numpy and using scipy
        target = target.to('cpu').detach().numpy()
        pi = pi.to('cpu').detach()
        mu = mu.to('cpu').detach().numpy()
        cov = cov.to('cpu').detach().numpy()
        var = var.to('cpu').detach().numpy()

        # TODO: @Manisha get rid of the for loops
        for b in range(bs):
            for s in range(seq_len):
                cov[b, s] = cov[b, s] * np.repeat(np.eye(n_dim)[np.newaxis, :, :], num_mixtures, axis=0)
                cov[b, s] = cov[b, s] * var[b, s]

                # CDF calculates from -infty to the upper limit.
                # Therefore subtracting the lower limit to calculate the cdf between lower limit to upper limit instead of -infty to upper limit
                for i in range(num_mixtures):
                    probs[b, s, i] = multivariate_normal.cdf(target[b, s, i] + dist_threshold, mean=mu[b, s, i], cov=cov[b, s, i]) - \
                                     multivariate_normal.cdf(target[b, s, i] - dist_threshold, mean=mu[b, s, i], cov=cov[b, s, i])
                    x_mean = target[b, s, i] - mu[b, s, i]

                    mahalanobis_dist[b, s, i] = np.sqrt(x_mean.T.dot(np.linalg.inv(cov[b, s, i])).dot(x_mean))

        # total_prob = torch.sum(pi * probs, axis=-1)
        total_prob = torch.max(probs, axis=-1)[0]

        # Calculate 1-sigma counts
        k = np.zeros_like(mahalanobis_dist)
        k[mahalanobis_dist < 1] = 1
        one_sigma_count = torch.sum(pi * k, axis=-1)

        # Calculate 2-sigma counts
        k = np.zeros_like(mahalanobis_dist)
        k[mahalanobis_dist < 2] = 1
        two_sigma_count = torch.sum(pi * k, axis=-1)

        # Calculate 3-sigma counts
        k = np.zeros_like(mahalanobis_dist)
        k[mahalanobis_dist < 3] = 1
        three_sigma_count = torch.sum(pi * k, axis=-1)

        # Distance threshold metric
        k = np.zeros_like(probs)
        k[probs >= 0.5] = 1
        num_steps_conf_thresh = torch.sum(pi * k, axis=-1)

        return total_prob, one_sigma_count, two_sigma_count, three_sigma_count, num_steps_conf_thresh

    def distance_threshold_metric_with_single_mean(self, nn_output, target, dist_threshold=0.05):
        pi, mu, sigma = nn_output
        bs, seq_len, num_mixtures, n_dim = sigma.shape

        # Get the means and sigmas of the gaussian with the highest mixing coeff
        idx = torch.argmax(pi, dim=-1, keepdim=True)
        mu_0 = torch.gather(mu[:, :, :, 0], 1, idx)
        mu_1 = torch.gather(mu[:, :, :, 1], 1, idx)
        mu = torch.cat((mu_0, mu_1), dim=-1)

        sigma_0 = torch.gather(sigma[:, :, :, 0], 1, idx)
        sigma_1 = torch.gather(sigma[:, :, :, 1], 1, idx)
        sigma = torch.cat((sigma_0, sigma_1), dim=-1)

        var = torch.clamp(sigma ** 2, min=1e-5, max=1)

        var = var.repeat(1, 1, n_dim)
        var = var.reshape(bs, seq_len, n_dim, n_dim)

        cov = torch.ones(bs, seq_len, n_dim, n_dim).to(self.device)
        probs = torch.empty((bs, seq_len)).to('cpu')
        mahalanobis_dist = torch.empty((bs, seq_len)).to('cpu').detach().numpy()

        # Torch does not have CDF for multi-variate normal.
        # Hence converting to numpy and using scipy
        target = target.to('cpu').detach().numpy()
        pi = pi.to('cpu').detach()
        mu = mu.to('cpu').detach().numpy()
        cov = cov.to('cpu').detach().numpy()
        var = var.to('cpu').detach().numpy()

        for b in range(bs):
            for s in range(seq_len):
                cov[b, s] = cov[b, s] * np.eye(n_dim)
                cov[b, s] = cov[b, s] * var[b, s]

                probs[b, s] = multivariate_normal.cdf(target[b, s] + dist_threshold, mean=mu[b, s],
                                                         cov=cov[b, s]) - \
                                 multivariate_normal.cdf(target[b, s] - dist_threshold, mean=mu[b, s],
                                                         cov=cov[b, s])
                x_mean = target[b, s] - mu[b, s]

                mahalanobis_dist[b, s] = np.sqrt(x_mean.T.dot(np.linalg.inv(cov[b, s])).dot(x_mean))


        # Calculate 1-sigma counts
        one_sigma_count = np.zeros_like(mahalanobis_dist)
        one_sigma_count[mahalanobis_dist < 1] = 1
        # one_sigma_count = torch.Tensor(one_sigma_count).to(self.device)

        # Calculate 2-sigma counts
        two_sigma_count = np.zeros_like(mahalanobis_dist)
        two_sigma_count[mahalanobis_dist < 2] = 1
        # two_sigma_count = torch.Tensor(two_sigma_count).to(self.device)

        # Calculate 3-sigma counts
        three_sigma_count = np.zeros_like(mahalanobis_dist)
        three_sigma_count[mahalanobis_dist < 3] = 1
        # three_sigma_count = torch.Tensor(three_sigma_count).to(self.device)

        # Distance threshold metric
        num_steps_conf_thresh = np.zeros_like(probs)
        num_steps_conf_thresh[probs >= 0.5] = 1

        return probs, one_sigma_count, two_sigma_count, three_sigma_count, num_steps_conf_thresh

    def predict(self, x):
        return self.forward(x)


class ContrastiveMixureDecoderMultiHead(nn.Module):
    """
    Modified the Mixture Density Decoder to include a contrastive loss by learning
    separate embeddings for timesteps when the fugitive was detected vs when the
    fugitive was not detected
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, loss_type="mse", log_std_init=0.0):
        super(ContrastiveMixureDecoderMultiHead, self).__init__()

        self.fc = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)

        input_dim = 32

        # self.batch_size = batch_size
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        # Loss type MSE or cosine similarity
        self.loss_type = loss_type

        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Linear(input_dim, num_gaussians * num_heads)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=2)

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.num_heads = num_heads

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

        # For computing contrastive loss...
        self.positive_out = None    # Store the positive embedding
        self.negative_out = None    # Store the negative embedding

        # Store the most recent timestep when the fugitive was detected
        self.last_positive_timestep = -1
        # Store the most recent timestep when the fugitive was NOT detected
        self.last_negative_timestep = -1


    def forward(self, x):
        self.positive_out, self.negative_out, self.last_positive_timestep, self.last_negative_timestep = x

        # Concatenate the two embeddings from positive and negative observations
        concat_embed = torch.cat((self.positive_out, self.negative_out), dim=-1)

        x = self.fc(concat_embed)
        x = self.dropout(x)

        batch_size = x.size(0)
        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
        pi = self.softmax(pi)

        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        mu = self.mu(x)
        mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        sigma = nn.ELU()(sigma) + 1e-15
        # sigma = torch.clamp(mu, min=0.00001)
        # sigma = self.relu(sigma)
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)

        losses = self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        loss = torch.sum(losses, dim=1).mean() + self.compute_contrastive_loss(contrastive_wt=1.0)
        return loss

    def get_stats(self, nn_output, red_locs):
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)
        return -self.mdn_negative_log_likelihood_loss(nn_output, red_locs)

    def mdn_negative_log_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training

        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        target = target.unsqueeze(2).expand_as(sigma)
        neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
                      ((target - mu) / sigma) ** 2 / 2

        # (B, num_heads, num_gaussians)
        inner = torch.log(pi) + torch.sum(neg_logprob, 3)  # Sum the log probabilities of (x, y) for each 2D Gaussian
        return -torch.logsumexp(inner, dim=2)

    def mdn_negative_log_likelihood_loss(self, nn_output, target):
        """
        Compute the negative log likelihood loss for a MoG model.
        """
        pi, mu, sigma = nn_output
        return self.mdn_negative_log_likelihood(pi, mu, sigma, target)

    def get_contrastive_loss(self):
        return self._contrastive_loss

    def predict(self, x):
        return self.forward(x)


class ModifiedContrastiveMixureDecoderMultiHead(nn.Module):
    """
    Modified the Mixture Density Decoder to include a contrastive loss by learning
    separate embeddings for timesteps when the fugitive was detected vs when the
    fugitive was not detected
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, loss_type="mse", log_std_init=0.0):
        super(ModifiedContrastiveMixureDecoderMultiHead, self).__init__()

        self.fc = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)

        input_dim = 32

        # self.batch_size = batch_size
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        # Loss type MSE or cosine similarity
        self.loss_type = loss_type

        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Linear(input_dim, num_gaussians * num_heads)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=2)

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim, output_dim * num_gaussians * num_heads)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.num_heads = num_heads

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

        # Store the most recent timestep when the fugitive was detected
        self.last_positive_timestep = -1
        # Store the most recent timestep when the fugitive was NOT detected
        self.last_negative_timestep = -1

    def forward(self, x):
        concat_embed, self.contrastive_loss = x

        x = self.fc(concat_embed)
        x = self.dropout(x)

        batch_size = x.size(0)
        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
        pi = self.softmax(pi)

        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        mu = self.mu(x)
        mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        sigma = nn.ELU()(sigma) + 1e-15
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)

        losses = self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        loss = torch.sum(losses, dim=1).mean() + self.contrastive_loss
        return loss

    def get_stats(self, nn_output, red_locs):
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)
        return -self.mdn_negative_log_likelihood_loss(nn_output, red_locs)

    def mdn_negative_log_likelihood(self, pi, mu, sigma, target):
        """ Use torch.logsumexp for more stable training

        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        target = target.unsqueeze(2).expand_as(sigma)
        neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
                      ((target - mu) / sigma) ** 2 / 2

        # (B, num_heads, num_gaussians)
        inner = torch.log(pi) + torch.sum(neg_logprob, 3)  # Sum the log probabilities of (x, y) for each 2D Gaussian
        return -torch.logsumexp(inner, dim=2)

    def mdn_negative_log_likelihood_loss(self, nn_output, target):
        """
        Compute the negative log likelihood loss for a MoG model.
        """
        pi, mu, sigma = nn_output
        return self.mdn_negative_log_likelihood(pi, mu, sigma, target)

    def get_contrastive_loss(self):
        return self.contrastive_loss


class ContrastiveGNNMixureDecoderMultiHead(MixureDecoderMultiHead):
    """
    Modified the Mixture Density Decoder to include a contrastive loss by learning
    separate embeddings for timesteps when the fugitive was detected vs when the
    fugitive was not detected
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, loss_type="mse", log_std_init=0.0):
        super().__init__(input_dim, output_dim, num_gaussians, num_heads, log_std_init)

        # For computing contrastive loss...
        self.positive_out = None    # Store the positive embedding
        self.negative_out = None    # Store the negative embedding

        # Store the most recent timestep when the fugitive was detected
        self.last_positive_timestep = -1
        # Store the most recent timestep when the fugitive was NOT detected
        self.last_negative_timestep = -1

        self._contrastive_loss = 0

    def forward(self, x):
        concat_embed, self._contrastive_loss = x

        # Concatenate the two embeddings from positive and negative observations
        # concat_embed = torch.cat((self.positive_out, self.negative_out), dim=-1)
        # concat_embed = torch.cat((concat_embed, hideout_obs, timestep_obs), dim=-1)

        x = self.fc(concat_embed)
        x = self.dropout(x)

        batch_size = x.size(0)
        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
        pi = self.softmax(pi)

        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        mu = self.mu(x)
        mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)

        sigma = nn.ELU()(sigma) + 1e-15
        # sigma = torch.clamp(mu, min=0.00001)
        # sigma = self.relu(sigma)
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        red_locs = red_locs.view(-1, self.num_heads, self.output_dim)

        losses = self.mdn_negative_log_likelihood_loss(nn_output, red_locs)
        self.logp_loss = torch.sum(losses, dim=1).mean()
        loss = self.logp_loss + self._contrastive_loss
        return loss

    def get_contrastive_loss(self):
        return self._contrastive_loss
    
    def get_logp_loss(self):
        return self.logp_loss

    def delta_empirical_sigma_value(self, nn_output, targets):
        pi, mu, sigma = nn_output

    def predict(self, x):
        return self.forward(x)



class Seq2SeqContrastiveAutoregressiveDecoder(ContrastiveGNNMixureDecoderMultiHead):
    """
    Autoregressive decoder for contrastive learning
    """
    def __init__(self, input_dim, output_dim=2, num_gaussians=2, num_heads=1, log_std_init=0.0,
                 teacher_forcing=0.5):
        super(Seq2SeqContrastiveAutoregressiveDecoder, self).__init__(input_dim, output_dim, num_gaussians, num_heads, log_std_init)

        self.fc = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)

        input_dim = 19

        # self.batch_size = batch_size
        # self.num_gaussians = num_gaussians
        # self.output_dim = output_dim

        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Linear(input_dim*2 + output_dim, num_gaussians).to(self.device)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=-1)

        self.sigma = nn.Linear(input_dim*2 + output_dim, output_dim * num_gaussians).to(self.device)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim*2 + output_dim, output_dim * num_gaussians).to(self.device)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        # self.num_heads = num_heads

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

        # For the autoregressive
        self.teacher_forcing_ratio = teacher_forcing

        n_layers = 1
        self.rnn = nn.GRU(output_dim + input_dim, input_dim, n_layers, batch_first=True).to(self.device)
        self._contrastive_loss = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def decoder_pass(self, input, hidden, context):
        input_con = torch.cat((input, context), dim=1)
        input_con = input_con.unsqueeze(dim=1)
        output, hidden = self.rnn(input_con, hidden)

        output = torch.cat((input, hidden.squeeze(0), context), dim=-1)
        pi_prediction = self.softmax(self.pi(output))
        mu_prediction = self.mu(output)
        sigma_prediction = torch.exp(self.sigma(output))

        return pi_prediction, mu_prediction, sigma_prediction, hidden

    def forward(self, x):
        """
        Autoregressive policy. Input should be a tuple of encoder output and the ground truth sequence
        We need the ground truth sequence for teacher forcing
        :param x:
        :return:
        """
        # First input to the decoder is the fugitive's detected location
        # Context is the embedding from the encoder
        # Target are the ground truth fugitive locations to be predicted
        context, self._contrastive_loss, input, target = x

        batch_size = target.shape[0]
        seq_len = target.shape[1]

        # tensor to store decoder outputs
        pi_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians).to(self.device)
        mu_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians, self.output_dim).to(self.device)
        sigma_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians, self.output_dim).to(self.device)

        # context also used as the initial hidden state of the encoder
        hidden = context.unsqueeze(0)

        for t in range(seq_len):
            pi_output, mu_output, sigma_output, hidden = self.decoder_pass(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            pi_outputs[:, t, :] = pi_output
            mu_output = mu_output.reshape(-1, self.num_gaussians, self.output_dim)
            mu_outputs[:, t, :] = mu_output
            sigma_output = sigma_output.reshape(-1, self.num_gaussians, self.output_dim)
            # sigma_output = (sigma_output)**2  ## DO NOT SQUARE THE SIGMA. THIS IS STD NOT VARIANCE
            sigma_output = nn.ELU()(sigma_output) + 1e-15
            sigma_outputs[:, t, :] = sigma_output

            # decide if we are going to use teacher forcing or not
            teacher_force = torch.rand(1) < self.teacher_forcing_ratio

            # if teacher forcing, use actual next ground truth location as next input
            # if not, use predicted output
            if teacher_force:
                input = target[:, t]
            else:
                # Take the mode with the max mixing coeff
                mode_idx = torch.argmax(pi_output, dim=-1)
                input = mu_output[torch.arange(batch_size), mode_idx, :]

        return pi_outputs, mu_outputs, sigma_outputs

    def predict(self, x):
        """
        Autoregressive policy. Input should be a tuple of encoder output and the ground truth sequence
        We need the ground truth sequence for teacher forcing
        :param x:
        :return:
        """
        # First input to the decoder is the fugitive's detected location
        # Context is the embedding from the encoder
        # Target are the ground truth fugitive locations to be predicted
        context, self._contrastive_loss, input, target = x

        batch_size = target.shape[0]
        seq_len = target.shape[1]

        # tensor to store decoder outputs
        pi_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians).to(self.device)
        mu_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians, self.output_dim).to(self.device)
        sigma_outputs = torch.zeros(batch_size, seq_len, self.num_gaussians, self.output_dim).to(self.device)

        # context also used as the initial hidden state of the encoder
        hidden = context.unsqueeze(0)

        for t in range(seq_len):
            pi_output, mu_output, sigma_output, hidden = self.decoder_pass(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            pi_outputs[:, t, :] = pi_output
            mu_output = mu_output.reshape(-1, self.num_gaussians, self.output_dim)
            mu_outputs[:, t, :] = mu_output
            sigma_output = sigma_output.reshape(-1, self.num_gaussians, self.output_dim)
            # sigma_output = (sigma_output)**2  ## DO NOT SQUARE THE SIGMA. THIS IS STD NOT VARIANCE
            sigma_output = nn.ELU()(sigma_output) + 1e-15
            sigma_outputs[:, t, :] = sigma_output

            # No teacher forcing here...
            # Take the mode with the max mixing coeff
            mode_idx = torch.argmax(pi_output, dim=-1)
            input = mu_output[torch.arange(batch_size), mode_idx, :]

        return pi_outputs, mu_outputs, sigma_outputs
