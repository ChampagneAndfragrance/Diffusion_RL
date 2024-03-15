import torch
import math
import torch.nn as nn
from models.encoders import EncoderRNN
from torch.distributions import Normal
from models.utils import log_prob

class BlueMI(nn.Module):
    """ This model predicts a single point as its output """
    def __init__(self, input_dim, output_dim, bayesian_embedding_dim = 4, h1=16, h2=32, non_linear=nn.ReLU()):
        super(BlueMI, self).__init__()

        self.encoder = EncoderRNN(input_dim, h1)
        self.input_dim = input_dim
        self.h1 = h1
        self.h2 = h2
        self.bayesian_embedding_dim = bayesian_embedding_dim
        self.output_dim = output_dim
        self.non_linear = non_linear

        network_input_dim = self.h1 + bayesian_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(network_input_dim, h1),
            self.non_linear,
            nn.Linear(h1, h2),
            self.non_linear,
            nn.Linear(h2, output_dim),
            )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, embedding_tensor):
        """ x is of shape (batch_size, input_dim)
        embedding_tensor is of shape (batch_size, bayesian_embedding_dim) which holds parameters """
        # predict the next location of the agent
        # ins = torch.cat((x, embedding_tensor), dim=1)
        z = self.encoder(x)
        ins = torch.cat((z, embedding_tensor), dim=1)
        return self.network(ins), z


class BlueMIGaussian(nn.Module):
    """ This one predicts a normal gaussian as its output rather than a single point"""
    def __init__(self, input_dim, output_dim, bayesian_embedding_dim = 4, h1=16, h2=32, non_linear=nn.ReLU()):
        super(BlueMIGaussian, self).__init__()

        self.encoder = EncoderRNN(input_dim, h1)
        self.input_dim = input_dim
        self.h1 = h1
        self.h2 = h2
        self.bayesian_embedding_dim = bayesian_embedding_dim
        self.output_dim = output_dim
        self.non_linear = non_linear

        network_input_dim = self.h1 + bayesian_embedding_dim
        # self.network = nn.Sequential(
        #     nn.Linear(network_input_dim, h1),
        #     self.non_linear,
        #     nn.Linear(h1, h2),
        #     self.non_linear,
        #     nn.Linear(h2, output_dim*2),
        #     )


        # for layer in self.network:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)

        self.sigma = nn.Linear(network_input_dim, output_dim)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(network_input_dim, output_dim)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, x, embedding_tensor):
        """ x is of shape (batch_size, input_dim)
        embedding_tensor is of shape (batch_size, bayesian_embedding_dim) which holds parameters """
        # predict the next location of the agent
        # ins = torch.cat((x, embedding_tensor), dim=1)
        z = self.encoder(x)
        ins = torch.cat((z, embedding_tensor), dim=1)

        mean = self.mu(ins)
        sigma = self.sigma(ins)
        # sigma = nn.ELU()(sigma) + 1e-15
        sigma = torch.exp(sigma)

        return (mean, sigma), z
    
    @property
    def device(self):
        return next(self.parameters()).device

class BlueMIMixture(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures,  h1, h2, non_linear=nn.ReLU()):
        super(BlueMIMixture, self).__init__()

        self.encoder = EncoderRNN(input_dim, h1)
        self.input_dim = input_dim
        self.h1 = h1
        self.h2 = h2
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.non_linear = non_linear

        network_input_dim = self.h1 + num_mixtures

        self.fc = nn.Linear(network_input_dim, h1)

        self.pi = nn.Linear(h1, 1)
        nn.init.xavier_normal_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)

        self.mu = nn.Linear(h1, output_dim)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

        self.sigma = nn.Linear(h1, output_dim)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        pi_categoricals = torch.eye(self.num_mixtures).to(self.device)
        pis, mus, sigmas = [], [], []
        z = self.encoder(x)
        batch_size = z.shape[0]
        

        for i in range(self.num_mixtures):
            pi_cat = pi_categoricals[i].unsqueeze(0).repeat(batch_size, 1)
            ins = torch.cat((z, pi_cat), dim=1)
            ins = self.fc(ins)
            ins = self.non_linear(ins)

            
            pi = self.pi(ins) # B x 1
            mu = self.mu(ins) # B x 2

            sigma = self.sigma(ins) # B x 2
            # sigma = torch.exp(sigma) # B x 2
            # sigma = nn.ELU()(sigma) + 1e-15
            sigma = torch.clip(torch.exp(sigma) + 1e-4, min=0, max=1)

            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

        pis = torch.cat(pis, dim=-1) # B x num_mixtures x 1
        pis = self.softmax(pis) # B x num_mixtures x 1
        mus = torch.stack(mus, dim=1) # B x num_mixtures x 2
        sigmas = torch.stack(sigmas, dim=1) # B x num_mixtures x 2
        
        return pis, mus, sigmas

    def forward_encoder(self, x):
        pi_categoricals = torch.eye(self.num_mixtures).to(self.device)
        pis, mus, sigmas = [], [], []
        z = self.encoder(x)
        batch_size = z.shape[0]
        

        pi_cats = []
        for i in range(self.num_mixtures):
            pi_cat = pi_categoricals[i].unsqueeze(0).repeat(batch_size, 1)
            ins = torch.cat((z, pi_cat), dim=1)
            ins = self.fc(ins)
            ins = self.non_linear(ins)
            
            pi = self.pi(ins) # B x 1
            mu = self.mu(ins) # B x 2

            sigma = self.sigma(ins) # B x 2
            # sigma = torch.exp(sigma) # B x 2
            # sigma = nn.ELU()(sigma) + 1e-15
            sigma = torch.clip(torch.exp(sigma) + 1e-4, min=0, max=1)

            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

            pi_cats.append(pi_cat)

        p = torch.cat(pis, dim=0)
        m = torch.cat(mus, dim=0)
        s = torch.cat(sigmas, dim=0)
        all_outputs = torch.cat((p, m, s), dim=1)


        pis = torch.cat(pis, dim=-1) # B x num_mixtures x 1
        pis = self.softmax(pis) # B x num_mixtures x 1
        mus = torch.stack(mus, dim=1) # B x num_mixtures x 2
        sigmas = torch.stack(sigmas, dim=1) # B x num_mixtures x 2
        
        pi_cats = torch.cat(pi_cats, dim=0) # B x num_mixtures x 3

        z_posterior = z.repeat((self.num_mixtures, 1))

        return (pis, mus, sigmas), (z_posterior, all_outputs), pi_cats


    def forward_embeds(self, x, embeds):
        """ Given a batch of embeddings, predict the location of the agent"""
        z = self.encoder(x)
        batch_size = z.shape[0]

        ins = torch.cat((z, embeds), dim=1)
        ins = self.fc(ins)
        ins = self.non_linear(ins)
        
        # pi = self.pi(ins) # B x 1
        mu = self.mu(ins) # B x 2
        sigma = self.sigma(ins) # B x 2
        # sigma = torch.exp(sigma) # B x 2
        # sigma = nn.ELU()(sigma) + 1e-15
        sigma = torch.clip(torch.exp(sigma) + 1e-4, min=0, max=1)

        return mu, sigma

    def compute_loss(self, x, red_locs):
        nn_output = self.forward(x)
        red_locs = red_locs.to(self.device)
        return mdn_negative_log_likelihood_loss(nn_output, red_locs)

    @property
    def device(self):
        return next(self.parameters()).device


def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """ Use torch.logsumexp for more stable training 
    
    This is equivalent to the mdn_loss but computed in a numerically stable way

    """
    target = target.unsqueeze(1).expand_as(sigma)
    # target = target.unsqueeze(2).expand_as(sigma)
    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
        ((target - mu) / sigma)**2 / 2
    
    inner = torch.log(pi) + torch.sum(neg_logprob, 2) # Sum the log probabilities of (x, y) for each 2D Gaussian
    return -torch.logsumexp(inner, dim=1)

def mdn_negative_log_likelihood_loss(nn_output, target):
    """
    Compute the negative log likelihood loss for a MoG model.
    """
    pi, mu, sigma = nn_output
    return mdn_negative_log_likelihood(pi, mu, sigma, target).mean()