import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

def sample_sequence_from_buffer(buffer, batch_size, sequence_length, observation_space, max_timesteps):
    """
    This function takes a stable baselines 3 buffer and samples a lentgh of n steps from it using timestamp from observation.
    """
    observation_shape = observation_space.shape

    batch = []
    indexes = np.random.randint(buffer.states.shape[0], size=batch_size)
    # reshape to rows, timestamps, features
    for index in indexes:
        last_observation = buffer.states[index]
        timestep = torch.round(last_observation[0] * max_timesteps).detach().cpu().numpy().astype(np.int)

        if timestep >= sequence_length - 1:
            sequence = torch.squeeze(buffer.states[index-sequence_length+1:index+1], 1)

        else:
            shape = (sequence_length-timestep-1,) + observation_shape
            empty_sequences = torch.zeros(shape).to("cuda")
            sequence = torch.squeeze(buffer.states[index-timestep:index+1], 1)
            sequence = torch.concat((empty_sequences, sequence), axis=0)

        batch.append(sequence)
        desired_shape = (sequence_length,) + observation_shape
        assert sequence.shape == desired_shape, "Wrong shape: %s, %s" % (sequence.shape, desired_shape)
    actions =  buffer.actions[indexes]
    return torch.stack(batch), torch.squeeze(actions, 1)


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def build_mlp_hidden_layers(input_dim, hidden_units=[64, 64], hidden_activation=nn.Tanh()):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    return nn.Sequential(*layers)

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1) # seems we need to remove log_std since it is standard Gaussian now
    # gaussian_log_probs = (-0.5 * noises.pow(2)).sum(
    #     dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1) 
    # return gaussian_log_probs - torch.log(
    #     1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True) # the second term is used as action norm penalty
    return gaussian_log_probs


def reparameterize(means, log_stds):
    # randn_like: returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
    noises = torch.randn_like(means) 
    us = means + noises * log_stds.exp() # e ^ (log_std)
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)

def reparameterize_sigmoid(means, log_stds):
    # randn_like: returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
    means = means.view(-1)
    noises = torch.randn_like(means) 
    us = means + noises * log_stds.exp() # e ^ (log_std)
    actions = torch.sigmoid(us)
    return actions, calculate_log_pi(log_stds, noises, actions)

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def asigmoid(x):
    return torch.log((x + 1e-8) / (1 + 1e-8 - x))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def evaluate_lop_pi_para(means, log_stds, actions):
    noises = (asigmoid(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def idx_to_onehot(option_idx, option_num):
    option_idx = option_idx.type(torch.long)
    batch_size = option_idx.shape[0]
    onehot = torch.zeros(batch_size, option_num)
    # onehot[0:batch_size,option_idx] = 1
    for i in range(batch_size):
        onehot[i,option_idx[i]] = 1
    return onehot

def para_agents(paras_gt, paras):
    batch_agent_para = torch.zeros_like(paras)
    for ba in range(paras.shape[0]):
        for ag in range(paras.shape[1]):
            for pa in range(paras.shape[2]):
                batch_agent_para[ba, ag, :] = paras_gt[ba, [ag, -2, -1]]
    return batch_agent_para



def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
