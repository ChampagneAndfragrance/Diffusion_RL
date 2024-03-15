import torch

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

def log_prob(distribution, actions: torch.Tensor) -> torch.Tensor:
    """
    Get the log probabilities of actions according to the distribution.
    :param distribution: (Torch.distributions type) Calculate prob w.r.t distribution
    :param actions: Actions whose probability is computed
    :return:
    """
    logprob = distribution.log_prob(actions)
    return sum_independent_dims(logprob)