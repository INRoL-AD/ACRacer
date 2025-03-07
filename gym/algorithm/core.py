import math
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import torch.nn.functional as F
from torch.distributions.normal import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x = [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


## Additional utilities for TRPO ###
def flat_grads(grads):
    return torch.cat([grad.contiguous().view(-1) for grad in grads])

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r).to(b.device)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x
####################################
    

class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, log_std_scale=-0.5):
        super().__init__()
        log_std = log_std_scale * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, obs, act=None, deterministic=False):
        if deterministic:
            return self.mu_net(obs), None
        else:
            pi = self._distribution(obs)
            logp_a = None
            if act is not None:
                logp_a = self._log_prob_from_distribution(pi, act)
            return pi, logp_a

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit_min, act_limit_max, **kwargs):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_dim = act_dim
        self.act_limit_min = act_limit_min
        self.act_limit_max = act_limit_max
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, deterministic=False, with_logprob=True, with_logprob_original=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        original_shape = pi_action.shape
        if len(pi_action.shape) == 1:
            pi_action = pi_action.unsqueeze(0)

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # Unified throttle-brake control
        if self.act_dim == 2:
            pi_action = torch.tanh(pi_action)
            # pi_action = torch.clamp(pi_action, self.act_limit_min, self.act_limit_max)  # Clamp if needed

        # Separated throttle-brake control
        elif self.act_dim == 3:
            brake = torch.sigmoid(pi_action[:, 0])
            steer = torch.tanh(pi_action[:, 1])
            throttle = torch.sigmoid(pi_action[:, 2])
            pi_action = torch.cat((brake.unsqueeze(1), steer.unsqueeze(1), throttle.unsqueeze(1)), dim=1)
        
        if original_shape == torch.Size([3]) or original_shape == torch.Size([2]):
            pi_action = pi_action.squeeze(0)
        
        if with_logprob_original:
            return pi_action, logp_pi, pi_distribution.log_prob(pi_action)
        return pi_action, logp_pi


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
    

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, **kwargs):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU, **kwargs):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("You are now using : {}".format(self.device))

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit_min = torch.tensor(action_space.low, dtype=torch.float32).to(self.device)
        act_limit_max = torch.tensor(action_space.high, dtype=torch.float32).to(self.device)
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit_min, act_limit_max)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.to(self.device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
        

class Disc(nn.Module):
    """
    Replace (o,a) with (o,a,log_pi).
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = mlp([obs_dim + 2*act_dim] + list(hidden_sizes) + [1], activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, states, actions, log_pis):
        return self.net(torch.cat([states, actions, log_pis], dim=-1))

    def d(self, states, actions, log_pis):
        with torch.no_grad():
            return torch.sigmoid(self.forward(states, actions, log_pis))