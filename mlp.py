# holds code for the policy network

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):

	def _distribution(self, obs):
		raise NotImplementedError

	def _log_prob_from_distribution(self, pi, act):
		raise NotImplementedError

	def forward(self, obs, act=None):
		pi = self._distribution(obs)
		logp_a = None
		if act is not None:
			logp_a = self._log_prob_from_distribution(pi, act)
		return pi, logp_a

	def step(self, obs):
		# obs: numpy (obs_dim,) → sample action and return (action, log_prob) as numpy
		with torch.no_grad():
			pi = self._distribution(torch.as_tensor(obs, dtype=torch.float32))
			action = pi.sample()
			logp = self._log_prob_from_distribution(pi, action)
		return action.numpy(), logp.numpy()


class MLPGaussianActor(Actor):

	def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
		super().__init__()
		log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
		self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		mu = self.mu_net(obs)
		std = torch.exp(self.log_std)
		return Normal(mu, std)

	def _log_prob_from_distribution(self, pi, act):
		# sum over action dims to get a scalar log prob per timestep
		return pi.log_prob(act).sum(axis=-1)