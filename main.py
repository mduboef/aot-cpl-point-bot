import os, pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from env.pointbot import PointBot
from mlp import MLPGaussianActor
from plotRollouts import computeStats, plotDemos

def rollout(env, policy):
	# runs one episode under the given policy; returns trajectory dict
	obs = env.reset()
	states, actions = [obs.copy()], []
	done = False
	while not done:
		action, _ = policy.step(obs)
		obs, _, done, _ = env.step(action)
		actions.append(action)
		states.append(obs.copy())
	return {
		'states':  np.array(states),
		'actions': np.array(actions),
		'feature': list(env.feature),
	}


def main():

	# read in pre-generated preference data
	prefDataPath = 'data/preferences.pkl'
	with open(prefDataPath, 'rb') as f:
		prefData = pickle.load(f)
	print(f'loaded {len(prefData)} preference pairs')

	# initialize environment
	env = PointBot()

	# initialize policy network (random weights)
	obsDim = env.observation_space.shape[0]
	actDim = env.action_space.shape[0]
	policy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)

	# rollout random policy, print stats, and collect for plotting
	rollouts = []
	for i in range(10):
		traj = rollout(env, policy)
		numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
		print(f'random policy rollout {i} — steps: {numSteps}  obs_steps: {obsSteps}  reward: {cumReward:.1f}')
		rollouts.append((f'rollout_{i}.pkl', {'Good_states': traj['states'], 'Good_actions': traj['actions']}))

	# plot all 10 rollouts overlaid on the obstacle map
	fig, ax = plt.subplots(figsize=(9, 9))
	plotDemos(ax, rollouts, 'Random Policy Rollouts')
	plt.tight_layout()
	savePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rollouts_random.png')
	plt.savefig(savePath, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print(f'\nfigure saved to {savePath}')

	# TODO behavior cloning on raw state-action for trajectories that appear in the preference data
	

	# TODO plot demos and BC rollouts

	# TODO train CPL policy

	# TODO train CPL policy with beta regularization of ___

	# TODO create pAOT pairings

	# TODO train cpl_paot policy

	# TODO create uAOT pairings

	# TODO train cpl_uaot policy

	# TODO evaluate all policies (BC, CPL, CPL_biased, cpl_paot, cpl_uaot)
		# Primary:   FSD violation loss (uAOT and pAOT pairings)
		# Secondary: preference accuracy on original annotator pairs
		# Tertiary:  ground-truth cumulative reward


main()