import os, pickle, argparse, json
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from env.pointbot import PointBot
from mlp import MLPGaussianActor
from algos.bc import trainBC
from algos.cpl_paot import trainCPLpAOT
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
	}


def getRunDir(modelsDir, methodName):
	# returns the next available models/<METHOD>_N path
	tag = methodName.upper()
	n = 1
	while os.path.exists(os.path.join(modelsDir, f'{tag}_{n}')):
		n += 1
	runDir = os.path.join(modelsDir, f'{tag}_{n}')
	os.makedirs(runDir)
	return runDir


def saveResults(runDir, policy, evalStats, rollouts, methodName):
	# policy weights
	torch.save(policy.state_dict(), os.path.join(runDir, 'policy.pt'))

	# per-rollout evaluation stats
	with open(os.path.join(runDir, 'eval_stats.json'), 'w') as f:
		json.dump(evalStats, f, indent=2)

	# rollout trajectory plot
	fig, ax = plt.subplots(figsize=(9, 9))
	plotDemos(ax, rollouts, f'{methodName.upper()} Rollouts')
	plt.tight_layout()
	plt.savefig(os.path.join(runDir, 'rollouts.png'), dpi=150, bbox_inches='tight')
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, default='bc',
		choices=['bc', 'cpl', 'cpl_biased', 'cpl_paot', 'cpl_uaot', 'cpl_uaot_ref'])
	args = parser.parse_args()

	scriptDir  = os.path.dirname(os.path.abspath(__file__))
	configPath = os.path.join(scriptDir, 'configs', f'{args.method}.yaml')
	with open(configPath) as f:
		cfg = yaml.safe_load(f)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'method: {args.method}  device: {device}')

	# read in pre-generated preference data
	prefPath = os.path.join(scriptDir, 'data', 'preferences.pkl')
	with open(prefPath, 'rb') as f:
		prefData = pickle.load(f)
	print(f'loaded {len(prefData)} preference pairs')

	# initialize environment
	env    = PointBot()
	obsDim = env.observation_space.shape[0]
	actDim = env.action_space.shape[0]
	policy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)


	# train using pure BC
	if args.method == 'bc':
		policy = trainBC(
			policy, prefData,
			bcSteps     = cfg['bc_steps'],
			batchSize   = cfg['batch_size'],
			lr          = cfg['lr'],
			device      = device,
			logInterval = cfg['log_interval'],
		)

	elif args.method == 'cpl_paot':
		# phase 1: BC-train reference policy π_ref
		refPolicy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)
		print('\n--- phase 1: BC training of π_ref ---')
		refPolicy = trainBC(
			refPolicy, prefData,
			bcSteps     = cfg['ref_bc_steps'],
			batchSize   = cfg['batch_size'],
			lr          = cfg['lr'],
			device      = device,
			logInterval = cfg['log_interval'],
		)
		for param in refPolicy.parameters():
			param.requires_grad = False
		refPolicy.eval()

		# phase 2: BC warmup of π_θ
		if cfg.get('bc_warmup_steps', 0) > 0:
			print('\n--- phase 2: BC warmup of π_θ ---')
			policy = trainBC(
				policy, prefData,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)

		# phase 3: pAOT contrastive training
		print('\n--- phase 3: pAOT contrastive training ---')
		policy = trainCPLpAOT(
			policy, refPolicy, prefData,
			paotSteps   = cfg['cpl_paot_steps'],
			batchSize   = cfg['paot_batch_size'],
			lr          = cfg['lr'],
			alpha       = cfg['alpha'],
			gamma       = cfg['gamma'],
			device      = device,
			logInterval = cfg['log_interval'],
		)

	# TODO train CPL policy

	# TODO train CPL policy with beta regularization

	# TODO train cpl_uaot policy

	else:
		raise NotImplementedError(f'{args.method} is not implemented')

	# generate rollouts
	nRollouts = cfg.get('eval_rollouts', 10)
	rollouts, evalStats = [], []
	print(f'\n--- evaluation ({nRollouts} rollouts) ---')
	for i in range(nRollouts):
		traj = rollout(env, policy)
		numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
		print(f'  rollout {i:>2}  steps: {numSteps}  obs_steps: {obsSteps}  reward: {cumReward:.1f}')
		rollouts.append((f'{args.method}_{i}',
			{'Good_states': traj['states'], 'Good_actions': traj['actions']}))
		evalStats.append({'rollout': i, 'steps': numSteps, 'obs_steps': obsSteps, 'reward': cumReward})

		# TODO evaluate all policy on:
			# Primary:   FSD violation loss (2 functions using uAOT and pAOT pairings respectively)
			# Secondary: preference accuracy on original annotator pairs
			# ? Would it be possible/informative to calculate the stochastic dominance and pareto dominance metrics used in the PSD paper?


	# save results to disk
	modelsDir = os.path.join(scriptDir, 'models')
	runDir    = getRunDir(modelsDir, args.method)
	saveResults(runDir, policy, evalStats, rollouts, args.method)
	print(f'\nresults saved → {runDir}')


main()



