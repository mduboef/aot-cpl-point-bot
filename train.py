import os, pickle, argparse, json
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from env.pointbot import PointBot
from mlp import MLPGaussianActor
from algos.bc import trainBC
from algos.cpl import trainCPL
from algos.cpl_paot import trainCPLpAOT
from plotRollouts import computeStats, plotDemos
from evaluate import preferenceAccuracy


# pretty-prints overall and per-strategy-type preference accuracy
def printAccuracy(label, acc):
	print(f'  {label}: {acc["overall"]:.3f} overall  ({acc["nPairs"]} pairs)')


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


def saveResults(runDir, policy, evalStats, prefStats, rollouts, methodName):
	# policy weights
	torch.save(policy.state_dict(), os.path.join(runDir, 'policy.pt'))

	# per-rollout stats plus raw-pair preference accuracy on train and test sets
	with open(os.path.join(runDir, 'eval_stats.json'), 'w') as f:
		json.dump({'rollouts': evalStats, 'preferenceAccuracy': prefStats}, f, indent=2)

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


	# read in training set of preference pairs
	prefPathTrain = os.path.join(scriptDir, 'data', 'trainPreferences.pkl')
	with open(prefPathTrain, 'rb') as f:
		prefDataTrain = pickle.load(f)
	print(f'loaded {len(prefDataTrain)} preference pairs')

	# read in testing set of preference pairs
	prefPathTest = os.path.join(scriptDir, 'data', 'testPreferences.pkl')
	with open(prefPathTest, 'rb') as f:
		prefDataTest = pickle.load(f)
	print(f'loaded {len(prefDataTest)} preference pairs')


	# initialize environment
	env    = PointBot()
	obsDim = env.observation_space.shape[0]
	actDim = env.action_space.shape[0]
	policy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)


	# set up the run directory and tensorboard writer up front so training curves are
	# written live; the policy, plots, and eval json are saved into the same dir at the end
	modelsDir = os.path.join(scriptDir, 'models')
	runDir    = getRunDir(modelsDir, args.method)
	writer    = SummaryWriter(os.path.join(runDir, 'tb'))
	print(f'tensorboard logdir → {os.path.join(runDir, "tb")}')

	# train using pure BC
	if args.method == 'bc':
		policy = trainBC(
			policy, prefDataTrain,
			bcSteps     = cfg['bc_steps'],
			batchSize   = cfg['batch_size'],
			lr          = cfg['lr'],
			device      = device,
			logInterval = cfg['log_interval'],
		)


	# train baseline CPL policy (no reference policy, λ = contrastive_bias)
	elif args.method == 'cpl':
		# phase 2: BC warmup of π_θ
		if cfg.get('bc_warmup_steps', 0) > 0:
			print('\n--- phase 2: BC warmup of π_θ ---')
			policy = trainBC(
				policy, prefDataTrain,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)

		# phase 3: CPL contrastive training
		print('\n--- phase 3: CPL contrastive training ---')
		policy = trainCPL(
			policy, prefDataTrain,
			cplSteps    = cfg['cpl_steps'],
			batchSize   = cfg['cpl_batch_size'],
			lr          = cfg['lr'],
			alpha       = cfg['alpha'],
			gamma       = cfg['gamma'],
			bias        = cfg['contrastive_bias'],
			device      = device,
			logInterval = cfg['log_interval'],
			writer       = writer,
			prefDataTest = prefDataTest,
			evalInterval = cfg.get('eval_interval', cfg['log_interval']),
		)


	# train conservative CPL policy (no reference policy, λ = contrastive_bias = 0.5)
	# identical pipeline to the cpl branch; the only difference is contrastive_bias in the config
	elif args.method == 'cpl_biased':
		# phase 2: BC warmup of π_θ
		if cfg.get('bc_warmup_steps', 0) > 0:
			print('\n--- phase 2: BC warmup of π_θ ---')
			policy = trainBC(
				policy, prefDataTrain,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)

		# phase 3: CPL contrastive training with λ = 0.5
		print('\n--- phase 3: CPL contrastive training (λ = 0.5) ---')
		policy = trainCPL(
			policy, prefDataTrain,
			cplSteps    = cfg['cpl_steps'],
			batchSize   = cfg['cpl_batch_size'],
			lr          = cfg['lr'],
			alpha       = cfg['alpha'],
			gamma       = cfg['gamma'],
			bias        = cfg['contrastive_bias'],
			device      = device,
			logInterval = cfg['log_interval'],
			writer       = writer,
			prefDataTest = prefDataTest,
			evalInterval = cfg.get('eval_interval', cfg['log_interval']),
		)


	# ! CPL_pAOT SEEMS TO BE BUGGED! THE ROLLOUTS ALL GO WAY OFF COURSE IN THE SAME DIRECTION
	elif args.method == 'cpl_paot':
		# phase 1: BC-train reference policy π_ref
		refPolicy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)
		print('\n--- phase 1: BC training of π_ref ---')
		refPolicy = trainBC(
			refPolicy, prefDataTrain,
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
				policy, prefDataTrain,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)

		# phase 3: cpl_pAOT preference training
		print('\n--- phase 3: pAOT contrastive training ---')
		policy = trainCPLpAOT(
			policy, refPolicy, prefDataTrain,
			paotSteps   = cfg['cpl_paot_steps'],
			batchSize   = cfg['paot_batch_size'],
			lr          = cfg['lr'],
			alpha       = cfg['alpha'],
			gamma       = cfg['gamma'],
			device      = device,
			logInterval = cfg['log_interval'],
			writer       = writer,
			prefDataTest = prefDataTest,
			evalInterval = cfg.get('eval_interval', cfg['log_interval']),
		)


	# TODO train cpl_uaot policy
	elif args.method == 'cpl_uaot':
		# phase 1: BC-train reference policy π_ref
		refPolicy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)
		print('\n--- phase 1: BC training of π_ref ---')
		refPolicy = trainBC(
			refPolicy, prefDataTrain,
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
				policy, prefDataTrain,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)

		# TODO phase 3: cpl_uAOT preference training with π_ref as the reference policy

		raise NotImplementedError(f'{args.method} is not implemented')


	else:
		raise NotImplementedError(f'{args.method} is not implemented')

	# generate rollouts
	nRollouts = 25
	rollouts, evalStats = [], []
	for i in range(nRollouts):
		traj = rollout(env, policy)
		numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
		rollouts.append((f'{args.method}_{i}',
			{'Good_states': traj['states'], 'Good_actions': traj['actions']}))
		evalStats.append({'rollout': i, 'steps': numSteps, 'obs_steps': obsSteps, 'reward': cumReward})


	# secondary metric: preference accuracy on the ORIGINAL annotator pairs (not OT pairings)
	# evaluated on both the training pairs and the unseen test pairs. broken out per
	# strategy type so we can see if the policy collapses on minority strategies (types 4, 5).
	# alpha/gamma fall back to the CPL defaults for methods (e.g. bc) whose config omits them.
	alphaEval = cfg.get('alpha', 0.1)
	gammaEval = cfg.get('gamma', 1.0)

	print('\n--- preference accuracy on raw (non-OT) pairs ---')
	trainAcc = preferenceAccuracy(policy, prefDataTrain, alphaEval, gammaEval, device)
	testAcc  = preferenceAccuracy(policy, prefDataTest,  alphaEval, gammaEval, device)
	print(f'  train: {trainAcc["overall"]:.3f} overall  ({trainAcc["nPairs"]} pairs)')
	print(f'  test:  {testAcc["overall"]:.3f} overall  ({testAcc["nPairs"]} pairs)')
	prefStats = {'train': trainAcc, 'test': testAcc}

	# TODO primary metric: FSD violation loss (pAOT and uAOT pairings).
	# hold off until the cpl_pAOT / cpl_uAOT pairings are confirmed working.
	# ? also consider the stochastic / Pareto dominance metrics from the PSD paper.

	# save results to disk (into the same runDir as the tensorboard logs)
	saveResults(runDir, policy, evalStats, prefStats, rollouts, args.method)
	writer.close()
	print(f'\nresults saved → {runDir}')


main()



