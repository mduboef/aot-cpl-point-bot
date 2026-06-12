import os, pickle, argparse, json, shutil
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
from algos.cpl_uaot import trainCPLuAOT
from plotRollouts import computeStats, plotDemos
from evaluate import preferenceAccuracy, paotLoss, uaotLoss


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


def saveResults(runDir, policy, evalStats, prefStats, rollouts, methodName, configPath, paotStats=None, uaotStats=None):
	# policy weights
	torch.save(policy.state_dict(), os.path.join(runDir, 'policy.pt'))

	# per-rollout stats plus raw-pair preference accuracy on train and test sets, and
	# (when provided) the pAOT / uAOT loss + FSD-violation diagnostics on train and test
	results = {'rollouts': evalStats, 'preferenceAccuracy': prefStats}
	if paotStats is not None:
		results['paotLoss'] = paotStats
	if uaotStats is not None:
		results['uaotLoss'] = uaotStats
	with open(os.path.join(runDir, 'eval_stats.json'), 'w') as f:
		json.dump(results, f, indent=2)

	# copy yaml config file used in training into runDir
	shutil.copy(configPath, os.path.join(runDir, os.path.basename(configPath)))

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

	device = 'cuda' if torch.cuda.is_available() else 'cpu'		# ? wtf is device? Does it specify the hardware type we will run on?
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


	# set up the run directory and tensorboard writer up front so training curves are written live
	# the policy, plots, and eval json are saved into the same dir at the end
	modelsDir = os.path.join(scriptDir, 'models')
	runDir    = getRunDir(modelsDir, args.method)
	writer    = SummaryWriter(os.path.join(runDir, 'tb'))
	print(f'tensorboard logdir → {os.path.join(runDir, "tb")}')

	# number of rollouts generated after training
	nRollouts = 25
	# ? what do these params control
		# supposed used to convert advantage function in cpl-based policies to "preference scores", not sure what that means though
	alphaEval = cfg.get('alpha', 0.1)
	gammaEval = cfg.get('gamma', 1.0)



	# establish reference policy, trained once and cached in models/REF_POLICY/
	refPolicyDir  = os.path.join(modelsDir, 'REF_POLICY')
	refPolicyPath = os.path.join(refPolicyDir, 'policy.pt')

	# load existing ref policy is one exists
	if os.path.isfile(refPolicyPath):
		with open(os.path.join(refPolicyDir, 'config.yaml')) as f:
			refConfig = yaml.safe_load(f)
		if refConfig.get('ref_bc_steps', 0) != cfg.get('ref_bc_steps', 0):
			print(f'WARNING: saved reference policy was trained with ref_bc_steps = {refConfig.get("ref_bc_steps", 0)}, but current config has ref_bc_steps = {cfg.get("ref_bc_steps", 0)}; loading saved ref policy anyway')
		refPolicy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)
		refPolicy.load_state_dict(torch.load(refPolicyPath, map_location=device))
		refPolicy.to(device)
		refPolicy.eval()
		print(f'reference policy loaded from {refPolicyPath}')

	# train and save new ref policy is one doesn't exist
	else:
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
		os.makedirs(refPolicyDir)
		refRollouts, refEvalStats = [], []
		for i in range(nRollouts):
			traj = rollout(env, refPolicy)
			numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
			refRollouts.append((f'ref_{i}', {'Good_states': traj['states'], 'Good_actions': traj['actions']}))
			refEvalStats.append({'rollout': i, 'steps': numSteps, 'obs_steps': obsSteps, 'reward': cumReward})
		refTrainAcc  = preferenceAccuracy(refPolicy, prefDataTrain, alphaEval, gammaEval, device)
		refTestAcc   = preferenceAccuracy(refPolicy, prefDataTest,  alphaEval, gammaEval, device)
		refPrefStats = {'train': refTrainAcc, 'test': refTestAcc}
		saveResults(refPolicyDir, refPolicy, refEvalStats, refPrefStats, refRollouts, 'ref', configPath)
		shutil.copy(configPath, os.path.join(refPolicyDir, 'config.yaml'))
		print(f'Reference policy saved → {refPolicyDir}')



	# train using pure BC
	if  args.method == 'bc':
		print('\n--- Pure BC training of π_θ ---')
		policy = trainBC(
			policy, prefDataTrain,
			bcSteps     = cfg['bc_steps'],
			batchSize   = cfg['batch_size'],
			lr          = cfg['lr'],
			device      = device,
			logInterval = cfg['log_interval'],
		)


	else:
		# phase 1: BC warmup of π_θ
		if cfg.get('bc_warmup_steps', 0) > 0:
			print('\n--- BC warmup of π_θ ---')
			policy = trainBC(
				policy, prefDataTrain,
				bcSteps     = cfg['bc_warmup_steps'],
				batchSize   = cfg['batch_size'],
				lr          = cfg['lr'],
				device      = device,
				logInterval = cfg['log_interval'],
			)
		

		# phase 2: preference learning kicks in

		# baseline CPL (λ = 1.0)
		if args.method == 'cpl':
			print('\n--- CPL on π_θ ---')
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


		# biased CPL (λ = 0.5)
		elif args.method == 'cpl_biased':
			print('\n---CPL (λ = 0.5) on π_θ ---')
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

		# CPL pAOT
		elif args.method == 'cpl_paot':
			print('\n--- CPL pAOT on π_θ ---')
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


		# CPL uAOT (no reference policy → raw scores; cpl_uaot_ref → log-ratio scores)
		elif args.method in ('cpl_uaot', 'cpl_uaot_ref'):
			useRef = args.method == 'cpl_uaot_ref'
			print(f'\n--- CPL uAOT{" (ref)" if useRef else ""} on π_θ ---')
			policy = trainCPLuAOT(
				policy, prefDataTrain,
				refPolicy   = refPolicy if useRef else None,
				uaotSteps   = cfg['cpl_uaot_steps'],
				batchSize   = cfg['uaot_batch_size'],
				lr          = cfg['lr'],
				alpha       = cfg['alpha'],
				gamma       = cfg['gamma'],
				device      = device,
				logInterval = cfg['log_interval'],
				writer       = writer,
				prefDataTest = prefDataTest,
				evalInterval = cfg.get('eval_interval', cfg['log_interval']),
			)


		else:
			raise NotImplementedError(f'{args.method} is not implemented')



	# generate rollouts
	rollouts, evalStats = [], []
	for i in range(nRollouts):
		traj = rollout(env, policy)
		numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
		rollouts.append((str(i + 1),
			{'Good_states': traj['states'], 'Good_actions': traj['actions']}))
		evalStats.append({'rollout': i, 'steps': numSteps, 'obs_steps': obsSteps, 'reward': cumReward})


	# secondary metric: preference accuracy on the ORIGINAL annotator pairs (not OT pairings)
	# evaluated on both the training pairs and the unseen test pairs. broken out per
	# strategy type so we can see if the policy collapses on minority strategies (types 4, 5).
	# alpha/gamma fall back to the CPL defaults for methods (e.g. bc) whose config omits them.
	print('\n--- preference accuracy on raw (non-OT) pairs ---')
	trainAcc = preferenceAccuracy(policy, prefDataTrain, alphaEval, gammaEval, device)
	testAcc  = preferenceAccuracy(policy, prefDataTest,  alphaEval, gammaEval, device)
	print(f'  train: {trainAcc["overall"]:.3f} overall  ({trainAcc["nPairs"]} pairs)')
	print(f'  test:  {testAcc["overall"]:.3f} overall  ({testAcc["nPairs"]} pairs)')
	prefStats = {'train': trainAcc, 'test': testAcc}


	# primary metric 1: pAOT loss of the trained policy (any method) against π_ref
	# measures first-order stochastic dominance violations
	print('\n--- pAOT loss vs π_ref (FSD violations) ---')
	paotTrain = paotLoss(policy, refPolicy, prefDataTrain, alphaEval, gammaEval, device)
	paotTest  = paotLoss(policy, refPolicy, prefDataTest,  alphaEval, gammaEval, device)
	for label, p in [('train', paotTrain), ('test', paotTest)]:
		print(f'  {label}: paot_loss {p["paotLoss"]:.4f}  '
			f'violations {p["nViolations"]}/{p["nPairs"]} ({p["violationFreq"]:.1%})  '
			f'shortfall mean {p["meanShortfall"]:.3f} max {p["maxShortfall"]:.3f}')
	paotStats = {'train': paotTrain, 'test': paotTest}


	# primary metric 2: uAOT loss of the trained policy (any method), measuring first-order
	# stochastic dominance violations between the pooled preferred- and rejected-score
	# distributions. computed with π_ref (log-ratio scores) so the metric is a common,
	# ease-of-imitation-normalized yardstick across every method, regardless of how it trained.
	print('\n--- uAOT loss vs π_ref (FSD violations) ---')
	uaotTrain = uaotLoss(policy, prefDataTrain, refPolicy, alphaEval, gammaEval, device)
	uaotTest  = uaotLoss(policy, prefDataTest,  refPolicy, alphaEval, gammaEval, device)
	for label, p in [('train', uaotTrain), ('test', uaotTest)]:
		print(f'  {label}: uaot_loss {p["uaotLoss"]:.4f}  '
			f'violations {p["nViolations"]}/{p["nPairs"]} ({p["violationFreq"]:.1%})  '
			f'shortfall mean {p["meanShortfall"]:.3f} max {p["maxShortfall"]:.3f}')
	uaotStats = {'train': uaotTrain, 'test': uaotTest}

	# ? Consider using the stochastic / Pareto dominance evaluation metrics from the PSD paper.

	# save results to disk (into the same runDir as the tensorboard logs)
	saveResults(runDir, policy, evalStats, prefStats, rollouts, args.method, configPath, paotStats, uaotStats)
	writer.close()
	print(f'\nresults saved → {runDir}')


main()



