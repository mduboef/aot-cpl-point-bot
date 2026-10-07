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
from algos.ref import prepareRefDir, loadRefPolicy
from plotRollouts import computeStats, plotDemos
from evaluate import preferenceAccuracy, paotLoss, uaotLoss
from tbLogging import PeriodicEvaluator, rollout


# pretty-prints overall and per-strategy-type preference accuracy
def printAccuracy(label, acc):
	print(f'  {label}: {acc["overall"]:.3f} overall  ({acc["nPairs"]} pairs)')


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
		choices=['ref', 'bc', 'cpl', 'cpl_biased', 'cpl_paot', 'cpl_uaot', 'cpl_uaot_ref'])
	args = parser.parse_args()

	scriptDir  = os.path.dirname(os.path.abspath(__file__))
	configPath = os.path.join(scriptDir, 'configs', f'{args.method}.yaml')
	with open(configPath) as f:
		cfg = yaml.safe_load(f)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'		# ? wtf is device? Does it specify the hardware type we will run on?
	print(f'method: {args.method}  device: {device}')


	# read in training set of preference pairs
	prefPathTrain = os.path.join(scriptDir, 'data', 'trainingData', 'trainPreferences.pkl')
	with open(prefPathTrain, 'rb') as f:
		prefDataTrain = pickle.load(f)
	print(f'loaded {len(prefDataTrain)} preference pairs')

	# read in testing set of preference pairs
	prefPathTest = os.path.join(scriptDir, 'data', 'testingData', 'testPreferences.pkl')
	with open(prefPathTest, 'rb') as f:
		prefDataTest = pickle.load(f)
	print(f'loaded {len(prefDataTest)} preference pairs')


	# initialize environment
	env    = PointBot()
	obsDim = env.observation_space.shape[0]
	actDim = env.action_space.shape[0]
	policy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)


	# number of rollouts generated after training (for plotting and final avg reward)
	nRollouts = 50

	# ? what do these params control
		# supposed used to convert advantage function in cpl-based policies to "preference scores", not sure what that means though
	alphaEval = cfg.get('alpha', 0.1)
	gammaEval = cfg.get('gamma', 1.0)


	# reference policy lives in models/REF_POLICY/ and is only (re)trained by --method ref
	# every other method loads it, erroring out before a run dir is created if it's missing
	modelsDir    = os.path.join(scriptDir, 'models')
	refPolicyDir = os.path.join(modelsDir, 'REF_POLICY')
	if args.method == 'ref':
		refPolicy = None
		runDir    = prepareRefDir(refPolicyDir)
	else:
		refPolicy = loadRefPolicy(refPolicyDir, obsDim, actDim, device)
		runDir    = getRunDir(modelsDir, args.method)


	# TensorBoard evaluator: logs the eval-table metrics every eval_interval steps and the
	# average reward of n_eval_rollouts rollouts every rollout_interval steps (see tbLogging.py)
	# its metrics are measured against π_ref, so the ref run itself trains without one
	writer    = None
	evaluator = None
	if refPolicy is not None:
		writer = SummaryWriter(os.path.join(runDir, 'tb'))
		print(f'tensorboard logdir → {os.path.join(runDir, "tb")}')
		evaluator = PeriodicEvaluator(
			writer, refPolicy, prefDataTrain, prefDataTest,
			env             = PointBot(),
			runDir          = runDir,
			evalInterval    = cfg['eval_interval'],
			rolloutInterval = cfg['rollout_interval'],
			nRollouts       = cfg['n_eval_rollouts'],
			alpha           = alphaEval,
			gamma           = gammaEval,
			device          = device,
		)
		# step 0: log the untrained policy so every curve starts from initialization
		evaluator.maybeLog(policy.to(device), 0)

	# step offset so a BC warmup phase is plotted before the preference-learning phase
	warmupSteps = cfg.get('bc_warmup_steps', 0)

	# train using pure BC (π_ref is always pure BC)
	if args.method in ('bc', 'ref'):
		print(f'\n--- Pure BC training of {"π_ref" if args.method == "ref" else "π_θ"} ---')
		policy = trainBC(
			policy, prefDataTrain,
			bcSteps      = cfg['bc_steps'],
			batchSize    = cfg['batch_size'],
			lr           = cfg['lr'],
			device       = device,
			logInterval  = cfg['log_interval'],
			evaluator    = evaluator,
		)


	else:
		# phase 1: BC warmup of π_θ
		if warmupSteps > 0:
			print('\n--- BC warmup of π_θ ---')
			policy = trainBC(
				policy, prefDataTrain,
				bcSteps      = warmupSteps,
				batchSize    = cfg['batch_size'],
				lr           = cfg['lr'],
				device       = device,
				logInterval  = cfg['log_interval'],
				evaluator    = evaluator,
			)
		

		# phase 2: preference learning kicks in

		# CPL (λ = 1.0)
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
				evaluator   = evaluator,
				stepOffset  = warmupSteps,
			)


		# biased CPL (λ < 1.0)
		elif args.method == 'cpl_biased':
			# get λ from config
			biasReg = cfg['contrastive_bias']
			print(f'\n---CPL (λ = {biasReg}) on π_θ ---')
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
				evaluator   = evaluator,
				stepOffset  = warmupSteps,
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
				evaluator   = evaluator,
				stepOffset  = warmupSteps,
			)


		# CPL uAOT (2 versions)
			# cpl_uaot → raw scores
			# cpl_uaot_ref → log-ratio scores using ref policy
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
				evaluator   = evaluator,
				stepOffset  = warmupSteps,
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


	# the AOT metrics below are measured against π_ref, so they're skipped for the ref run itself
	paotStats, uaotStats = None, None
	if refPolicy is not None:

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
	if writer is not None:
		writer.close()
	print(f'\nresults saved → {runDir}')


main()



