# tbLogging.py
#
# All TensorBoard logging during training lives here. train.py builds one
# PeriodicEvaluator per run and hands it to whichever trainer runs (trainBC,
# trainCPL, trainCPLpAOT, trainCPLuAOT). Each trainer calls
#   evaluator.logLoss(name, loss, gStep)   every step
#   evaluator.maybeLog(policy, gStep)      every step (no-op off-interval)
# so the trainers decide only WHEN to log; WHAT is logged and under which tag is
# decided here.
#
# TensorBoard tags (the chart title is the tag; the prefix before '/' is the section):
#   train/bcLoss, train/contrastiveLoss            every step
#   pairAcc/{train,test}                           every eval_interval steps
#   paotLoss/{train,test}, paotViolRate/{train,test}
#   uaotLoss/{train,test}, uaotViolRate/{train,test}
#   logLik/{prefTrain,rejTrain,prefTest,rejTest}
#   rollout/avgReward                              every rollout_interval steps
#
# The metrics mirror the post-training ones in evaluate.py (preferenceAccuracy,
# paotLoss, uaotLoss with π_ref log-ratio scores) but are computed far more cheaply:
# every preference pair is built from a small pool of unique demos (~162 per split),
# so each unique demo is scored once per eval and the per-pair scores are gathered
# from those, instead of re-scoring ~26k segments.

import os, json
import numpy as np
import torch

from algos.cpl_paot import paot_loss
from algos.cpl_uaot import uaot_loss
from plotRollouts import computeStats


# runs one episode under the given policy; returns trajectory dict
def rollout(env, policy):
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


# _buildSplit
# Deduplicate the segments of one preference split into its unique demos.
# Returns flat (obs, action) tensors over all unique demos, the demo id of every
# flat row, each demo's length, and the demo index of every pair's pos / neg segment.
def _buildSplit(prefData, device):
	demoIdx = {}
	demoObs, demoAct = [], []
	posIdx, negIdx = [], []
	for pair in prefData:
		for side, idxList in (('pos', posIdx), ('neg', negIdx)):
			acts = np.asarray(pair[f'{side}_actions'], dtype=np.float32)
			obs  = np.asarray(pair[f'{side}_states'][:len(acts)], dtype=np.float32)
			key  = obs.tobytes() + acts.tobytes()
			if key not in demoIdx:
				demoIdx[key] = len(demoObs)
				demoObs.append(obs)
				demoAct.append(acts)
			idxList.append(demoIdx[key])

	lengths = torch.tensor([len(a) for a in demoAct], device=device)
	return {
		'flatObs': torch.tensor(np.concatenate(demoObs), device=device),
		'flatAct': torch.tensor(np.concatenate(demoAct), device=device),
		'rowDemo': torch.repeat_interleave(torch.arange(len(demoAct), device=device), lengths),
		'lengths': lengths.float(),
		'posIdx':  torch.tensor(posIdx, device=device),
		'negIdx':  torch.tensor(negIdx, device=device),
		'nDemos':  len(demoAct),
	}


# _demoLogpSums
# Σ_t log π(a_t | s_t) for every unique demo in a split, in one forward pass.
def _demoLogpSums(policy, split):
	with torch.no_grad():
		_, logp = policy(split['flatObs'], split['flatAct'])
	sums = torch.zeros(split['nDemos'], device=logp.device)
	return sums.index_add_(0, split['rowDemo'], logp)


# _violationRate
# fraction of quantiles where the sorted u fails to dominate the sorted v
# (same definition as the nViolations / nPairs in evaluate.paotLoss and evaluate.uaotLoss)
def _violationRate(u, v):
	return float((torch.sort(v).values > torch.sort(u).values).float().mean().item())


class PeriodicEvaluator:

	def __init__(self, writer, refPolicy, prefDataTrain, prefDataTest, env, runDir,
			evalInterval, rolloutInterval, nRollouts, alpha=0.1, gamma=1.0, device='cpu'):
		# segment scores are α·Σ log π, so per-step log likelihood = score / (α·T);
		# that identity (and the cheap per-demo summing) only holds for γ = 1
		assert gamma == 1.0, 'PeriodicEvaluator assumes γ = 1 (undiscounted segment scores)'
		self.writer          = writer
		self.env             = env
		self.evalInterval    = evalInterval
		self.rolloutInterval = rolloutInterval
		self.nRollouts       = nRollouts
		self.alpha           = alpha
		self.rolloutPath     = os.path.join(runDir, 'training_rollouts.jsonl')

		# per split: unique demos, plus frozen π_ref pos / neg scores (π_ref never changes)
		refPolicy.eval()
		self.splits = {}
		for name, prefData in (('train', prefDataTrain), ('test', prefDataTest)):
			split = _buildSplit(prefData, device)
			refSums = _demoLogpSums(refPolicy, split)
			split['posRef'] = alpha * refSums[split['posIdx']]
			split['negRef'] = alpha * refSums[split['negIdx']]
			split['totalPosSteps'] = split['lengths'][split['posIdx']].sum()
			split['totalNegSteps'] = split['lengths'][split['negIdx']].sum()
			self.splits[name] = split
			print(f'evaluator: {name} split has {len(prefData)} pairs over {split["nDemos"]} unique demos')

	# per-step training loss; name is 'bcLoss' or 'contrastiveLoss'
	def logLoss(self, name, loss, step):
		self.writer.add_scalar(f'train/{name}', loss, step)

	# called every training step; runs the metric eval and / or rollouts on their intervals
	def maybeLog(self, policy, step):
		doEval    = self.evalInterval > 0 and step % self.evalInterval == 0
		doRollout = self.rolloutInterval > 0 and step % self.rolloutInterval == 0
		if not (doEval or doRollout):
			return
		policy.eval()
		if doEval:
			self.logMetrics(policy, step)
		if doRollout:
			self.logRollouts(policy, step)
		policy.train()

	# eval-table metrics + preferred / rejected log likelihood on both splits
	def logMetrics(self, policy, step):
		for name, split in self.splits.items():
			demoSums = _demoLogpSums(policy, split)
			posLogp  = demoSums[split['posIdx']]   # Σ_t log π over each pair's σ+
			negLogp  = demoSums[split['negIdx']]   # Σ_t log π over each pair's σ-
			pos, neg = self.alpha * posLogp, self.alpha * negLogp

			# accuracy on the original annotator pairs
			pairAcc = (pos > neg).float().mean().item()

			# pAOT: policy margins vs frozen π_ref margins
			u, v = pos - neg, split['posRef'] - split['negRef']
			paot, _ = paot_loss(u, v)
			paotViol = _violationRate(u, v)

			# uAOT: pooled log-ratio preferred vs rejected scores
			u, v = pos - split['posRef'], neg - split['negRef']
			uaot, _ = uaot_loss(u, v)
			uaotViol = _violationRate(u, v)

			# average per-action log likelihood over the preferred and rejected sets
			logLikPref = (posLogp.sum() / split['totalPosSteps']).item()
			logLikRej  = (negLogp.sum() / split['totalNegSteps']).item()

			suffix = name.capitalize()
			self.writer.add_scalar(f'pairAcc/{name}',      pairAcc,     step)
			self.writer.add_scalar(f'paotLoss/{name}',     paot.item(), step)
			self.writer.add_scalar(f'paotViolRate/{name}', paotViol,    step)
			self.writer.add_scalar(f'uaotLoss/{name}',     uaot.item(), step)
			self.writer.add_scalar(f'uaotViolRate/{name}', uaotViol,    step)
			self.writer.add_scalar(f'logLik/pref{suffix}', logLikPref,  step)
			self.writer.add_scalar(f'logLik/rej{suffix}',  logLikRej,   step)

	# nRollouts episodes; per-rollout stats appended to training_rollouts.jsonl, average to TB
	def logRollouts(self, policy, step):
		rows = []
		for i in range(self.nRollouts):
			traj = rollout(self.env, policy)
			numSteps, obsSteps, cumReward = computeStats(traj['states'], traj['actions'])
			rows.append({'step': step, 'rollout': i, 'reward': float(cumReward),
				'steps': int(numSteps), 'obs_steps': int(obsSteps)})
		with open(self.rolloutPath, 'a') as f:
			for row in rows:
				f.write(json.dumps(row) + '\n')
		self.writer.add_scalar('rollout/avgReward', np.mean([r['reward'] for r in rows]), step)
