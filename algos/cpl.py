# algos/cpl.py
#
# Baseline Contrastive Preference Learning (CPL).
# Reference: Hejna et al. 2024 (CPL) and the "Baseline Methods" section of
# finalReport.tex. Implementation mirrors biased_bce_with_logits from
# aot-cpl/research/algs/cpl.py, specialized to our always-ordered pairs.
#
# This file implements phase 3 only (CPL contrastive training of π_θ).
# Phase 2 (BC warmup of π_θ) is handled by trainBC() from algos/bc.py, called
# from train.py. Unlike pAOT, baseline CPL uses no reference policy.
#
# Loss for a single preference pair (σ+ preferred, σ- rejected):
#   L = -log σ( score(σ+) - λ · score(σ-) )
# where score(σ; π) = Σ_t γ^t α log π(a_t | s_t)   [finalReport.tex step 3]
# λ (contrastiveBias) = 1.0 is unbiased CPL; λ = 0.5 is the conservative
# (biased) variant from the CPL paper. The asymmetric λ < 1 discounts the
# rejected score, preventing the policy from driving neg-action log-probs
# down without bound (the degeneracy that corrupts absolute action density).
#
# Data format (preference pairs from data/preferenceGen.py):
#   pos_states:  np array (T+1, obs_dim)
#   pos_actions: np array (T,   act_dim)
#   neg_states:  np array (T+1, obs_dim)
#   neg_actions: np array (T,   act_dim)
#   (trajectory lengths T vary across pairs)

import numpy as np
import torch

from evaluate import preferenceAccuracy


# cpl_loss
# Standard CPL cross-entropy loss on a batch of preference pairs. Because every
# pair is ordered with σ+ preferred, the BCE label is fixed and the loss reduces
# to -log σ(posScores - bias·negScores).
#
# posScores: per-pair scores of preferred segments,  shape (n,)
# negScores: per-pair scores of rejected segments,    shape (n,)
def cpl_loss(posScores, negScores, bias=1.0):
	logit = posScores - bias * negScores
	# numerically stable: -log sigmoid(logit) = log(1 + exp(-logit))
	maxVal = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-maxVal) + torch.exp(-logit - maxVal)) + maxVal).mean()

	with torch.no_grad():
		# accuracy uses the UNBIASED margin: does the policy actually prefer σ+?
		accuracy = (posScores > negScores).float().mean()

	return loss, accuracy


# _cachePrefTensors
# Convert all numpy arrays in prefData to float32 tensors once before training.
# Avoids repeated numpy→tensor conversions inside the training loop. Each entry
# stores only the T obs/act pairs needed for scoring (not the T+1 states).
def _cachePrefTensors(prefData, device):
	cache = []
	for pair in prefData:
		tPos = len(pair['pos_actions'])
		tNeg = len(pair['neg_actions'])
		cache.append({
			'posS': torch.tensor(pair['pos_states'][:tPos], dtype=torch.float32, device=device),
			'posA': torch.tensor(pair['pos_actions'],        dtype=torch.float32, device=device),
			'negS': torch.tensor(pair['neg_states'][:tNeg],  dtype=torch.float32, device=device),
			'negA': torch.tensor(pair['neg_actions'],         dtype=torch.float32, device=device),
		})
	return cache


# _batchSegmentScores
# Compute per-segment scores for a batch of segments using a single forward pass.
# All (state, action) pairs from all segments are concatenated into one flat
# batch, scored in one forward pass, then split back by segment and summed.
#
# score(σ; π) = Σ_t γ^t α log π(a_t | s_t)   [finalReport.tex step 3]
# (γ=1.0 by default, matching CPL paper convention)
#
# Returns a list of scalar tensors (one per segment), differentiable w.r.t. policy.
def _batchSegmentScores(policy, segObs, segAct, alpha, gamma):
	lengths = [a.shape[0] for a in segAct]

	flatObs = torch.cat(segObs, dim=0)  # (Σ T_i, obs_dim)
	flatAct = torch.cat(segAct, dim=0)  # (Σ T_i, act_dim)
	_, flatLogp = policy(flatObs, flatAct)  # (Σ T_i,)

	scores = []
	offset = 0
	for T in lengths:
		segLogp = flatLogp[offset:offset + T]
		if gamma == 1.0:
			scores.append(alpha * segLogp.sum())
		else:
			discounts = gamma ** torch.arange(T, dtype=torch.float32, device=flatLogp.device)
			scores.append((alpha * discounts * segLogp).sum())
		offset += T

	return scores


# _batchPosNegScores
# Compute posScores and negScores for a batch of pairs in one batched forward
# pass over all 2*batchSize segments. Differentiable w.r.t. policy parameters.
def _batchPosNegScores(policy, cache, indices, alpha, gamma):
	segObs, segAct = [], []
	for idx in indices:
		entry = cache[idx]
		segObs += [entry['posS'], entry['negS']]
		segAct += [entry['posA'], entry['negA']]

	scores = _batchSegmentScores(policy, segObs, segAct, alpha, gamma)
	posScores = torch.stack(scores[0::2])
	negScores = torch.stack(scores[1::2])
	return posScores, negScores


# _logRawAccuracy
# Evaluate and log overall + per-type preference accuracy on a raw preference set.
# Leaves the policy back in train mode (preferenceAccuracy flips it to eval) so the
# training loop can continue uninterrupted.
def _logRawAccuracy(writer, tag, policy, prefData, alpha, gamma, device, step):
	acc = preferenceAccuracy(policy, prefData, alpha, gamma, device)
	policy.train()
	writer.add_scalar(f'accuracy/{tag}', acc['overall'], step)
	for st, d in acc['perType'].items():
		writer.add_scalar(f'{tag}PerType/type_{st}', d['accuracy'], step)
	return acc['overall']


# trainCPL
# Phase 3: CPL contrastive training of π_θ (no reference policy).
#
# π_θ (policy) should have its BC warmup done before calling this.
#
# Key steps per training iteration:
#   1. Sample a random mini-batch of batchSize preference pairs.
#   2. Compute per-pair pos/neg scores (differentiable) via one batched forward pass.
#   3. Compute the CPL loss -log σ(pos - bias·neg).
#   4. Backpropagate and update π_θ.
def trainCPL(
	policy,
	prefData,
	cplSteps,
	batchSize,
	lr,
	alpha=0.1,
	gamma=1.0,
	bias=1.0,
	device='cpu',
	logInterval=500,
	writer=None,
	prefDataTest=None,
	evalInterval=500,
):
	n = len(prefData)

	# pre-convert all numpy arrays to device tensors once
	cache = _cachePrefTensors(prefData, device)

	policy = policy.to(device)
	policy.train()
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	for step in range(1, cplSteps + 1):
		indices = np.random.randint(0, n, size=batchSize)

		posScores, negScores = _batchPosNegScores(policy, cache, indices, alpha, gamma)
		loss, accuracy = cpl_loss(posScores, negScores, bias=bias)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# per-step curves: CPL loss and preference accuracy on the sampled batch
		if writer is not None:
			writer.add_scalar('loss/cpl', loss.item(), step)
			writer.add_scalar('accuracy/batch', accuracy.item(), step)

		# periodic curves: raw preference accuracy on the full train and test sets
		if writer is not None and step % evalInterval == 0:
			_logRawAccuracy(writer, 'rawTrain', policy, prefData, alpha, gamma, device, step)
			if prefDataTest is not None:
				_logRawAccuracy(writer, 'rawTest', policy, prefDataTest, alpha, gamma, device, step)

		if step % logInterval == 0:
			print(f'  step {step:>6}/{cplSteps}  cpl_loss: {loss.item():.4f}  accuracy: {accuracy.item():.3f}')

	policy.eval()
	return policy
