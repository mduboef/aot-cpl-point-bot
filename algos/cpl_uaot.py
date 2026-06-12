# algos/cpl_uaot.py
#
# CPL with Unpaired Alignment via Optimal Transport (uAOT).
# Formalization reference: "CPL with uAOT and No Reference Policy" and
# "CPL with uAOT and a Reference Policy" sections of finalReport.tex.
#
# This file implements phase 3 only (uAOT contrastive training of π_θ).
# Phases 1 and 2 (BC training of π_ref and optional BC warmup of π_θ) are
# handled by trainBC() from algos/bc.py, called from train.py.
#
# Three-phase structure (driven by train.py):
#   Phase 1 — BC-train π_ref with trainBC()  (only used by the reference variant)
#   Phase 2 — BC-warmup π_θ with trainBC()  (optional; skip with bc_warmup_steps=0)
#   Phase 3 — uAOT contrastive training with trainCPLuAOT()  ← this file
#
# How uAOT differs from baseline CPL (cpl.py):
#   Baseline CPL compares each (σ+, σ-) pair directly. uAOT instead pools all
#   preferred-segment scores into one set U and all rejected-segment scores into
#   another set V, sorts each independently, and applies the CPL loss to the
#   i-th ranked preferred score against the i-th ranked rejected score (1-D
#   optimal transport via the northwest corner method). This enforces first-order
#   stochastic dominance of the preferred score distribution over the rejected one.
#
# How uAOT differs from pAOT (cpl_paot.py):
#   pAOT operates on per-pair MARGINS (score(σ+) - score(σ-)) and pushes the
#   policy margin distribution to dominate the frozen reference margin distribution.
#   uAOT operates on individual SEGMENT scores, pooled across the batch, and pushes
#   the preferred score distribution to dominate the rejected score distribution.
#
# Two variants (selected by passing refPolicy or not):
#   no reference (refPolicy=None):
#       score(σ) = Σ_t γ^t α log π_θ(a_t | s_t)                          [raw scores]
#   with reference (refPolicy given):
#       score(σ) = Σ_t γ^t α (log π_θ(a_t | s_t) - log π_ref(a_t | s_t))  [log-ratio]
#   log-ratio scores normalize out ease-of-imitation so that scores are comparable
#   across preference pairs, which matters because uAOT mixes segments from
#   different pairs when it pools and sorts them.
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


# uaot_loss
# Formalization steps 5 and 6 of "CPL with uAOT" in finalReport.tex.
#
# u: pooled preferred-segment scores for the batch, shape (n,)
# v: pooled rejected-segment scores for the batch,  shape (n,)
#
# Step 5 — sort both sets independently (1D optimal transport, northwest corner):
#   u^(1) ≤ u^(2) ≤ ... ≤ u^(n)
#   v^(1) ≤ v^(2) ≤ ... ≤ v^(n)
#   the i-th lowest preferred score is matched with the i-th lowest rejected score.
#
# Step 6 — CPL loss on OT-matched pairs:
#   L(θ) = (1/n) Σ_i -log [ exp(u^(i)) / (exp(u^(i)) + exp(v^(i))) ]
def uaot_loss(u, v):
	u_sorted = torch.sort(u).values
	v_sorted = torch.sort(v).values

	logit = u_sorted - v_sorted
	# numerically stable: -log sigmoid(logit) = log(1 + exp(-logit))
	max_val = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-max_val) + torch.exp(-logit - max_val)) + max_val).mean()

	with torch.no_grad():
		accuracy = (u_sorted > v_sorted).float().mean()

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
# Compute per-segment preferred and rejected scores for a batch of pairs in one
# batched forward pass over all 2*len(indices) segments. The segments are laid out
# interleaved as [pos_0, neg_0, pos_1, neg_1, ...] so the even slice recovers the
# preferred scores and the odd slice the rejected scores. Differentiable w.r.t. policy.
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


# computeRefScores
# Pre-compute the frozen per-segment reference scores for the preferred and rejected
# segment of every preference pair, using a single forward pass over all segments.
# Called once before training begins (only for the reference variant); π_ref is frozen
# so these never change. Returns two float32 CPU tensors of shape (n,) — posRef and
# negRef — for cheap random indexing during the training loop. The log-ratio score of
# a segment is then (policy score) - (reference score), computed per batch.
def computeRefScores(refPolicy, cache, alpha, gamma, device):
	refPolicy.eval()
	segObs, segAct = [], []
	for entry in cache:
		segObs += [entry['posS'], entry['negS']]
		segAct += [entry['posA'], entry['negA']]

	with torch.no_grad():
		scores = _batchSegmentScores(refPolicy, segObs, segAct, alpha, gamma)

	posRef = torch.stack(scores[0::2]).detach().cpu()
	negRef = torch.stack(scores[1::2]).detach().cpu()
	return posRef, negRef  # (n,) CPU each


# _logRawAccuracy
# Evaluate and log overall + per-type preference accuracy on a raw (non-OT)
# preference set. Leaves the policy back in train mode (preferenceAccuracy flips it
# to eval) so the training loop can continue uninterrupted.
def _logRawAccuracy(writer, tag, policy, prefData, alpha, gamma, device, step):
	acc = preferenceAccuracy(policy, prefData, alpha, gamma, device)
	policy.train()
	writer.add_scalar(f'accuracy/{tag}', acc['overall'], step)
	for st, d in acc['perType'].items():
		writer.add_scalar(f'{tag}PerType/type_{st}', d['accuracy'], step)
	return acc['overall']


# trainCPLuAOT
# Phase 3: uAOT contrastive training of π_θ.
#
# π_θ (policy) should have its BC warmup done before calling this.
# refPolicy is optional:
#   refPolicy=None  → no-reference variant, raw segment scores.
#   refPolicy given → reference variant, log-ratio scores; π_ref must be BC-trained
#                     and frozen before calling this.
#
# Key steps per training iteration:
#   1. Sample a random mini-batch of batchSize preference pairs.
#   2. Compute per-segment preferred/rejected scores (differentiable) via one
#      batched forward pass. For the reference variant, subtract the pre-computed
#      frozen reference scores to form log-ratio scores.
#   3. Pool into the preferred set u and rejected set v, sort each independently
#      (OT step 5), compute the uAOT loss (step 6).
#   4. Backpropagate and update π_θ.
def trainCPLuAOT(
	policy,
	prefData,
	uaotSteps,
	batchSize,
	lr,
	refPolicy=None,
	alpha=0.1,
	gamma=1.0,
	device='cpu',
	logInterval=1000,
	writer=None,
	prefDataTest=None,
	evalInterval=1000,
):
	n = len(prefData)
	useRef = refPolicy is not None

	# pre-convert all numpy arrays to device tensors once
	cache = _cachePrefTensors(prefData, device)

	# pre-compute all reference segment scores (π_ref is frozen, so these never change)
	posRefAll = negRefAll = None
	if useRef:
		print(f'pre-computing reference segment scores for {n} pairs...')
		posRefAll, negRefAll = computeRefScores(refPolicy, cache, alpha, gamma, device)  # (n,) CPU
		print('  done')

	policy = policy.to(device)
	policy.train()
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	for step in range(1, uaotSteps + 1):
		indices = np.random.randint(0, n, size=batchSize)

		# per-segment policy scores for this batch (differentiable)
		posScores, negScores = _batchPosNegScores(policy, cache, indices, alpha, gamma)

		# reference variant: subtract frozen reference scores → log-ratio scores
		if useRef:
			u = posScores - posRefAll[indices].to(device)
			v = negScores - negRefAll[indices].to(device)
		else:
			u, v = posScores, negScores

		loss, accuracy = uaot_loss(u, v)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# per-step curves: uAOT loss and OT-matched accuracy on the sampled batch
		if writer is not None:
			writer.add_scalar('loss/uaot', loss.item(), step)
			writer.add_scalar('accuracy/otUnpaired', accuracy.item(), step)

		# periodic curves: raw (non-OT) preference accuracy on the full train and test sets
		if writer is not None and step % evalInterval == 0:
			_logRawAccuracy(writer, 'rawTrain', policy, prefData, alpha, gamma, device, step)
			if prefDataTest is not None:
				_logRawAccuracy(writer, 'rawTest', policy, prefDataTest, alpha, gamma, device, step)

		if step % logInterval == 0:
			print(f'  step {step:>6}/{uaotSteps}  uaot_loss: {loss.item():.4f}  accuracy: {accuracy.item():.3f}')

	policy.eval()
	return policy
