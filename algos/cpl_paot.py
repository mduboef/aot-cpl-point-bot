# algos/cpl_paot.py
#
# CPL with Paired Alignment via Optimal Transport (pAOT).
# Formalization reference: "CPL with pAOT" section of finalReport.tex.
#
# This file implements phase 3 only (pAOT contrastive training of π_θ).
# Phases 1 and 2 (BC training of π_ref and optional BC warmup of π_θ)
# are handled by trainBC() from algos/bc.py, called from train.py.
#
# Three-phase structure (driven by train.py):
#   Phase 1 - BC-train π_ref with trainBC()
#   Phase 2 - BC-warmup π_θ with trainBC()  (optional; skip by setting theta_bc_steps=0)
#   Phase 3 - pAOT contrastive training with trainCPLpAOT()  ← this file
#
# Data format (preference pairs from data/preferenceGen.py):
#   pos_states:  np array (T+1, obs_dim)
#   pos_actions: np array (T,   act_dim)
#   neg_states:  np array (T+1, obs_dim)
#   neg_actions: np array (T,   act_dim)
#   (trajectory lengths T vary across pairs: min=36, max=98, mean≈68)

import numpy as np
import torch


# paot_loss
# Formalization steps 5 and 6 of "CPL with pAOT" in finalReport.tex.
#
# u_theta: per-pair policy margins,    shape (n,) - u_θ^i = score(σ+; π_θ) - λ·score(σ-; π_θ)
# v_ref:   per-pair reference margins, shape (n,) - v_ref^i = score(σ+; π_ref) - λ·score(σ-; π_ref)
# (λ, the contrastive bias, is applied when the margins are built - see
# computeRefMargins / _batchPolicyMargins - so this function is unchanged by it)
#
# Step 5 - sort both margin sets independently (1D optimal transport, northwest corner):
#   u_θ^(1) ≤ u_θ^(2) ≤ ... ≤ u_θ^(n)
#   v_ref^(1) ≤ v_ref^(2) ≤ ... ≤ v_ref^(n)
#   i-th lowest policy margin is matched with the i-th lowest reference margin.
#
# Step 6 - CPL loss on OT-matched pairs:
#   L(θ) = (1/n) Σ_i -log [ exp(u_θ^(i)) / (exp(u_θ^(i)) + exp(v_ref^(i))) ]
def paot_loss(u_theta, v_ref):
	u_sorted = torch.sort(u_theta).values
	v_sorted = torch.sort(v_ref).values

	logit = u_sorted - v_sorted
	# numerically stable: -log sigmoid(logit) = log(1 + exp(-logit))
	max_val = torch.clamp(-logit, min=0)
	loss = (torch.log(torch.exp(-max_val) + torch.exp(-logit - max_val)) + max_val).mean()

	with torch.no_grad():
		accuracy = (u_sorted > v_sorted).float().mean()

	return loss, accuracy


# Convert all numpy arrays in prefData to float32 tensors once before training.
# Avoids repeated numpy→tensor conversions inside the training loop.
# Returns a list of dicts with pre-allocated device tensors.
# Each entry stores only the T obs/act pairs needed for scoring (not the T+1 states).
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
#
# Each segment i has length T_i (variable). We concatenate all (state, action) pairs
# from all segments into one flat batch (Σ T_i, obs_dim) / (Σ T_i, act_dim), run
# one forward pass, then split the resulting log probs back by segment and sum.
#
# score(σ; π) = Σ_t γ^t α log π(a_t | s_t)   [finalReport.tex step 3]
# (γ=1.0 by default, matching CPL paper convention)
#
# segObs:  list of (T_i, obs_dim) tensors, one per segment
# segAct:  list of (T_i, act_dim) tensors, one per segment
# Returns: list of scalar tensors (one per segment), differentiable w.r.t. policy
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


# computeRefMargins
# Pre-compute v_ref^i for all n preference pairs using a single forward pass over
# all segments. Called once before training begins; π_ref is frozen so v_ref never
# changes. Returns a float32 CPU tensor of shape (n,) for cheap random indexing
# during the training loop.
#
# bias (λ) downweights the rejected-segment score: v_ref^i = score(σ+) - λ·score(σ-).
# λ must match the one used for the policy margins so that u_θ = v_ref when π_θ = π_ref.
def computeRefMargins(refPolicy, cache, alpha, gamma, device, bias=1.0):
	refPolicy.eval()
	segObs, segAct = [], []
	for entry in cache:
		segObs += [entry['posS'], entry['negS']]
		segAct += [entry['posA'], entry['negA']]

	with torch.no_grad():
		scores = _batchSegmentScores(refPolicy, segObs, segAct, alpha, gamma)

	vList = []
	for i in range(len(cache)):
		posScore = scores[2 * i]
		negScore = scores[2 * i + 1]
		vList.append((posScore - bias * negScore).item())

	return torch.tensor(vList, dtype=torch.float32)  # (n,) CPU


# _batchPolicyMargins
# Compute u_theta^i = score(σ+; π_θ) - λ·score(σ-; π_θ) for a batch of pairs.
# Uses a single batched forward pass over all 2*batch_size segments.
# Differentiable w.r.t. policy parameters.
def _batchPolicyMargins(policy, cache, indices, alpha, gamma, device, bias=1.0):
	segObs, segAct = [], []
	for idx in indices:
		entry = cache[idx]
		segObs += [entry['posS'], entry['negS']]
		segAct += [entry['posA'], entry['negA']]

	scores = _batchSegmentScores(policy, segObs, segAct, alpha, gamma)

	uList = []
	for i in range(len(indices)):
		posScore = scores[2 * i]
		negScore = scores[2 * i + 1]
		uList.append(posScore - bias * negScore)

	return torch.stack(uList)


# trainCPLpAOT
# Phase 3: pAOT contrastive training of π_θ.
#
# π_ref (refPolicy) must be BC-trained and frozen before calling this.
# π_θ (policy) should have its BC warmup done before calling this.
#
# Key steps per training iteration:
#   1. Sample a random mini-batch of batchSize preference pairs.
#   2. Compute λ-biased policy margins u_theta (differentiable) via one batched forward pass.
#   3. Index into pre-computed reference margins v_ref (no forward pass needed).
#   4. Sort both independently (OT step 5), compute pAOT loss (step 6).
#   5. Backpropagate and update π_θ.
def trainCPLpAOT(
	policy,
	refPolicy,
	prefData,
	paotSteps,
	batchSize,
	lr,
	alpha=0.1,
	gamma=1.0,
	bias=1.0,
	device='cpu',
	logInterval=1000,
	evaluator=None,
	stepOffset=0,
):
	n = len(prefData)

	# pre-convert all numpy arrays to device tensors once
	cache = _cachePrefTensors(prefData, device)

	# pre-compute all reference margins (π_ref is frozen, so these never change)
	print(f'pre-computing reference margins for {n} pairs...')
	vRefAll = computeRefMargins(refPolicy, cache, alpha, gamma, device, bias=bias)  # (n,) CPU
	print(f'  done')

	policy = policy.to(device)
	policy.train()
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	for step in range(1, paotSteps + 1):
		# global step continues past any BC warmup so both phases share one x-axis
		gStep   = stepOffset + step
		indices = np.random.randint(0, n, size=batchSize)

		# compute policy margins (differentiable) for this batch
		u_theta = _batchPolicyMargins(policy, cache, indices, alpha, gamma, device, bias=bias)

		# retrieve pre-computed reference margins for this batch
		v_ref = vRefAll[indices].to(device)

		loss, accuracy = paot_loss(u_theta, v_ref)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# TensorBoard: per-step contrastive loss, plus periodic evals on their intervals (tbLogging.py)
		if evaluator is not None:
			evaluator.logLoss('contrastiveLoss', loss.item(), gStep)
			evaluator.maybeLog(policy, gStep)

		if step % logInterval == 0:
			print(f'  step {step:>6}/{paotSteps}  paot_loss: {loss.item():.4f}  accuracy: {accuracy.item():.3f}')

	policy.eval()
	return policy
