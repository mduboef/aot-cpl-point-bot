# evaluation metrics shared across all methods (BC, CPL, CPL+pAOT, CPL+uAOT)
#
# preference accuracy here is computed on the ORIGINAL annotator pairs, not on the
# optimal-transport (sorted) pairings used inside the pAOT/uAOT training loss. This
# lets us check whether a policy actually prefers the truly-preferred segment in each
# real pair, independent of whatever pairing its training objective optimized.

import numpy as np
import torch


# computes score_π(σ) = Σ_t γ^t α log π(a_t | s_t) for a single segment
# states has shape (T+1, obs_dim) and actions has shape (T, act_dim); we align each
# action a_t with the state s_t it was taken from by slicing states to T rows.
def segmentScore(policy, states, actions, alpha, gamma, device):
	tSteps = len(actions)
	obsT = torch.tensor(np.asarray(states[:tSteps], dtype=np.float32), device=device)
	actT = torch.tensor(np.asarray(actions,         dtype=np.float32), device=device)
	with torch.no_grad():
		_, logp = policy(obsT, actT)
	if gamma == 1.0:
		return (alpha * logp.sum()).item()
	discounts = gamma ** torch.arange(tSteps, dtype=torch.float32, device=device)
	return (alpha * discounts * logp).sum().item()


# fraction of original (non-OT) preference pairs where score_π(σ+) > score_π(σ-)
# returns overall accuracy, a per-strategy-type breakdown, and pair counts; the
# per-type breakdown matters because minority dominant strategies (types 4 and 5)
# are exactly where a collapsed policy is expected to fail.
def preferenceAccuracy(policy, prefData, alpha=0.1, gamma=1.0, device='cpu'):
	policy = policy.to(device)
	policy.eval()

	correct = 0
	correctByType, totalByType = {}, {}
	for pair in prefData:
		posScore = segmentScore(policy, pair['pos_states'], pair['pos_actions'], alpha, gamma, device)
		negScore = segmentScore(policy, pair['neg_states'], pair['neg_actions'], alpha, gamma, device)
		isCorrect = int(posScore > negScore)
		correct += isCorrect

		st = int(pair.get('strategy_type', -1))
		correctByType[st] = correctByType.get(st, 0) + isCorrect
		totalByType[st]   = totalByType.get(st, 0) + 1

	overall = correct / len(prefData) if len(prefData) else 0.0
	perType = {
		st: {'accuracy': correctByType[st] / totalByType[st], 'nPairs': totalByType[st]}
		for st in sorted(totalByType)
	}
	return {'overall': overall, 'nPairs': len(prefData), 'perType': perType}


# computes the pAOT loss of a trained policy against π_ref over a full preference set, plus
# first-order stochastic dominance (FSD) violation diagnostics. mirrors the training objective
# in algos/cpl_paot.py (sort both margin sets independently, match by quantile, CPL loss on the
# matched pairs) but runs once over every pair with no gradients. a high loss / high violation
# rate means the trained policy's margin distribution fails to stochastically dominate π_ref's.
#
# u_θ^i = score(σ+; π_θ) - score(σ-; π_θ);  v_ref^i = score(σ+; π_ref) - score(σ-; π_ref)
# after independent sorting, a violation at quantile i is u_sorted[i] < v_sorted[i] (the policy
# margin fails to dominate the reference margin there). frequency = fraction of quantiles
# violated; severity = mean / max shortfall (v_sorted - u_sorted) over the violated quantiles.
def paotLoss(policy, refPolicy, prefData, alpha=0.1, gamma=1.0, device='cpu'):
	# imported lazily: algos.cpl_paot imports preferenceAccuracy from this module at load time,
	# so a top-level import here would be circular
	from algos.cpl_paot import _cachePrefTensors, computeRefMargins, _batchPolicyMargins, paot_loss

	policy = policy.to(device)
	policy.eval()
	refPolicy = refPolicy.to(device)
	refPolicy.eval()

	cache = _cachePrefTensors(prefData, device)
	n = len(cache)

	# reference and policy margins over every pair (one batched forward pass each)
	vRef = computeRefMargins(refPolicy, cache, alpha, gamma, device).to(device)  # (n,)
	with torch.no_grad():
		uTheta = _batchPolicyMargins(policy, cache, range(n), alpha, gamma, device)  # (n,)
		loss, otAccuracy = paot_loss(uTheta, vRef)

		# FSD violation diagnostics on the sorted (quantile-matched) margins
		uSorted   = torch.sort(uTheta).values
		vSorted   = torch.sort(vRef).values
		shortfall = vSorted - uSorted          # > 0 where the policy violates dominance
		violated  = shortfall > 0
		nViolations = int(violated.sum().item())
		meanShortfall = float(shortfall[violated].mean().item()) if nViolations else 0.0
		maxShortfall  = float(shortfall[violated].max().item())  if nViolations else 0.0

	return {
		'paotLoss':      float(loss.item()),
		'nPairs':        n,
		'nViolations':   nViolations,
		'violationFreq': nViolations / n if n else 0.0,
		'meanShortfall': meanShortfall,
		'maxShortfall':  maxShortfall,
		'otAccuracy':    float(otAccuracy.item()),
	}
