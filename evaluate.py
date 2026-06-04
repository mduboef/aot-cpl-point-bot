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
