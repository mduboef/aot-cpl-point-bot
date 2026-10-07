import numpy as np
import torch

from evaluate import preferenceAccuracy


# evaluate and log overall + per-type raw preference accuracy; mirrors the helper in cpl.py
# so BC curves land on the same accuracy/rawTrain and accuracy/rawTest tags as the CPL phases
def _logRawAccuracy(writer, tag, policy, prefData, alpha, gamma, device, step):
	acc = preferenceAccuracy(policy, prefData, alpha, gamma, device)
	policy.train()
	writer.add_scalar(f'accuracy/{tag}', acc['overall'], step)
	for st, d in acc['perType'].items():
		writer.add_scalar(f'{tag}PerType/type_{st}', d['accuracy'], step)
	return acc['overall']


def buildBCDataset(prefData):
	# pool (obs, action) pairs from every pos and neg segment in the preference dataset
	obsList, actList = [], []
	for pair in prefData:
		tPos = len(pair['pos_actions'])
		tNeg = len(pair['neg_actions'])
		obsList.append(pair['pos_states'][:tPos].astype(np.float32))
		actList.append(pair['pos_actions'].astype(np.float32))
		obsList.append(pair['neg_states'][:tNeg].astype(np.float32))
		actList.append(pair['neg_actions'].astype(np.float32))
	obs     = np.concatenate(obsList, axis=0)
	actions = np.concatenate(actList, axis=0)
	return obs, actions


def trainBC(
	policy,
	prefData,
	bcSteps,
	batchSize,
	lr,
	device='cpu',
	logInterval=1000,
	alpha=0.1,
	gamma=1.0,
	writer=None,
	prefDataTest=None,
	evalInterval=1000,
	stepOffset=0,
):
	obs, actions = buildBCDataset(prefData)
	n = len(obs)
	print(f'BC dataset: {n} (obs, action) pairs')

	obsT = torch.tensor(obs,     device=device)
	actT = torch.tensor(actions, device=device)

	policy = policy.to(device)
	policy.train()
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	for step in range(1, bcSteps + 1):
		# global step shares the x-axis with any following CPL phase so a warmup curve
		# is drawn before, not on top of, the preference-learning curve
		gStep    = stepOffset + step
		idx      = np.random.randint(0, n, size=batchSize)
		batchObs = obsT[idx]
		batchAct = actT[idx]

		_, logp = policy(batchObs, batchAct)
		loss = -logp.mean()

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		# per-step curve: BC loss
		if writer is not None:
			writer.add_scalar('loss/bc', loss.item(), gStep)

		# periodic curves: raw preference accuracy on the full train and test sets
		if writer is not None and step % evalInterval == 0:
			_logRawAccuracy(writer, 'rawTrain', policy, prefData, alpha, gamma, device, gStep)
			if prefDataTest is not None:
				_logRawAccuracy(writer, 'rawTest', policy, prefDataTest, alpha, gamma, device, gStep)

		if step % logInterval == 0:
			print(f'  step {step:>6}/{bcSteps}  bc_loss: {loss.item():.4f}')

	policy.eval()
	return policy
