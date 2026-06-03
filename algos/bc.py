import numpy as np
import torch


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


def trainBC(policy, prefData, bcSteps, batchSize, lr, device='cpu', logInterval=1000):
	obs, actions = buildBCDataset(prefData)
	n = len(obs)
	print(f'BC dataset: {n} (obs, action) pairs')

	obsT = torch.tensor(obs,     device=device)
	actT = torch.tensor(actions, device=device)

	policy = policy.to(device)
	policy.train()
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	for step in range(1, bcSteps + 1):
		idx      = np.random.randint(0, n, size=batchSize)
		batchObs = obsT[idx]
		batchAct = actT[idx]

		_, logp = policy(batchObs, batchAct)
		loss = -logp.mean()

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		if step % logInterval == 0:
			print(f'  step {step:>6}/{bcSteps}  bc_loss: {loss.item():.4f}')

	policy.eval()
	return policy
