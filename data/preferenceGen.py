import os
import pickle
import itertools
import numpy as np

# single-annotator reward on RAW (unnormalized) features (from env/pointbot.py augmentFeature):
#   feature[0] = grey_steps      (timesteps inside an obstacle)
#   feature[1] = white_steps     (timesteps in free space)  -- unused, weight 0
#   feature[2] = cumulative_dist (sum of ||state - GOAL|| over the trajectory)
#
# reward(sigma) = -(GREY_WEIGHT * grey_steps + DIST_WEIGHT * cumulative_dist)
# the ratio GREY_WEIGHT / DIST_WEIGHT ~= 160 lands the around/mid/through corridors
# (demo types 1/2/3) within ~5% of each other while types 4/5 are clearly worse and rejected.

WHITE_WEIGHT = 0.0
GREY_WEIGHT = 160.0
DIST_WEIGHT = 1.0

DEMO_TYPES = range(1, 6)


# single-annotator reward on raw features; white_steps gets no weight
def reward(feature):
	greySteps, whiteSteps, cumDist = feature
	return -(GREY_WEIGHT * greySteps + DIST_WEIGHT * cumDist)


# trims a labeled trajectory so actions has length T and states has length T+1
def extractTrajectory(data, prefix):
	states  = data[f'{prefix}states']
	actions = data[f'{prefix}actions']
	feature = data[f'{prefix}feature']
	T = len(actions)
	trimmedStates  = np.array(states[:T + 1])  # (T+1, 4)
	trimmedActions = np.array(actions)          # (T, 2)
	return trimmedStates, trimmedActions, feature


# returns the label and key prefix to keep for one pkl; Bad demos are dropped
# Optimal demos are single-trajectory (plain keys); Good demos are the Good half of a Good/Bad pair
def pickTrajectory(data):
	if 'feature' in data:
		return 'Optimal', ''
	if 'Good_feature' in data:
		return 'Good', 'Good_'
	return None, None


# loads every Good and Optimal demo across all corridor folders for one split
# split dir is 'trainingData' or 'testingData'; Bad demos are ignored
# returns one flat pool of items, each tagged with its source corridor type, label and reward
def loadDemos(dataDir, splitDir):
	items = []
	for typeIdx in DEMO_TYPES:
		typeDir = os.path.join(dataDir, splitDir, str(typeIdx))
		if not os.path.isdir(typeDir):
			continue
		for fname in sorted(os.listdir(typeDir)):
			if not fname.endswith('.pkl'):
				continue
			with open(os.path.join(typeDir, fname), 'rb') as f:
				data = pickle.load(f)
			label, prefix = pickTrajectory(data)
			if label is None:
				continue
			states, actions, feature = extractTrajectory(data, prefix)
			items.append({
				'fname':   fname,
				'type':    typeIdx,
				'label':   label,
				'states':  states,
				'actions': actions,
				'feature': feature,
				'reward':  reward(feature),
			})
	return items


# builds hard-labeled preference pairs over the whole pool (every unordered pair); the
# higher-reward demo is the preferred (pos) one. exact reward ties are skipped.
def generatePairs(items):
	pairs = []
	for a, b in itertools.combinations(range(len(items)), 2):
		if items[a]['reward'] > items[b]['reward']:
			posItem, negItem = items[a], items[b]
		elif items[b]['reward'] > items[a]['reward']:
			posItem, negItem = items[b], items[a]
		else:
			continue  # skip exact ties

		pairs.append({
			'pos_states':  posItem['states'],
			'pos_actions': posItem['actions'],
			'pos_feature': posItem['feature'],
			'pos_reward':  posItem['reward'],
			'pos_type':    posItem['type'],
			'neg_states':  negItem['states'],
			'neg_actions': negItem['actions'],
			'neg_feature': negItem['feature'],
			'neg_reward':  negItem['reward'],
			'neg_type':    negItem['type'],
		})
	return pairs


# prints the per-corridor reward structure so the near-optimal trio (types 1/2/3) and the
# always-rejected tail (types 4/5) are visible before any training consumes the data
def reportCorridors(items, split):
	byType = {}
	for it in items:
		byType.setdefault(it['type'], []).append(-it['reward'])  # cost = -reward, lower is better

	globalMin = min(min(costs) for costs in byType.values())
	print(f'\n=== {split} corridor structure (best cost = {globalMin:.0f}) ===')
	print(f'  {"type":>4} {"demos":>6} {"bestCost":>10} {"%aboveBest":>11}')
	for t in sorted(byType):
		best = min(byType[t])
		pct  = 100 * (best / globalMin - 1)
		print(f'  {t:>4} {len(byType[t]):>6} {best:>10.0f} {pct:>10.1f}%')


def main():
	dataDir = os.path.dirname(os.path.abspath(__file__))

	# (split label, source dir under dataDir, output preference filename)
	splits = [
		('train', 'trainingData', 'trainPreferences.pkl'),
		('test',  'testingData',  'testPreferences.pkl'),
	]

	for split, splitDir, outName in splits:
		items = loadDemos(dataDir, splitDir)
		pairs = generatePairs(items)

		reportCorridors(items, split)
		print(f'{split}: {len(items)} demos -> {len(pairs)} pairs')

		savePath = os.path.join(dataDir, splitDir, outName)
		with open(savePath, 'wb') as f:
			pickle.dump(pairs, f)
		print(f'saved -> {savePath}')


if __name__ == '__main__':
	main()
