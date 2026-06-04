import os
import sys
import pickle
import itertools
import numpy as np

# feature vector layout (from env/pointbot.py augmentFeature):
#   feature[0] = grey_steps      (timesteps inside an obstacle)
#   feature[1] = white_steps     (timesteps in free space)
#   feature[2] = cumulative_dist (sum of ||state - GOAL|| over trajectory)

# normalization constants — bring each feature into [0, 1] before applying weights
# grey/white: divided by HORIZON (100); dist: divided by observed max across all demos (~11554)
NORM_GREY  = 100.0
NORM_WHITE = 100.0
NORM_DIST  = 12000.0

# synthetic annotator reward (applied to normalized features):
#   reward(σ) = w_grey * (grey/NORM_GREY) + w_white * (white/NORM_WHITE) + w_dist * (dist/NORM_DIST)

ANNOTATORS = {
	1: {'grey': -0.95, 'white':  0.0,  'dist': -0.05},
	2: {'grey': -0.70, 'white':  0.0,  'dist': -0.30},
	3: {'grey':  0.0,  'white':  0.0,  'dist': -1.00},
	4: {'grey':  0.0,  'white': -0.70, 'dist': -0.30},
	5: {'grey':  0.0,  'white': -0.95, 'dist': -0.05},
}


def loadDemos(directory):
	demos = []
	for fname in sorted(os.listdir(directory)):
		if fname.endswith('.pkl'):
			with open(os.path.join(directory, fname), 'rb') as f:
				data = pickle.load(f)
			demos.append((fname, data))
	return demos


def annotatorReward(feature, annotator):
	greySteps, whiteSteps, cumDist = feature
	return (annotator['grey']  * (greySteps  / NORM_GREY)
		+   annotator['white'] * (whiteSteps / NORM_WHITE)
		+   annotator['dist']  * (cumDist    / NORM_DIST))


def extractTrajectory(data, kind):
	# kind: 'Good' or 'Bad'
	states  = data[f'{kind}_states']
	actions = data[f'{kind}_actions']
	feature = data[f'{kind}_feature']
	T = len(actions)
	trimmedStates  = np.array(states[:T + 1])  # (T+1, 4)
	trimmedActions = np.array(actions)          # (T, 2)
	return trimmedStates, trimmedActions, feature


def generatePairs(trajectories, annotator, strategyType):
	rewards = [annotatorReward(f, annotator) for _, _, f in trajectories]

	pairs = []
	for i, j in itertools.combinations(range(len(trajectories)), 2):
		if rewards[i] > rewards[j]:
			posIdx, negIdx = i, j
		elif rewards[j] > rewards[i]:
			posIdx, negIdx = j, i
		else:
			continue  # skip ties

		posStates,  posActions,  posFeat  = trajectories[posIdx]
		negStates,  negActions,  negFeat  = trajectories[negIdx]

		pairs.append({
			'pos_states':    posStates,
			'pos_actions':   posActions,
			'pos_feature':   posFeat,
			'pos_reward':    rewards[posIdx],
			'neg_states':    negStates,
			'neg_actions':   negActions,
			'neg_feature':   negFeat,
			'neg_reward':    rewards[negIdx],
			'strategy_type': strategyType,
		})

	return pairs


def main():
	dataDir = os.path.dirname(os.path.abspath(__file__))

	trainingPairs = []


	print("\n=== Generating training preference pairs ===")

	for typeIdx in range(1, 6):
		typeDir    = os.path.join(dataDir, f'{typeIdx}_train')
		annotator  = ANNOTATORS[typeIdx]
		demos      = loadDemos(typeDir)

		# pool only the Good trajectories from every demo in this type
		trajectories = []
		for fname, data in demos:
			states, actions, feature = extractTrajectory(data, 'Good')
			trajectories.append((states, actions, feature))

		pairs = generatePairs(trajectories, annotator, typeIdx)
		trainingPairs.extend(pairs)

		print(f"type {typeIdx}: {len(demos)} demos  "
			f"→ {len(pairs)} pairs")

		ranked = sorted(
			zip(demos, trajectories),
			key=lambda x: annotatorReward(x[1][2], annotator),
			reverse=True,
		)
		print(f"  {'rank':<5} {'file':<40} {'grey':>5} {'white':>6} {'dist':>8} {'reward':>9}")
		for rank, ((fname, _), (_, _, feat)) in enumerate(ranked, 1):
			reward = annotatorReward(feat, annotator)
			print(f"  {rank:<5} {fname:<40} {feat[0]:>5} {feat[1]:>6} {feat[2]:>8.0f} {reward:>9.4f}")
		print()

	print(f"\ntotal pairs: {len(trainingPairs)}")

	savePath = os.path.join(dataDir, 'trainPreferences.pkl')
	with open(savePath, 'wb') as f:
		pickle.dump(trainingPairs, f)
	print(f"saved → {savePath}")









	print("\n=== Generating testing preference pairs ===")


	testingPairs = []

	for typeIdx in range(1, 6):
		typeDir    = os.path.join(dataDir, f'{typeIdx}_test')
		annotator  = ANNOTATORS[typeIdx]
		demos      = loadDemos(typeDir)

		# pool only the Good trajectories from every demo in this type
		trajectories = []
		for fname, data in demos:
			states, actions, feature = extractTrajectory(data, 'Good')
			trajectories.append((states, actions, feature))

		pairs = generatePairs(trajectories, annotator, typeIdx)
		testingPairs.extend(pairs)

		print(f"type {typeIdx}: {len(demos)} demos  "
			f"→ {len(pairs)} pairs")

		ranked = sorted(
			zip(demos, trajectories),
			key=lambda x: annotatorReward(x[1][2], annotator),
			reverse=True,
		)
		print(f"  {'rank':<5} {'file':<40} {'grey':>5} {'white':>6} {'dist':>8} {'reward':>9}")
		for rank, ((fname, _), (_, _, feat)) in enumerate(ranked, 1):
			reward = annotatorReward(feat, annotator)
			print(f"  {rank:<5} {fname:<40} {feat[0]:>5} {feat[1]:>6} {feat[2]:>8.0f} {reward:>9.4f}")
		print()

	print(f"\ntotal pairs: {len(testingPairs)}")

	savePath = os.path.join(dataDir, 'testPreferences.pkl')
	with open(savePath, 'wb') as f:
		pickle.dump(testingPairs, f)
	print(f"saved → {savePath}")


if __name__ == '__main__':
	main()
