import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.pointbot_const import OBSTACLE, MODE, GOAL_STATE, START_POS, END_POS, COLLISION_COST


def loadDemos(directory):
	# return sorted list of (filename, data_dict) tuples
	demos = []
	for fname in sorted(os.listdir(directory)):
		if fname.endswith('.pkl'):
			with open(os.path.join(directory, fname), 'rb') as f:
				data = pickle.load(f)
			demos.append((fname, data))
	return demos


def trimStates(states, actions):
	# strip zero-padding: actual trajectory has len(actions)+1 states
	T = len(actions) + 1
	return np.array(states)[:T]


def computeStats(states, actions):
	# returns (num_steps, obs_timesteps, cumulative_reward)
	T = len(actions) + 1
	trimmed = states[:T]
	obstacle = OBSTACLE[MODE]
	obsSteps = sum(1 for s in trimmed if obstacle.in_obs((s[0], s[2]), 0))
	cumReward = sum(
		-(np.linalg.norm(np.subtract(GOAL_STATE, s)) + COLLISION_COST * obstacle(s))
		for s in trimmed[:-1]  # T-1 rewards, one per action step
	)
	return T, obsSteps, cumReward


def drawObstacles(ax):
	# draw MODE=7 obstacle blocks as grey rectangles
	for obs in OBSTACLE[MODE].obs:
		xMin, xMax = obs.boundsx
		yMin, yMax = obs.boundsy
		rect = patches.Rectangle(
			(xMin, yMin),
			abs(xMax - xMin),
			abs(yMax - yMin),
			linewidth=1,
			facecolor='#a3a3a3',
			alpha=0.55,
			zorder=2,
		)
		ax.add_patch(rect)


def plotDemos(ax, demos, title):
	# plot all good (solid) and bad (dashed) trajectories; print per-traj stats
	drawObstacles(ax)
	colors = plt.cm.tab10(np.linspace(0, 1, max(len(demos), 1)))

	print(f"\n=== {title} ===")
	print(f"  {'file':<42} {'type':<5} {'steps':>6}  {'obs_steps':>9}  {'reward':>10}")
	print(f"  {'-'*42} {'-'*5} {'-'*6}  {'-'*9}  {'-'*10}")

	for i, (fname, data) in enumerate(demos):
		color = colors[i]

		goodStates = trimStates(data['Good_states'], data['Good_actions'])
		tg, obsG, rewG = computeStats(data['Good_states'], data['Good_actions'])
		ax.plot(goodStates[:, 0], goodStates[:, 2],
				color=color, linewidth=1.8, label=f'{fname[:-4]}')
		print(f"  {fname:<42} {'Good':<5} {tg:>6}  {obsG:>9}  {rewG:>10.1f}")

		# badStates = trimStates(data['Bad_states'], data['Bad_actions'])
		# tb, obsB, rewB = computeStats(data['Bad_states'], data['Bad_actions'])
		# ax.plot(badStates[:, 0], badStates[:, 2],
		# 		color=color, linewidth=1.8, linestyle='--', alpha=0.55)
		# print(f"  {fname:<42} {'Bad':<5} {tb:>6}  {obsB:>9}  {rewB:>10.1f}")

	# mark start and goal
	ax.scatter(START_POS[0], START_POS[1], c='green', s=120, zorder=6, marker='o', label='Start')
	ax.scatter(END_POS[0], END_POS[1], c='red', s=120, zorder=6, marker='o', label='Goal')


	# set limits so that they are 15 units beyond furtherest obstacle
	minX, maxX = START_POS[0], END_POS[0]
	minY, maxY = START_POS[1], END_POS[1]
	for obs in OBSTACLE[MODE].obs:
		minX = min(minX, obs.boundsx[0])
		maxX = max(maxX, obs.boundsx[1])
		minY = min(minY, obs.boundsy[0])
		maxY = max(maxY, obs.boundsy[1])
	padding = 15
	ax.set_xlim(minX-padding, maxX+padding)
	ax.set_ylim(minY-padding, maxY+padding)

	ax.set_xlabel('x position')
	ax.set_ylabel('y position')
	ax.set_title(title)
	ax.legend(fontsize=6, loc='upper right', ncol=2)
	ax.set_aspect('equal')
	ax.grid(True, alpha=0.25)


def main(savePathArg=None):
	# scriptDir = os.path.dirname(os.path.abspath(__file__))
	# trainDir = os.path.join(scriptDir, 'data', 'training_demos')
	# testDir = os.path.join(scriptDir, 'data', 'test_demos')

	# trainDemos = loadDemos(trainDir)
	# testDemos = loadDemos(testDir)

	# fig, (axTrain, axTest) = plt.subplots(1, 2, figsize=(16, 9))
	# plotDemos(axTrain, trainDemos, f'Training Demos')
	# plotDemos(axTest, testDemos, f'Test Demos')
	# plt.tight_layout()

	# savePath = savePathArg or os.path.join(scriptDir, 'rollouts.png')
	# plt.savefig(savePath, dpi=150, bbox_inches='tight')
	# print(f"\nFigure saved to {savePath}")
	# plt.show()


	scriptDir = os.path.dirname(os.path.abspath(__file__))
	dir1 = os.path.join(scriptDir, 'data', '1')
	dir2 = os.path.join(scriptDir, 'data', '2')
	dir3 = os.path.join(scriptDir, 'data', '3')
	dir4 = os.path.join(scriptDir, 'data', '4')
	dir5 = os.path.join(scriptDir, 'data', '5')


	allDemos = [
		(loadDemos(dir1), 'Demos Type 1', 'rollouts_1.png'),
		(loadDemos(dir2), 'Demos Type 2', 'rollouts_2.png'),
		(loadDemos(dir3), 'Demos Type 3', 'rollouts_3.png'),
		(loadDemos(dir4), 'Demos Type 4', 'rollouts_4.png'),
		(loadDemos(dir5), 'Demos Type 5', 'rollouts_5.png'),
	]

	for demos, title, fname in allDemos:
		fig, ax = plt.subplots(figsize=(9, 9))
		plotDemos(ax, demos, title)
		plt.tight_layout()
		savePath = os.path.join(scriptDir, fname)
		plt.savefig(savePath, dpi=150, bbox_inches='tight')
		print(f"\nFigure saved to {savePath}")
		plt.close(fig)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--save', type=str, default=None, help='output image path')
	args = parser.parse_args()
	main(savePathArg=args.save)
