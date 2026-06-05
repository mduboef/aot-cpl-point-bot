import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.pointbot_const import OBSTACLE, MODE, START_POS, END_POS

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
GREY_WEIGHT = 160.0
DIST_WEIGHT = 1.0
DEMO_TYPES  = range(1, 6)


def reward(feature):
	greySteps, whiteSteps, cumDist = feature
	return -(GREY_WEIGHT * greySteps + DIST_WEIGHT * cumDist)


def loadDemos(dataDir, split):
	items = []
	for typeIdx in DEMO_TYPES:
		typeDir = os.path.join(dataDir, f'{typeIdx}_{split}')
		if not os.path.isdir(typeDir):
			continue
		for fname in sorted(os.listdir(typeDir)):
			if not fname.endswith('.pkl'):
				continue
			with open(os.path.join(typeDir, fname), 'rb') as f:
				data = pickle.load(f)
			actions = data['Good_actions']
			states  = np.array(data['Good_states'])[:len(actions) + 1]
			feature = data['Good_feature']
			items.append({
				'fname':   fname,
				'type':    typeIdx,
				'states':  states,
				'feature': feature,
				'reward':  reward(feature),
			})
	return items


def printRanking(items, split):
	ranked = sorted(items, key=lambda x: x['reward'], reverse=True)
	n = len(ranked)

	print(f'\n=== {split.upper()} — demos ranked by reward (best → worst) ===')
	print(f'  {"rank":>5}  {"type":>5}  {"grey_steps":>11}  {"white_steps":>12}  {"cum_dist":>10}  {"reward":>10}  file')
	print(f'  {"─"*5}  {"─"*5}  {"─"*11}  {"─"*12}  {"─"*10}  {"─"*10}  {"─"*38}')

	for rank, item in enumerate(ranked):
		grey, white, cumDist = item['feature']
		print(
			f'  {rank+1:>5}  {item["type"]:>5}  {grey:>11.1f}  {white:>12.1f}  '
			f'{cumDist:>10.1f}  {item["reward"]:>10.1f}  {item["fname"]}'
		)

	print()
	print(f'  Best  reward: {ranked[0]["reward"]:.1f}  (type {ranked[0]["type"]}, {ranked[0]["fname"]})')
	print(f'  Worst reward: {ranked[-1]["reward"]:.1f}  (type {ranked[-1]["type"]}, {ranked[-1]["fname"]})')

	byType = {}
	for item in items:
		byType.setdefault(item['type'], []).append(item['reward'])

	print(f'\n  Per-type reward summary ({split}):')
	print(f'  {"type":>5}  {"count":>6}  {"best":>10}  {"mean":>10}  {"worst":>10}')
	print(f'  {"─"*5}  {"─"*6}  {"─"*10}  {"─"*10}  {"─"*10}')
	for t in sorted(byType):
		vals = byType[t]
		print(f'  {t:>5}  {len(vals):>6}  {max(vals):>10.1f}  {np.mean(vals):>10.1f}  {min(vals):>10.1f}')


def drawObstacles(ax):
	for obs in OBSTACLE[MODE].obs:
		xMin, xMax = obs.boundsx
		yMin, yMax = obs.boundsy
		rect = patches.Rectangle(
			(xMin, yMin),
			abs(xMax - xMin),
			abs(yMax - yMin),
			linewidth=1,
			facecolor="#646464",
			alpha=0.55,
			zorder=2,
		)
		ax.add_patch(rect)


# plots all demos colored green→white→red by reward rank (not raw reward value)
def plotRankedRollouts(items, split, savePath):
	ranked = sorted(items, key=lambda x: x['reward'], reverse=True)
	n = len(ranked)

	cmap = mcolors.LinearSegmentedColormap.from_list('ranked', ['green', 'whitesmoke', 'red'])

	fig, ax = plt.subplots(figsize=(10, 10))
	drawObstacles(ax)

	for rank, item in enumerate(ranked):
		normalizedRank = rank / max(n - 1, 1)
		color = cmap(normalizedRank)
		states = item['states']
		ax.plot(states[:, 0], states[:, 2], color=color, linewidth=1.4, alpha=0.85, zorder=3)

	ax.scatter(START_POS[0], START_POS[1], c='green', s=140, zorder=6, marker='o', label='Start')
	ax.scatter(END_POS[0], END_POS[1], c='red', s=140, zorder=6, marker='o', label='Goal')

	minX = min(START_POS[0], END_POS[0])
	maxX = max(START_POS[0], END_POS[0])
	minY = min(START_POS[1], END_POS[1])
	maxY = max(START_POS[1], END_POS[1])
	for obs in OBSTACLE[MODE].obs:
		minX = min(minX, obs.boundsx[0])
		maxX = max(maxX, obs.boundsx[1])
		minY = min(minY, obs.boundsy[0])
		maxY = max(maxY, obs.boundsy[1])
	padding = 15
	ax.set_xlim(minX - padding, maxX + padding)
	ax.set_ylim(minY - padding, maxY + padding)

	sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=n))
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
	cbar.set_label('rank (1 = best reward)', fontsize=9)
	cbar.set_ticks([1, (n + 1) // 2, n])
	cbar.set_ticklabels(['best', 'median', 'worst'])

	ax.set_xlabel('x position')
	ax.set_ylabel('y position')
	ax.set_title(f'Ranked Rollouts — {split} ({n} demos)')
	ax.legend(fontsize=8, loc='upper right')
	ax.set_aspect('equal')
	ax.grid(True, alpha=0.25)

	plt.tight_layout()
	plt.savefig(savePath, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print(f'saved -> {savePath}')


def main():
	for split in ['train', 'test']:
		items = loadDemos(DATA_DIR, split)
		printRanking(items, split)
		savePath = os.path.join(os.path.dirname(DATA_DIR), 'demoPlots', f'rankedRollouts_{split}.png')
		plotRankedRollouts(items, split, savePath)


if __name__ == '__main__':
	main()
