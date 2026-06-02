import math
import numpy as np
from gym import Env, utils
from gym.spaces import Box

from .pointbot_const import (
	START_STATE, GOAL_STATE, END_POS, GOAL_THRESH,
	MAX_FORCE, HORIZON, NOISE_SCALE, AIR_RESIST, COLLISION_COST,
	MODE, OBSTACLE,
)


def processAction(a):
	return np.clip(a, -MAX_FORCE, MAX_FORCE)


def lqrGains(A, B, Q, R, T):
	Ps = [Q]
	Ks = []
	for t in range(T):
		P = Ps[-1]
		Ps.append(Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B)
			.dot(np.linalg.inv(R + B.T.dot(P).dot(B))).dot(B.T).dot(P).dot(A))
	Ps.reverse()
	for t in range(T):
		Ks.append(-np.linalg.inv(R + B.T.dot(Ps[t + 1]).dot(B)).dot(B.T).dot(P).dot(A))
	return Ks, Ps


class PointBot(Env, utils.EzPickle):
	def __init__(self):
		utils.EzPickle.__init__(self)
		self.hist = self.rewards = self.done = self.time = self.state = self.obsTime = None
		self.A = np.eye(4)
		self.A[0, 1] = self.A[2, 3] = 1
		self.A[1, 1] = self.A[3, 3] = 1 - AIR_RESIST
		self.B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
		self.horizon = HORIZON
		self._max_episode_steps = HORIZON
		self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
		self.observation_space = Box(-np.ones(4) * np.inf, np.ones(4) * np.inf)
		self.mode = MODE
		self.obstacle = OBSTACLE[MODE]
		self.startState = np.array(START_STATE, dtype=float)

	def step(self, a):
		a = processAction(a)
		self.augmentFeature(self.state)
		nextState = self._nextState(self.state, a)
		curCost = self.stepCost(self.state, a)
		reward = -curCost
		self.rewards.append(reward)
		self.state = nextState
		self.time += 1
		self.hist.append(self.state)
		if self.obstacle.in_obs((self.state[0], self.state[2]), 0):
			self.obsTime += 1
		goalReached = np.linalg.norm(np.array([self.state[0], self.state[2]]) - np.array(END_POS)) <= GOAL_THRESH
		self.done = self.time >= HORIZON or goalReached
		return self.state, reward, self.done, {}

	def augmentFeature(self, state):
		point = (state[0], state[2])
		inObs = False
		for obs in self.obstacle.obs:
			if obs.in_obs(point, 0):
				self.feature[0] += 1
				inObs = True
		if not inObs:
			self.feature[1] += 1
		distToGoal = np.linalg.norm(np.subtract(END_POS, state[[0, 2]]))
		self.feature[2] += distToGoal

	def reset(self):
		self.state = self.startState + np.random.randn(4) * NOISE_SCALE
		self.time = 0
		self.obsTime = 0
		self.rewards = []
		self.done = False
		self.hist = [self.state.copy()]
		initDist = np.linalg.norm(np.subtract(END_POS, self.state[[0, 2]]))
		self.feature = [0, 1, initDist]
		return self.state

	def _nextState(self, s, a):
		return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

	def stepCost(self, s, a):
		return np.linalg.norm(np.subtract(GOAL_STATE, s)) + self.collisionCost(s)

	def collisionCost(self, s):
		return COLLISION_COST * self.obstacle(s)

	def isStable(self, s):
		return np.linalg.norm(np.array([s[0], s[2]]) - np.array(END_POS)) <= GOAL_THRESH

	def teacher(self):
		return PointBotTeacher()


class PointBotTeacher:
	def __init__(self):
		self.env = PointBot()
		self.Ks, self.Ps = lqrGains(self.env.A, self.env.B, np.eye(4), 50 * np.eye(2), HORIZON)

	def getRollout(self):
		obs = self.env.reset()
		O, A, rewards = [obs.copy()], [], []
		noiseStd = 0.2
		for i in range(HORIZON):
			noiseIdx = np.random.randint(HORIZON)
			if i < HORIZON / 4:
				action = [0.1, 0.25]
			elif i < HORIZON / 2:
				action = [0.4, 0.0]
			elif i < HORIZON * 2 / 3:
				action = [0.0, -0.5]
			else:
				action = self._expertControl(obs, i)
			if i < noiseIdx:
				action = (np.array(action) + np.random.normal(0, noiseStd, 2)).tolist()
			A.append(action)
			obs, reward, done, _ = self.env.step(action)
			O.append(obs.copy())
			rewards.append(reward)
			if done:
				break
		if not self.env.isStable(obs):
			return self.getRollout()
		return {"obs": np.array(O), "ac": np.array(A), "rewards": np.array(rewards)}

	def _expertControl(self, s, t):
		return self.Ks[t].dot(s)
