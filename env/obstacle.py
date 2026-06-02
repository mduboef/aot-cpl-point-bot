import numpy as np


class Obstacle:
	def __init__(self, boundsx, boundsy):
		self.boundsx = boundsx
		self.boundsy = boundsy

	# x is a state array (x, vx, y, vy); collision checks position dims 0 and 2
	def __call__(self, x):
		return float(
			self.boundsx[0] <= x[0] <= self.boundsx[1] and
			self.boundsy[0] <= x[2] <= self.boundsy[1]
		)

	# point is a 2d (x, y) where x=state[0], y=state[2]
	def in_obs(self, point, buffer):
		return (
			self.boundsx[0] - buffer <= point[0] <= self.boundsx[1] + buffer and
			self.boundsy[0] - buffer <= point[1] <= self.boundsy[1] + buffer
		)


class ComplexObstacle:
	def __init__(self, bounds):
		self.obs = [Obstacle(bx, by) for bx, by in bounds]

	def __call__(self, x):
		return float(np.max([o(x) for o in self.obs]))

	def in_obs(self, point, buffer):
		return any(o.in_obs(point, buffer) for o in self.obs)
