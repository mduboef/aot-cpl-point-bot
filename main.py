# TODO read in pre-generated preference data

# TODO do any requried initalizion & environment setup need
	# we will need to rollout the policy and stuff

	# trajectories are characterized by 3 features
		# 1. total distance to goal
		# 2. number of steps in grey (obstacle) area
		# 3. number of steps in white (non-obstacle) area

	# reward function unknown
	# hypothesis space for the reward function is the negative simplex
		# w_white >= 0
		# w_grey <= 0
		# w_distance <= 0
		# -1 = w_distance + w_grey + w_white

# TODO behavior cloning on raw preference data

# TODO plot demos and BC rollouts

# TODO train CPL policy

# TODO train CPL policy with beta regulariztion of ___

# TODO create pAOT pairings

# TODO train cpl_paot policy

# TODO create uAOT pairings

# TODO train cpl_uaot policy

# TODO evaluate all policies (BC, CPL, CPL_biased, cpl_paot, cpl_uaot)

	# the metrics we are planning on using

		# Primary: measure of first order stochastic domiance violations
			# the loss function we are optimizing in cpl_uAOT and cpl_pAOT

		# Secondary: classification accuracy on original preference pairs



	# Metrics used in the PSD paper that are worth considering

		# pareto domiance (min, avg & max)
			# Calculate Pr( f(\xi_{\pi}) >= f(\xi_{demo_i}) ) for all demos
				# ? What is the function f doing here? Is it just a measure of feature counts or are we looking at reward?
					# ? sample random reward function (random weights)

		# stochastic dominance
			# sample random reward function (random weights)
			# determine if policy stochastically dominates the testing demos
				# stochastic dominance meaning \pi is more likely to exceed the 
				# ? how is this calculated? How do we get the Pr(training demos exeed a reward threshold)
					# ? is it just (# of demos that exeed the threshold) / (total # of testing demos)
					#  since the demos have a single predetermined set of features each trajectory individually has 100% prob of scoring higher than the threshold if when the threshold is < their actualized return

		# these can be measures for the demonstrations themselves as a baseline
			# the PSD paper does this
						

import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from env.pointbot import PointBotEnv
from plotRollouts import loadDemos, plotDemos


def main():
	# TODO load preference data
	return 

main()