import os, shutil
import torch
import torch.nn as nn
from colorama import Fore

from mlp import MLPGaussianActor


# π_ref is trained once with pure BC (python3 train.py --method ref, configured by configs/ref.yaml)
# and saved to models/REF_POLICY/; every other method loads it from there


def prepareRefDir(refPolicyDir):
	# clears out any existing reference policy so the new ref training run can replace it
	if os.path.exists(refPolicyDir):
		print(Fore.YELLOW + f'\nWARNING: reference policy already exists at {refPolicyDir}. It will be overwritten.'+ Fore.WHITE)
		shutil.rmtree(refPolicyDir)
	os.makedirs(refPolicyDir)
	return refPolicyDir


def loadRefPolicy(refPolicyDir, obsDim, actDim, device):
	# loads the frozen π_ref, erroring out if it hasn't been trained yet
	refPolicyPath = os.path.join(refPolicyDir, 'policy.pt')
	if not os.path.isfile(refPolicyPath):
		raise FileNotFoundError(f'no reference policy found at {refPolicyPath}; '
			'a reference policy needs to be trained first with "python3 train.py --method ref"')
	refPolicy = MLPGaussianActor(obs_dim=obsDim, act_dim=actDim, hidden_sizes=(256, 256), activation=nn.Tanh)
	refPolicy.load_state_dict(torch.load(refPolicyPath, map_location=device))
	refPolicy.to(device)
	refPolicy.eval()
	for param in refPolicy.parameters():
		param.requires_grad = False
	print(f'reference policy loaded from {refPolicyPath}')
	return refPolicy
