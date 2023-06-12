from torchvision import datasets, transforms
import sys
import os
import numpy as np
import math
import torch

# logger
class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum

def cat_two(x, y):
	if type(x) is np.ndarray:
		if type(y) is not np.ndarray:
			y = y.numpy().astype(x.dtype)
		return np.concatenate((x,y), axis = 0)
	else:
		if type(y) is np.ndarray:
			y = torch.from_numpy(y).type(x.dtype)
		return torch.cat((x, y), 0)





