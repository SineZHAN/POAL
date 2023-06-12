import copy
import os
import random
import warnings
import math
import numpy as np
import sys
import argparse

import scipy
from scipy.spatial.distance import mahalanobis

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.datasets import load_svmlight_file
from sklearn.utils.multiclass import type_of_target
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier


import matplotlib.pyplot as plt

from mmd import mmd_rbf

class Logger(object):
		def __init__(self, filename="Default.log"):
				self.terminal = sys.stdout
				self.log = open(filename, "a")

		def write(self, message):
				self.terminal.write(message)
				self.log.write(message)

		def flush(self):
				pass

def sigmoid_func(arr):
	arr_s =  [1.0/(1.0 + np.exp(-(x+1e-8))) - 0.5 for x in arr]
	return np.array(arr_s)

def form_dataset(dict_data, data_id):
	X=[]
	y=[]
	for i in data_id:
		X.append(dict_data[i][0])
		y.append(dict_data[i][1])
	return X,y

def cal_average(res_total):
		res = []
		print(np.array(res_total).shape)
		for i in range(len(res_total[0])):
				sum = 0.0
				for j in range(len(res_total)):
						sum = sum + res_total[j][i]
				sum = float((sum *1.0) / (len(res_total)*1.0))
				res.append(sum)
		return res

def test_model(labeled_X, labeled_y,test_X, test_y,clf = None):
		if clf == None:
				clf = SVC(kernel = 'rbf',gamma='auto')
		if len(set(test_y))>2:
				return test_model_multi(labeled_X, labeled_y,test_X, test_y,clf)
		else:
				return test_model_binary(labeled_X, labeled_y,test_X, test_y,clf)

def test_model_multi(labeled_X, labeled_y,test_X, test_y, clf):
		X_train = labeled_X
		X_test = test_X
		y_train = labeled_y
		y_test = test_y

		y_pred = clf.predict(X_test)

		acc = accuracy_score(y_test, y_pred) * 1.0
		auc = roc_auc_score(y_test, y_pred) * 1.0
		f1 = f1_score(y_test, y_pred) * 1.0

		return acc, auc, f1

def test_model_binary(labeled_X, labeled_y,test_X, test_y, clf):
		X_train = labeled_X
		X_test = test_X
		y_train = labeled_y
		y_test = test_y

		y_pred=clf.predict(X_test)
		acc = accuracy_score(y_test, y_pred) * 1.0

		idx = int(np.max(y_test))
		y_test1=np.zeros((len(y_pred),idx),dtype = 'int')
		y_pred1=np.zeros((len(y_pred),idx),dtype = 'int')

		for i in range(0,len(y_pred)):
				y_test1[i][int(y_test[i])-1] = 1
				y_pred1[i][int(y_pred[i])-1] = 1
		auc = roc_auc_score(y_test1,y_pred1, multi_class='ovr', average='macro')

		f1 = f1_score(y_test1, y_pred1,average = 'macro') * 1.0
		return acc, auc, f1

def get_mean_stddev(datax):
		return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
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

def readdata(train_path, test_path, n_labeled, max_id_class):
	# train
	train_X, train_y = load_svmlight_file(train_path)
	train_X = train_X.toarray().tolist()
	train_y = train_y.tolist()
	num_class = len(set(train_y))
	dict_data = dict()
	split_class = []
	for i in range(0, len(train_y)):
		if train_y[i] >= max_id_class:
			dict_data[i+1] = [train_X[i],np.nan]
			split_class.append(np.nan)
		else:
			dict_data[i+1] = [train_X[i],train_y[i]]
			split_class.append(int(train_y[i]))
	split_class = np.array(split_class)
	labeled_data = []
	unlabeled_data = []

	flag = False
	new_labeled_data = []
	new_corresponding_label = []
	full_set = list(range(1, len(train_X)+1))
	for i in range(max_id_class):
		new_idx = np.where(split_class == i)[0]
		#print(new_idx)
		random.shuffle(new_idx)
		labeled_data.append(new_idx[0]+1)
		labeled_data.append(new_idx[1]+1)
		full_set.remove(new_idx[0]+1)
		full_set.remove(new_idx[1]+1)
	new_idx = full_set
	random.shuffle(new_idx)
	remain_quota = n_labeled - len(labeled_data)
	labeled_data = labeled_data + new_idx[0:remain_quota]
	unlabeled_data = new_idx[remain_quota:]

	# test
	test_X, test_y = load_svmlight_file(test_path)
	test_X = test_X.toarray().tolist()
	test_y = test_y.tolist()
	for i in range(len(train_X), len(train_X)+len(test_X)):
		dict_data[i+1] = [test_X[i-len(train_X)],test_y[i-len(train_X)]]
	test_data = list(range(len(train_X)+1, len(train_X)+len(test_X)+1))
	return dict_data, labeled_data, test_data, unlabeled_data

def caculate_mean_cov(X):
	mean_X = X.mean()
	cov_X = np.cov(X)
	return mean_X, cov_X

def gmm_fit(X):
	lowest_bic = np.infty
	bic = []
	best_gmm = None
	n_components_range = range(1, min(10, len(X)))
	for n_components in n_components_range:
		# Fit a Gaussian mixture with EM
		gmm = GaussianMixture(n_components=n_components)
		gmm.fit(X)
		bic.append(gmm.bic(X))
		if bic[-1] < lowest_bic:
			lowest_bic = bic[-1]
			best_gmm = gmm
	if best_gmm == None:
		return [], [], []
	else:
		return best_gmm, best_gmm.means_, best_gmm.covariances_

def gmm_clustering(X, y):
	y = np.array(y)
	X = np.array(X)
	n_classes = len(np.unique(y))
	gmm = []
	gmm_means = []
	gmm_cov = []
	for i in range(n_classes - 1):
		gmm_t, means, cov = gmm_fit(X[y == i])
		gmm.append(gmm_t)
		for tx in means:
			gmm_means.append(tx)
		for ty in cov:
			gmm_cov.append(ty)
	return gmm, gmm_means, gmm_cov


def uncertainty_sampling(X, clf):
	if not hasattr(clf, 'predict_proba'):
		raise Exception('model object must implement predict_proba methods in current algorithm.')
	predict = clf.predict_proba(X)
	pv = np.asarray(predict)
	pv[pv <= 0] = 1e-06  # avoid zero division
	entro = [-np.sum(vec * np.log(vec+1e-9)) for vec in pv]
	return entro

def select_poal(infor_value1, infor_value2, costs, budget):
	"""
	POAL (Pareto Optimization for Active Learning Subset Selection) method.
	Paremeters:
	----------
	infor_value1 and infor_value2: array-like
		The values (objectives) corresponding to each item.
	costs: array-like
		The cost corresponding to each item.
	Budget: float
		the constraint on the cost of selected variables.
	Returns:
	----------
	top: array-like
		selections.
	References
	----------
	[1] Chao Qian, Yang Yu, and Zhi-Hua Zhou.
		Subset selection by pareto optimization. In Advances
		in Neural Information Processing Systems, pages 1774-
		1782, 2015.
	"""
	optimal_set = []
	assert(len(infor_value1) == len(infor_value2))
	numbers = len(infor_value1)
	#init
	optimal_set = normal_pareto_optimization(infor_value1, infor_value2)

	#add
	while len(optimal_set) < 5 * budget:
		#new index
		index = [i for i in range(numbers)]
		new_index = [x for x in index if x not in optimal_set]
		#new infor_value
		infor_value1i = [infor_value1[i] for i in new_index]
		infor_value2i = [infor_value2[i] for i in new_index]
		#new optimal_set
		optimal_set_tmp = normal_pareto_optimization(infor_value1i, infor_value2i)
		optimal_seti = [new_index[i] for i in optimal_set_tmp]
		optimal_set.extend(optimal_seti)

	if len(optimal_set) == budget:
		return optimal_set

	#new infor_value
	infor_value1_new = [infor_value1[i] for i in optimal_set]
	infor_value2_new = [infor_value2[i] for i in optimal_set]

	# select subset
	_, query_index = select_POSS_intersection(infor_value1_new,  infor_value2_new, None, budget)
	top = [optimal_set[i] for i in range(len(query_index)) if query_index[i] == 1]


	return top

def select_poal_early_stopping(infor_value1, infor_value2, costs, budget):
	optimal_set = []
	assert(len(infor_value1) == len(infor_value2))
	numbers = len(infor_value1)
	#init
	optimal_set = normal_pareto_optimization(infor_value1, infor_value2)

	#add
	while len(optimal_set) < 4 * budget:
		#new index
		index = [i for i in range(numbers)]
		new_index = [x for x in index if x not in optimal_set]
		#new infor_value
		infor_value1i = [infor_value1[i] for i in new_index]
		infor_value2i = [infor_value2[i] for i in new_index]
		#new optimal_set
		optimal_set_tmp = normal_pareto_optimization(infor_value1i, infor_value2i)
		optimal_seti = [new_index[i] for i in optimal_set_tmp]
		optimal_set.extend(optimal_seti)

	if len(optimal_set) == budget:
		return optimal_set

	#new infor_value
	infor_value1_new = [infor_value1[i] for i in optimal_set]
	infor_value2_new = [infor_value2[i] for i in optimal_set]

	# select subset
	_, query_index = select_POSS_intersection_early_stopping(infor_value1_new,  infor_value2_new, None, budget)
	top = [optimal_set[i] for i in range(len(query_index)) if query_index[i] == 1]


	return top



def select_POSS_intersection_early_stopping(infor_value1, infor_value2, costs, budget):
	if costs == None:
		costs = np.ones(len(infor_value1))
	assert(len(infor_value1) == len(costs) == len(infor_value2))
	num = len(infor_value1)
	population = np.zeros((1, num))

	popSize = 1
	fitness = np.zeros((1, 5))
	fitness[0][0] = -np.infty
	fitness[0][1] = -np.infty
	fitness[0][2] = 0.

	fitness[0][3] = np.infty
	fitness[0][4] = np.infty

	#pop_information = [T, f, mean, std]
	pop_information = []
	# repeat to improve the population; 
	# the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis of POSS, can set others.
	T = int(2 * np.e * np.power(budget, 2) * num)

	for round in np.arange(T):
		# randomly select a solution from the population and mutate it to generate a new solution.
		#offspring = np.abs(population[np.random.randint(0, popSize), :] - np.random.choice([1, 0], size=(num), p=[1/num, 1 - 1/num]))
		offspring = np.zeros(len(infor_value1))
		offspring[random.sample(range(0,len(infor_value1)), budget)] = 1
		# compute the fitness of the new solution.
		offspringFit = np.array([0., 0., 0., 0., 0.])
		offspringFit[2] = np.sum(offspring * costs)

		if offspringFit[2] == 0 or offspringFit[2] > budget:
			offspringFit[0] = -np.infty
			offspringFit[1] = -np.infty
			offspringFit[3] = np.infty
			offspringFit[4] = np.infty
		else:
			offspringFit[0] = np.sum(offspring * infor_value1)
			offspringFit[1] = np.sum(offspring * infor_value2)
			a = (offspring * infor_value1)[np.nonzero(offspring * infor_value1)]
			b = (offspring * infor_value2)[np.nonzero(offspring * infor_value2)]
			offspringFit[3] = a.min() if len(a) > 0 else 0 
			offspringFit[4] = b.min() if len(b) > 0 else 0 

		# use the new solution to update the current population.
		# if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
		judge1 = np.array(fitness[0: popSize, 0] >= offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])
		judge2 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] >= offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])  
		judge3 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] <= offspringFit[2])

		c= judge1 | judge2 
		if c.any():
			continue
		else:
			# deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
			index = [i for i in range(len(fitness))]
			condi_1 = np.where(fitness[0: popSize, 0] <= offspringFit[0])
			condi_2 = np.where(fitness[0: popSize, 1] <= offspringFit[1])
			condi_3 = np.where(fitness[0: popSize, 2] >= offspringFit[2])
			deleteIndex = [val for val in condi_1[0] if val in condi_2[0] and val in condi_3[0]] 
			nodeleteIndex = [j for j in index if j not in deleteIndex]	
	   
		# ndelete: record the index of the solutions to be kept.
		population = np.row_stack((population[nodeleteIndex, :], offspring))
		fitness = np.row_stack((fitness[nodeleteIndex, :], offspringFit))
		popSize = len(nodeleteIndex) + 1

		# early stopping condition
		if round > 0 and round % 200 == 0:
			# get current pareto set information
			f = []
			tmp = np.where(fitness[:, 2] == budget)[0]
			for tmp_idx in tmp:
				f.append(np.array([fitness[tmp_idx][0], fitness[tmp_idx][1]]))

			# get MMD 
			if len(pop_information)>0:
				mmd_information = []
				mean = 0
				std = 0
				for p in pop_information:
					t_f = p[1]
					mmd_t = mmd_rbf(np.array(f), np.array(t_f)) 
					mmd_information.append(mmd_t)
					mean = np.mean(np.array(mmd_t))
					std = np.std(np.array(mmd_t))
				pop_information.append([round,f,mean,std])

			else:
				pop_information.append([round, f, 0.0, 0.0])
			if len(pop_information)>10:
				del(pop_information[0])
			# early stopping
			m = np.array(pop_information)[:,2]
			s = np.array(pop_information)[:,3]
			d_m = np.max(m) - np.min(m)
			d_s = np.max(s) - np.min(s)

			if len(pop_information) >=10 and d_m <= 0.01 and d_s <= 0.01:
				print('!!!')
				#print(pop_information)
				print(round, T)
				break
	temp = np.where(fitness[:, 2] <= budget)[0]
	temp2 = np.where(fitness[:, 2] == budget)[0]
	count_sum = np.zeros(len(infor_value1))

	for t in temp2:
		x = population[t,:]
		count_sum = count_sum + x
	count_inter = np.zeros(len(temp2))

	for t_idx in range(len(temp2)):
		x = population[temp2[t_idx],:]
		count_intersection = 0
		count_intersection = np.dot(x, count_sum)
		count_inter[t_idx] = count_intersection
	max_info_indx = np.argmax(count_inter)
	max_info_indx = temp2[max_info_indx]
	max_infovalue = [fitness[max_info_indx][0]]
	selectedVariables = population[max_info_indx, :]

	return max_infovalue, selectedVariables

def normal_pareto_optimization(infor_value1, infor_value2):
	assert(len(infor_value1) == len(infor_value2))
	numbers = len(infor_value1)
	optimal_set = []
	optimal_set_value1 = []
	optimal_set_value2 = []

	for i in range(numbers):
		if i == 0:
			optimal_set.append(i)
			optimal_set_value1.append(infor_value1[i])
			optimal_set_value2.append(infor_value2[i])
		else:
			judge1 = np.array(optimal_set_value1 >= infor_value1[i]) & np.array(optimal_set_value2 > infor_value2[i])
			judge2 = np.array(optimal_set_value1 > infor_value1[i]) & np.array(optimal_set_value2 >= infor_value2[i])
			c= judge1 | judge2 
			if c.any():
				continue
			else:
				# deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
				index = [i for i in range(len(optimal_set))]
				condi_1 = np.where(optimal_set_value1 <= infor_value1[i])
				condi_2 = np.where(optimal_set_value2 <= infor_value2[i])
				deleteIndex = [val for val in condi_1[0] if val in condi_2[0]] 
				nodeleteIndex = [x for x in index if x not in deleteIndex]	
				nodeleteIndex.append(i)
	   
			# ndelete: record the index of the solutions to be kept.
			optimal_set = []
			optimal_set_value1 = []
			optimal_set_value2 = []
			for x in nodeleteIndex:
				optimal_set.append(x)
				optimal_set_value1.append(infor_value1[x])
				optimal_set_value2.append(infor_value2[x])
	return optimal_set


def select_POSS(infor_value1, infor_value2, costs, budget):
	if costs == None:
		costs = np.ones(len(infor_value1))
	assert(len(infor_value1) == len(costs) == len(infor_value2))
	num = len(infor_value1)
	population = np.zeros((1, num))

	popSize = 1
	fitness = np.zeros((1, 5))
	fitness[0][0] = -np.infty
	fitness[0][1] = -np.infty
	fitness[0][2] = 0.
	fitness[0][3] = np.infty
	fitness[0][4] = np.infty
	# repeat to improve the population; 
	# the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
	T = 2 * np.e * np.power(budget, 2) * num

	for round in np.arange(T):
		# randomly select a solution from the population and mutate it to generate a new solution.
		#offspring = np.abs(population[np.random.randint(0, popSize), :] - np.random.choice([1, 0], size=(num), p=[1/num, 1 - 1/num]))
		offspring = np.zeros(len(infor_value1))
		offspring[random.sample(range(0,len(infor_value1)), budget)] = 1
		# compute the fitness of the new solution.
		offspringFit = np.array([0., 0., 0., 0., 0.])
		offspringFit[2] = np.sum(offspring * costs)

		if offspringFit[2] == 0 or offspringFit[2] > budget:
			offspringFit[0] = -np.infty
			offspringFit[1] = -np.infty
			offspringFit[3] = np.infty
			offspringFit[4] = np.infty
		else:
			offspringFit[0] = np.sum(offspring * infor_value1)
			offspringFit[1] = np.sum(offspring * infor_value2)
			a = (offspring * infor_value1)[np.nonzero(offspring * infor_value1)]
			b = (offspring * infor_value2)[np.nonzero(offspring * infor_value2)]
			offspringFit[3] = a.min() if len(a) > 0 else 0 
			offspringFit[4] = b.min() if len(b) > 0 else 0 

		# use the new solution to update the current population.
		# if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
		judge1 = np.array(fitness[0: popSize, 0] >= offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])
		judge2 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] >= offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])  
		judge3 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] <= offspringFit[2])

		c= judge1 | judge2 
		if c.any():
			continue
		else:
			# deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
			index = [i for i in range(len(fitness))]
			condi_1 = np.where(fitness[0: popSize, 0] <= offspringFit[0])
			condi_2 = np.where(fitness[0: popSize, 1] <= offspringFit[1])
			condi_3 = np.where(fitness[0: popSize, 2] >= offspringFit[2])
			deleteIndex = [val for val in condi_1[0] if val in condi_2[0] and val in condi_3[0]] 
			nodeleteIndex = [j for j in index if j not in deleteIndex]	
	   
		# ndelete: record the index of the solutions to be kept.
		population = np.row_stack((population[nodeleteIndex, :], offspring))
		fitness = np.row_stack((fitness[nodeleteIndex, :], offspringFit))
		popSize = len(nodeleteIndex) + 1

	temp = np.where(fitness[:, 2] <= budget)[0]
	temp2 = np.where(fitness[:, 2] == budget)[0]

	max_info_indx = np.argmax(fitness[temp2, 0])
	max_info_indx = temp2[max_info_indx]
	max_infovalue = [fitness[max_info_indx][0]]
	selectedVariables = population[max_info_indx, :]

	return max_infovalue, selectedVariables



def select_POSS_intersection(infor_value1, infor_value2, costs, budget):
	if costs == None:
		costs = np.ones(len(infor_value1))
	assert(len(infor_value1) == len(costs) == len(infor_value2))
	num = len(infor_value1)
	population = np.zeros((1, num))

	popSize = 1
	fitness = np.zeros((1, 5))
	fitness[0][0] = -np.infty
	fitness[0][1] = -np.infty
	fitness[0][2] = 0.
	fitness[0][3] = np.infty
	fitness[0][4] = np.infty
	# repeat to improve the population; 
	# the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
	T = 2 * np.e * np.power(budget, 2) * num

	for round in np.arange(T):
		# randomly select a solution from the population and mutate it to generate a new solution.
		#offspring = np.abs(population[np.random.randint(0, popSize), :] - np.random.choice([1, 0], size=(num), p=[1/num, 1 - 1/num]))
		offspring = np.zeros(len(infor_value1))
		offspring[random.sample(range(0,len(infor_value1)), budget)] = 1
		# compute the fitness of the new solution.
		offspringFit = np.array([0., 0., 0., 0., 0.])
		offspringFit[2] = np.sum(offspring * costs)

		if offspringFit[2] == 0 or offspringFit[2] > budget:
			offspringFit[0] = -np.infty
			offspringFit[1] = -np.infty
			offspringFit[3] = np.infty
			offspringFit[4] = np.infty
		else:
			offspringFit[0] = np.sum(offspring * infor_value1)
			offspringFit[1] = np.sum(offspring * infor_value2)
			a = (offspring * infor_value1)[np.nonzero(offspring * infor_value1)]
			b = (offspring * infor_value2)[np.nonzero(offspring * infor_value2)]
			offspringFit[3] = a.min() if len(a) > 0 else 0 
			offspringFit[4] = b.min() if len(b) > 0 else 0 

		# use the new solution to update the current population.
		# if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
		judge1 = np.array(fitness[0: popSize, 0] >= offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])
		judge2 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] >= offspringFit[1]) & np.array(fitness[0: popSize, 2] < offspringFit[2])  
		judge3 = np.array(fitness[0: popSize, 0] > offspringFit[0]) & np.array(fitness[0: popSize, 1] > offspringFit[1]) & np.array(fitness[0: popSize, 2] <= offspringFit[2])

		c= judge1 | judge2 
		if c.any():
			continue
		else:
			# deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
			index = [i for i in range(len(fitness))]
			condi_1 = np.where(fitness[0: popSize, 0] <= offspringFit[0])
			condi_2 = np.where(fitness[0: popSize, 1] <= offspringFit[1])
			condi_3 = np.where(fitness[0: popSize, 2] >= offspringFit[2])
			deleteIndex = [val for val in condi_1[0] if val in condi_2[0] and val in condi_3[0]] 
			nodeleteIndex = [j for j in index if j not in deleteIndex]	
	   
		# ndelete: record the index of the solutions to be kept.
		population = np.row_stack((population[nodeleteIndex, :], offspring))
		fitness = np.row_stack((fitness[nodeleteIndex, :], offspringFit))
		popSize = len(nodeleteIndex) + 1

	temp = np.where(fitness[:, 2] <= budget)[0]
	temp2 = np.where(fitness[:, 2] == budget)[0]
	count_sum = np.zeros(len(infor_value1))

	for t in temp2:
		x = population[t,:]
		count_sum = count_sum + x
	count_inter = np.zeros(len(temp2))

	for t_idx in range(len(temp2)):
		x = population[temp2[t_idx],:]
		count_intersection = 0
		count_intersection = np.dot(x, count_sum)
		count_inter[t_idx] = count_intersection
	max_info_indx = np.argmax(count_inter)
	max_info_indx = temp2[max_info_indx]
	max_infovalue = [fitness[max_info_indx][0]]
	selectedVariables = population[max_info_indx, :]

	return max_infovalue, selectedVariables

def get_minimum_maha_dist(mean, cov, unlabeled_sample):
	num_components = len(mean)
	num_unlabeled_samples = len(unlabeled_sample)
	maha_dis_score = np.zeros((num_unlabeled_samples, num_components))
	for i in range(0, num_unlabeled_samples):
		for j in range(0, num_components):
			maha_dis_score[i][j] = mahalanobis(unlabeled_sample[i], mean[j], cov[j])
	return np.min(maha_dis_score, axis = 1)


def main():

	parser = argparse.ArgumentParser(description='OODAL + pareto optimal for subset selection')
	parser.add_argument('--repeat', '-r', default=100, type=int, help='repeat trials')
	parser.add_argument('--mode', '-m', default='entropy', type=str, help='mode of experiments')
	#parser.add_argument('--name', '-n', default='entropy', type=str, help='name of saved files')
	parser.add_argument('--batch', '-b', default=10, type=int, help='batch size')
	parser.add_argument('--quota', '-q', default=500, type=int, help='quota')
	parser.add_argument('--dname', '-d', default='toy_data', type=str, help='name of dataset')
	parser.add_argument('--init', '-i', default=20, type=int, help='initial size of labeled set')
	parser.add_argument('--category', '-c', default=2, type=int, help='number of categories of ID data')
	parser.add_argument('--model', '-f', default='GPC', type=str, help='basic model')

	args = parser.parse_args()

	warnings.filterwarnings('ignore')
	dataset_name = args.dname
	train_path = os.path.join(os.path.abspath('./../') + '/data/', '%s_train.txt' % dataset_name)
	print(train_path)
	test_path = os.path.join(os.path.abspath('./../') + '/data/', '%s_test.txt' % dataset_name)
	print(test_path)




	cnt = 0
	random.seed(4666)
	repeat = args.repeat
	tot_acc = []
	tot_count_ood = []
	tot_cnt_ood = []
	tot_count_labeled_set = []
	while(cnt < repeat):
		cnt = cnt + 1
		print('Repeat time of experiment', cnt)
		
		dict_data=dict()
		labeled_data=[]
		test_data=[]
		unlabeled_data=[]

		dict_data,labeled_data,test_data,unlabeled_data = readdata(train_path, test_path, args.init, args.category)

		acc = []
		count_ood = []


		dict_train_idx=labeled_data+unlabeled_data

		kernel = 1.0 * RBF(1.0)
		if args.model == 'GPC':
			clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
		elif args.model == 'LR':
			clf = LogisticRegression(random_state=0)
		elif args.model == 'MLP':
			clf = MLPClassifier(random_state=1, max_iter=250, hidden_layer_sizes=(5,2))
		else:
			raise NotImplementedError
		test_X, test_y = form_dataset(dict_data, test_data)
		
		
		batch_size  = args.batch
		count = 0
		round_iter = 0
		count_labeled_set = args.init
		cnt_labeled_set = []
		cnt_labeled_set.append(args.init)

		iter_num = int(args.quota/args.batch)
		for round_iter in range(0, iter_num):
			print('round', round_iter)
			#round_iter = round_iter + 1
			current_label_X, current_label_y = form_dataset(dict_data, labeled_data)
			trn_X = []
			trn_y = []
			for i in range(len(current_label_y)):

				if not np.isnan(current_label_y[i]):
					trn_X.append(current_label_X[i])
					trn_y.append(current_label_y[i])
				else:
					if round_iter == 0:
						count = count + 1
			if round_iter == 0:
				count_ood.append(count)
			clf.fit(trn_X, trn_y)
			test_pred = clf.predict(test_X)
			test_acc = round(accuracy_score(test_y, test_pred) * 1.0, 3)
			print('test accuracy', test_acc)
			acc.append(test_acc)

			current_unlabel_X, current_unlabel_y = form_dataset(dict_data, unlabeled_data)
			print('size of unlabeled data pool', len(current_unlabel_X))

			
			gmm, mean, cov = gmm_clustering(current_label_X, current_label_y)
			entropy = None
			maha_min_dis = None
			maha_min_dis = get_minimum_maha_dist(mean, cov, current_unlabel_X)
			entropy = uncertainty_sampling(current_unlabel_X, clf)


			top = []
			if args.mode == 'non_ood_entropy':
				entropy = uncertainty_sampling(current_unlabel_X, clf)
				tmp = np.argsort(-np.array(entropy))
				tmp2 = []
				tmp3 = []
				for idx in tmp:
					if not np.isnan(current_unlabel_y[idx]):
						tmp2.append(idx)
					else:
						tmp3.append(idx)
				tmp2.extend(tmp3)
				top = np.array(tmp2)[0:batch_size]
			elif args.mode == 'entropy':
				entropy = uncertainty_sampling(current_unlabel_X, clf)
				top = np.argsort(-np.array(entropy))[0:batch_size]
			elif args.mode == 'random':
				new_idx = list(range(0,len(unlabeled_data)))
				random.shuffle(new_idx)
				top = np.array(new_idx)[0:batch_size]
			elif args.mode == 'maha':
				maha_min_dis = get_minimum_maha_dist(mean, cov, current_unlabel_X)
				top = np.argsort(np.array(maha_min_dis))[0:batch_size]
			elif args.mode == 'poal':
				entropy = uncertainty_sampling(current_unlabel_X, clf)
				maha_min_dis = get_minimum_maha_dist(mean, cov, current_unlabel_X)
				top = select_poal(np.max(np.array(maha_min_dis)) - np.array(maha_min_dis), entropy, None, batch_size)
			elif args.mode == 'poal_early_stopping':
				entropy = uncertainty_sampling(current_unlabel_X, clf)
				maha_min_dis = get_minimum_maha_dist(mean, cov, current_unlabel_X)
				top = select_poal_early_stopping(np.max(np.array(maha_min_dis)) - np.array(maha_min_dis), entropy, None, batch_size)
			else:
				raise NotImplementedError

			count_labeled_set = count_labeled_set + len(top)
			cnt_labeled_set.append(count_labeled_set)
			count_tmp = 0
			for i in top:
				if np.isnan(current_unlabel_y[i]):
					count_tmp = count_tmp + 1
					count = count + 1
			count_ood.append(count_tmp)
			
			top_idx = [unlabeled_data[i] for i in top]
			labeled_data.extend(top_idx)
			for i in top_idx:
				unlabeled_data.remove(i)

		# last round
		current_label_X, current_label_y = form_dataset(dict_data, labeled_data)
		trn_X = []
		trn_y = []
		for i in range(len(current_label_y)):
			if not np.isnan(current_label_y[i]):
				trn_X.append(current_label_X[i])
				trn_y.append(current_label_y[i])
		clf.fit(trn_X, trn_y)
		test_pred = clf.predict(test_X)
		test_acc = round(accuracy_score(test_y, test_pred) * 1.0, 3)
		print('test accuracy', test_acc)
		acc.append(test_acc)
		print('ood samples selected', count)
		tot_acc.append(acc)
		tot_count_ood.append(count_ood)
		print(count_ood)
		tot_cnt_ood.append(count)
		tot_count_labeled_set.append(cnt_labeled_set)

	acc = np.mean(tot_acc, axis = 0)
	count_ood_num = np.mean(tot_count_ood, axis = 0)
	count_labeled_set = np.mean(tot_count_labeled_set, axis = 0)
	print(len(acc))
	print(len(count_ood_num))
	print(len(count_labeled_set))
	file_name_res =  args.dname + '_' + args.mode + '_' + args.model + '_' + str(args.init) +'_' + str(args.batch) + '_' + str(args.quota) + '.txt'
	file_name_count_ood = args.dname +'_' + args.mode+ '_' + args.model + '_' + str(args.init) +'_' + str(args.batch) + '_' + str(args.quota) + '_oodnum.txt'
	file_res = open(os.path.join(os.path.abspath('./../') + '/result10/', '%s' % file_name_res),'w')
	file_res_ood = open(os.path.join(os.path.abspath('./../') + '/result10/', '%s' % file_name_count_ood),'w')
	for i in range(len(acc)):
		tmp0 = 'Size of training set is '+str(int(count_labeled_set[i])) +', the accuracy is ' + str(acc[i]) + '\n'
		file_res.writelines(tmp0)
	file_res.close()
	for i in range(len(count_ood_num)):
		tmp1 = 'Size of training set is '+str(int(count_labeled_set[i])) +', the number of OOD samples is ' + str(count_ood_num[i]) + '\n'
		file_res_ood.writelines(tmp1)
	file_res_ood.writelines(str(np.average(tot_cnt_ood)) + '\n')
	file_res.close()
	file_res_ood.close()
	print(np.average(tot_cnt_ood))



if __name__ == '__main__':
	main()




