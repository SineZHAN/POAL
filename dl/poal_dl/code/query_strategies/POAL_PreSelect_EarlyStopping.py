import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .strategy import Strategy
from copy import deepcopy
import random

class POAL_PSES(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(POAL_PSES, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		X = self.X[idxs_unlabeled]
		Y = self.Y[idxs_unlabeled]
		U = self.entropy(X,Y)
		maha_score = self.maha(X,Y)
		top = select_poal(U.tolist(), maha_score.tolist(), None, n)
		return idxs_unlabeled[top]

	def entropy(self, X, Y):
		probs = self.predict_prob(X, Y)
		log_probs = torch.log(probs)
		# the larger the better
		U = (-1.0 * probs * log_probs).sum(1)
		return U
	
	def maha(self, X, Y):

		# get id train loader
		idxs_train = np.arange(self.n_pool)[self.idxs_lb]
		X_train_full = self.X[idxs_train]
		Y_train_full = self.Y[idxs_train]
		a = list(range(Y_train_full.shape[0]))
		b = torch.where(Y_train_full<0)[0].numpy()
		d = sorted(list(set(a).difference(set(b))))
		Y_train = torch.index_select(Y_train_full, 0, torch.tensor(d))
		if type(X_train_full) is np.ndarray:
			tmp = deepcopy(X_train_full)
			tmp = torch.from_numpy(tmp)
			X_train = torch.index_select(tmp, 0, torch.tensor(d))
			X_train = X_train.numpy().astype(X_train_full.dtype)
		else:
			X_train = torch.index_select(X_train_full, 0, torch.tensor(d))
		
		train_loader = DataLoader(self.handler(X_train, Y_train, transform=self.args['transform_train']),
							**self.args['loader_tr_args'])

		# set feature_list shape
		model = self.get_model()
		temp_x = torch.rand(2, X_train[0].shape[2], X_train[0].shape[0], X_train[0].shape[1]).to(self.device)
		temp_x = Variable(temp_x)
		temp_list = model.feature_list(temp_x)[1]
		num_output = len(temp_list)
		feature_list = np.empty(num_output)
		count = 0
		for out in temp_list:
			feature_list[count] = out.size(1)
			count += 1

		sample_mean, sample_cov = self.sample_estimator(model, self.args['num_class'], feature_list, train_loader)
		# get mahalanobis score
		test_loader = DataLoader(self.handler(X, Y, transform=self.args['transform']),
							shuffle=False, **self.args['loader_te_args'])

		for i in range(num_output):
			M_score = self.get_Mahalanobis_score(model, test_loader, self.args['num_class'], sample_mean, sample_cov, i)
			if i == 0:
				Maha_score = M_score.reshape(M_score.shape[0], -1)
			else:
				Maha_score = np.concatenate((Maha_score, M_score.reshape((M_score.shape[0], -1))), axis=1)
		Maha_score = np.asarray(Maha_score, dtype = np.float32)

		# the smaller the better
		Maha_avg_score = np.mean(Maha_score, axis = 1)

		inv_Maha_avg_score = np.max(Maha_avg_score) - Maha_avg_score
		return inv_Maha_avg_score
		
	def get_Mahalanobis_score(self, model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude = 0.01):
		'''
		Compute the proposed Mahalanobis confidence score on input dataset
		return: Mahalanobis score from layer_index
		'''
		model.eval()
		Mahalanobis = []
		
		
		for data, target, idx in test_loader:
			
			data, target = data.to(self.device), target.to(self.device)
			data, target = Variable(data, requires_grad = True), Variable(target)
			
			out_features = model.intermediate_forward(data, layer_index)
			out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
			out_features = torch.mean(out_features, 2)
			
			# compute Mahalanobis score
			gaussian_score = 0
			for i in range(num_classes):
				batch_sample_mean = sample_mean[layer_index][i]
				zero_f = out_features.data - batch_sample_mean
				term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
				if i == 0:
					gaussian_score = term_gau.view(-1,1)
				else:
					gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
			
			# Input_processing
			sample_pred = gaussian_score.max(1)[1]
			batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
			zero_f = out_features - Variable(batch_sample_mean)
			pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
			loss = torch.mean(-pure_gau)
			loss.backward()
			
			gradient =  torch.ge(data.grad.data, 0)
			gradient = (gradient.float() - 0.5) * 2
			gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).to(self.device)) / (0.2023))
			gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).to(self.device)) / (0.1994))
			gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).to(self.device)) / (0.2010))
			tempInputs = torch.add(data.data, -magnitude, gradient)
	
			noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
			noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
			noise_out_features = torch.mean(noise_out_features, 2)
			noise_gaussian_score = 0
			for i in range(num_classes):
				batch_sample_mean = sample_mean[layer_index][i]
				zero_f = noise_out_features.data - batch_sample_mean
				term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
				if i == 0:
					noise_gaussian_score = term_gau.view(-1,1)
				else:
					noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)	  

			noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
			Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

		return np.array(Mahalanobis)

	def sample_estimator(self,model, num_classes, feature_list, train_loader):
		"""
		compute sample mean and precision (inverse of covariance)
		return: sample_class_mean: list of class mean
				precision: list of precisions
		"""
		import sklearn.covariance
		
		model.eval()
		group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

		num_output = len(feature_list)
		num_sample_per_class = np.empty(num_classes)
		num_sample_per_class.fill(0)
		list_features = []
		for i in range(num_output):
			temp_list = []
			for j in range(num_classes):
				temp_list.append(0)
			list_features.append(temp_list)
		
		for data, target,idx in train_loader:
			data = data.to(self.device)
			data = Variable(data, volatile=True)
			output, out_features = model.feature_list(data)
			
			# get hidden features
			for i in range(num_output):
				out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
				out_features[i] = torch.mean(out_features[i].data, 2)
				
			
			# construct the sample matrix
			for i in range(data.size(0)):
				label = target[i]
				if num_sample_per_class[label] == 0:
					out_count = 0
					for out in out_features:
						list_features[out_count][label] = out[i].view(1, -1)
						out_count += 1
				else:
					out_count = 0
					for out in out_features:
						list_features[out_count][label] \
						= torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
						out_count += 1				
				num_sample_per_class[label] += 1
				
		sample_class_mean = []
		out_count = 0
		for num_feature in feature_list:
			temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
			for j in range(num_classes):
				temp_list[j] = torch.mean(list_features[out_count][j], 0)
			sample_class_mean.append(temp_list)
			out_count += 1
			
		precision = []
		for k in range(num_output):
			X = 0
			for i in range(num_classes):
				if i == 0:
					X = list_features[k][i] - sample_class_mean[k][i]
				else:
					X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
					
			# find inverse			
			group_lasso.fit(X.cpu().numpy())
			temp_precision = group_lasso.precision_
			temp_precision = torch.from_numpy(temp_precision).float().cuda()
			precision.append(temp_precision)
			
		return sample_class_mean, precision

def select_poal(infor_value1, infor_value2, costs, budget):
	optimal_set = []
	assert(len(infor_value1) == len(infor_value2))
	numbers = len(infor_value1)
	#init
	optimal_set = normal_pareto_optimization(infor_value1, infor_value2)

	# 6 * budget can change
	while len(optimal_set) < 6 * budget:
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

	T = int(budget*num)
	print(T)

	# flag stop iteration
	Flag = False
	count = 0
	last_result_sum = 0
    #pop_information = [T, f, mean, std]
	pop_information = []
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
				print(round, T)
				break


	temp = np.where(fitness[:, 2] <= budget)[0]
	temp2 = np.where(fitness[:, 2] == budget)[0]
	count_sum = np.zeros(len(infor_value1))

	for t in temp2:
		x = population[t,:]
		count_sum = count_sum + x
	#print(count_sum)
	count_inter = np.zeros(len(temp2))

	for t_idx in range(len(temp2)):
		x = population[temp2[t_idx],:]
		count_intersection = 0
		count_intersection = np.dot(x, count_sum)
		count_inter[t_idx] = count_intersection
	#print(count_inter)
	max_info_indx = np.argmax(count_inter)
	max_info_indx = temp2[max_info_indx]
	max_infovalue = [fitness[max_info_indx][0]]
	selectedVariables = population[max_info_indx, :]

	return max_infovalue, selectedVariables

def normal_pareto_optimization(infor_value1, infor_value2):
	assert(len(infor_value1) == len(infor_value2))
	numbers = len(infor_value1)
	infor_value1 = np.array(infor_value1)
	infor_value2 = np.array(infor_value2)
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


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

	n_samples = int(source.size()[0])+int(target.size()[0])
	total = torch.cat([source, target], dim=0)

	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
	L2_distance = ((total0-total1)**2).sum(2) 
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
	bandwidth /= kernel_mul ** (kernel_num // 2)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
	return sum(kernel_val)

def mmd_rbf(sourcex, targetx, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	source=torch.from_numpy(sourcex)
	target=torch.from_numpy(targetx)
	source_size = int(source.size()[0])
	target_size = int(target.size()[0])
	kernels = guassian_kernel(source, target,
		kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
	XX = kernels[:source_size, :source_size]
	YY = kernels[target_size:, target_size:]
	XY = kernels[:source_size, target_size:]
	YX = kernels[target_size:, :source_size]
	loss = torch.mean(XX + YY - XY -YX)

	return loss.numpy()