import numpy as np
from dataset import get_dataset, get_handler
from model_new import get_net
from torchvision import transforms

import torch


import warnings
import argparse
import sys
import os
import re
import random
import math
import datetime


import arguments
from parameters import *
from utils import *

from query_strategies import RandomSampling, EntropySampling, EntropySamplingIDEAL, POAL_PSES
# parameters
print(torch.cuda.is_available())
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed 
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset
STRATEGY_NAME = args_input.ALstrategy
MODEL_NAME = args_input.model

SEED = args_input.seed
os.environ['CUDA_VISIBLE_DEVICES'] = str(args_input.gpu)

torch.set_printoptions(profile='full')
#print(args_input.gpu)
#torch.cuda.set_device(args_input.gpu)

sys.stdout = Logger(os.path.abspath('') + '/../logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_' + MODEL_NAME + '_normal_log.txt')
warnings.filterwarnings('ignore')

args = args_pool[DATA_NAME]

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= False
torch.backends.cudnn.deterministic = True

# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
X_tr = X_tr
Y_tr = Y_tr

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('Number of labeled pool: {}'.format(NUM_INIT_LB))
print('Number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('Number of testing pool: {}'.format(n_test))
print('Batch size: {}'.format(NUM_QUERY))
print('Quota: {}'.format(NUM_ROUND*NUM_QUERY))
print('AL strategy: {}'.format(STRATEGY_NAME))
print('Dataset: {}'.format(DATA_NAME))
print('Model: {}'.format(MODEL_NAME))
print('\n')

iteration = args_input.iteration

# load network
net = get_net(DATA_NAME, MODEL_NAME, STRATEGY_NAME)
handler = get_handler(DATA_NAME)

#print(net)

all_acc = []
acq_time = []
all_ood_sample_num = []

while (iteration > 0):
	iteration = iteration - 1
	start = datetime.datetime.now()

	# generate initial labeled pool
	idxs_lb = np.zeros(n_pool, dtype=bool)
	idxs_tmp = np.arange(n_pool)
	np.random.shuffle(idxs_tmp)
	idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

	# only for special cases that need additional data
	new_X = torch.empty(0)
	new_Y = torch.empty(0)
	
	strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
	
	if STRATEGY_NAME == 'EntropySampling':
		strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
	elif STRATEGY_NAME == 'RandomSampling':
		strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
	elif STRATEGY_NAME == 'EntropySamplingIDEAL':
		strategy = EntropySamplingIDEAL(X_tr, Y_tr, idxs_lb, net, handler, args)
	elif STRATEGY_NAME == 'POAL_PSES':
		strategy = POAL_PSES(X_tr, Y_tr, idxs_lb, net, handler, args)
	else:
		print('No legal input of AL strategy, please try again.')
		sys.exit('sorry, goodbye!')
	
	
	# print info
	print(DATA_NAME)
	#print('RANDOM SEED {}'.format(SEED))
	print(type(strategy).__name__)
	
	ood_sample_num = np.zeros(NUM_ROUND+1)
	# round 0 accuracy
	
	ood_sample_num[0] = strategy.train()
	
	#strategy.train()
	P = strategy.predict(X_te, Y_te)
	acc = np.zeros(NUM_ROUND+1)
	acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
	
	print('Round 0\ntesting accuracy {}'.format(acc[0]))
	print('Round 0 ood sample num {}'.format(ood_sample_num[0]))
	print('\n')
	
	for rd in range(1, NUM_ROUND+1):
		print('Round {}'.format(rd))
		high_confident_idx = []
		high_confident_pseudo_label = []
		# query
		q_idxs = strategy.query(NUM_QUERY)
		idxs_lb[q_idxs] = True
	
		# update
		strategy.update(idxs_lb)

		ood_sample_num[rd] = strategy.train()
		print('ood sample num {}'.format(ood_sample_num[rd]))
		#strategy.train()
	
		# round accuracy
		P = strategy.predict(X_te, Y_te)
		acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
		
		print('testing accuracy {}'.format(acc[rd]))
	
	# print results
	#print('SEED {}'.format(SEED))
	#print(type(strategy).__name__)
	print('\n')
	print(acc)
	all_acc.append(acc)
	all_ood_sample_num.append(ood_sample_num)
	
	#save model
	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	model_path = './../modelpara/'+timestamp + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_' + MODEL_NAME  +'.params'
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds),3))
	torch.save(strategy.get_model().state_dict(), model_path)
	
# cal mean & standard deviation
acc_m = []
file_name_res_tot = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_' + MODEL_NAME  + '_normal_res_tot.txt'
file_res_tot =  open(os.path.join(os.path.abspath('') + '/../results', '%s' % file_name_res_tot),'w')

file_res_tot.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res_tot.writelines('model: {}'.format(MODEL_NAME) + '\n')
file_res_tot.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res_tot.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of testing pool: {}'.format(n_test) + '\n')
file_res_tot.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res_tot.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res_tot.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')

# result
for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print(str(i)+': '+str(acc_m[i]))
	file_res_tot.writelines(str(i)+': '+str(acc_m[i])+'\n')
mean_acc,stddev_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)

print('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc))
print('mean time: '+str(mean_time)+'. std dev time: '+str(stddev_time))

file_res_tot.writelines('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc)+'\n')
file_res_tot.writelines('mean time: '+str(mean_time)+'. std dev acc: '+str(stddev_time)+'\n')

# save result

file_name_res = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_' + MODEL_NAME  + '_normal_res.txt'
file_res =  open(os.path.join(os.path.abspath('') + '/../results', '%s' % file_name_res),'w')


file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('model: {}'.format(MODEL_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res.writelines('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB) + '\n')
file_res.writelines('number of testing pool: {}'.format(n_test) + '\n')
file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
avg_acc = np.mean(np.array(all_acc),axis=0)
for i in range(len(avg_acc)):
	tmp = 'Size of training set is ' + str(NUM_INIT_LB + i*NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
	file_res.writelines(tmp)

# ood sample selected

# save result

file_name_res_ood = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_' + MODEL_NAME  + '_normal_ood_num.txt'
file_res_ood =  open(os.path.join(os.path.abspath('') + '/../results', '%s' % file_name_res_ood),'w')


file_res_ood.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res_ood.writelines('model: {}'.format(MODEL_NAME) + '\n')
file_res_ood.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res_ood.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res_ood.writelines('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB) + '\n')
file_res_ood.writelines('number of testing pool: {}'.format(n_test) + '\n')
file_res_ood.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res_ood.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res_ood.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
avg_ood = np.mean(np.array(all_ood_sample_num),axis=0)
for i in range(len(avg_ood)):
	tmp = 'Size of training set is ' + str(NUM_INIT_LB + i*NUM_QUERY) + ', ' + 'the number of ood sample contained is ' + str(round(avg_ood[i],1)) + '.' + '\n'
	file_res_ood.writelines(tmp)

file_res.close()
file_res_tot.close()
file_res_ood.close()
