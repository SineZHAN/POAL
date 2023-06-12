import os
from torchvision import transforms
import random

args_pool = {'CIFAR10_04':
				{'n_epoch': 30, 
				 'transform_train': transforms.Compose([transforms.RandomCrop(size=32, padding=4),
    				transforms.RandomHorizontalFlip(), 
					transforms.ToTensor(),
				 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'loader_tr_args':{'batch_size': 128, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 1000, 'num_workers': 4},
				 'num_class':6,
				 'optimizer_args':{'lr': 0.001}},
			'CIFAR10_06':
				{'n_epoch': 30, 
				 'transform_train': transforms.Compose([transforms.RandomCrop(size=32, padding=4),
    				transforms.RandomHorizontalFlip(), 
					transforms.ToTensor(),
				 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
				 'loader_tr_args':{'batch_size': 128, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 1000, 'num_workers': 4},
				 'num_class':4,
				 'optimizer_args':{'lr': 0.001}},
			'CIFAR100_04':
				{'n_epoch': 40, 
				 'transform_train': transforms.Compose([transforms.RandomCrop(size=32, padding=4),
    				transforms.RandomHorizontalFlip(), 
					transforms.ToTensor(),
				 	transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
				 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
				 'loader_tr_args':{'batch_size': 128, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 1000, 'num_workers': 4},
				 'num_class':60,
				 'optimizer_args':{'lr': 0.001}},
			'CIFAR100_06':
				{'n_epoch': 40, 
				 'transform_train': transforms.Compose([transforms.RandomCrop(size=32, padding=4),
    				transforms.RandomHorizontalFlip(), 
					transforms.ToTensor(),
				 	transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
				 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
				 'loader_tr_args':{'batch_size': 128, 'num_workers': 4},
				 'loader_te_args':{'batch_size': 1000, 'num_workers': 4},
				 'num_class':40,
				 'optimizer_args':{'lr': 0.001}}
			}



