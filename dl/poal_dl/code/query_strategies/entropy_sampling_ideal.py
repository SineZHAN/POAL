import numpy as np
import torch
from .strategy import Strategy

class EntropySamplingIDEAL(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(EntropySamplingIDEAL, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		Y = self.Y[idxs_unlabeled]
		inf_num = torch.tensor(-float('inf'))
		Y_id_ood = torch.ones(Y.shape[0])
		for i in range(Y.shape[0]):
			if Y[i] >= 0:
				Y_id_ood[i] = 1
			else:
				Y_id_ood[i] = inf_num
		U = U.mul(Y_id_ood)
		return idxs_unlabeled[U.sort()[1][:n]]
