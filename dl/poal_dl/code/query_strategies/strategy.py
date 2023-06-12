import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from copy import deepcopy



class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, net_lpl=None):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        #print(use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #print(self.device)

        self.net_lpl = net_lpl


    def query(self, n):
        pass
    def get_model(self):
        return self.clf
        
    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            #print(loss)
            loss.backward()
            optimizer.step()

    def train(self, X_train = None, Y_train = None):
        
        n_epoch = self.args['n_epoch']
        dim = self.X.shape[1:]
        self.clf = self.net(dim = dim, num_classes = self.args['num_class']).to(self.device)
        #optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        if X_train is None:
            idxs_train = np.arange(self.n_pool)[self.idxs_lb]
            X_train_full = self.X[idxs_train]
            Y_train_full = self.Y[idxs_train]
        # find ID samples 
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
        ood_sample_num = Y_train_full.shape[0] - Y_train.shape[0]
    
        loader_tr = DataLoader(self.handler(X_train, Y_train, transform=self.args['transform_train']),
                            shuffle=True, **self.args['loader_tr_args'])

        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)
        return ood_sample_num


        

        loader_tr = DataLoader(self.handler(X_train, Y_train, transform=self.args['transform_train']),
                            **self.args['loader_tr_args'])
        #weight = 0.1
        for epoch in range(1, n_epoch+1):
            print(epoch)
            #if epoch >= epoch_loss:
            #     weight = 1.0
            self._train_LPL(epoch, loader_tr, optimizer, optimizer_lpl, epoch_loss, weight, margin)
        return ood_sample_num

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), self.args['num_class']])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            #print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.args['num_class']])
        for i in range(n_drop):
            #print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([self.args['num_class'], self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y)) -1
        embedding = np.zeros([len(Y) -1, embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y) - 1):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0
            return torch.Tensor(embedding)

