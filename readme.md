# POAL: Pareto Optimization for Active Learning under Out-Of-Distribution (OOD) data scenarios.

This is the source code of our Transactions on Machine Learning Research (TMLR) paper "[Pareto Optimization for Active Learning under Out-of-Distribution Data Scenarios](https://openreview.net/forum?id=dXnccpSSYF)".

We aim to solve the problem that target-task-unrelated data (OOD data) would exist in unlabeled data pool under pool-based Active Learning tasks. In this paper, we propose a Monte-Carlo Pareto Optimization for Active Learning (POAL) sampling scheme, which selects optimal subsets of unlabeled samples with fixed batch size from the unlabeled data pool. Our framework is flexible and can accommodate different combinations of AL and OOD detection methods
according to various target tasks. We use Entropy as our basic AL sampling strategy and Mahalanobis distance as the basic ID/OOD confidence score calculation.

## Prerequisites 

- numpy        
- scipy        
- pytorch     
- torchvision      0.11.1
- scikit-learn     1.0.1

## Demo 
- For classical ML tasks
```
  python poal_classical_ml.py \
      -m poal \
      -s 200 \
      -q 500 \
      -b 20 \
      -d EX8ab \
      -r 100 \
      -g 0
More options can be found in classical_ml/poal_classical_ml.py.
```
- for DL tasks:
```
  python run_oodal.py \
      -a POAL_PSES \
      -s 100 \
      -q 1000 \
      -b 100 \
      -d CIFAR10_04 \
      -t 3 \
      -g 0
More options can be found in dl/code/arguments.py.
```

## References
We completed this code with references of:
1.  Pareto Optimization for Subset Selection. [Source Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/nips15poss.pdf) and [Source Code](http://www.lamda.nju.edu.cn/code_POSS.ashx)
2.  DeepAL toolkit. [Source Paper](https://arxiv.org/pdf/2111.15258v1.pdf) and [Source Code](https://github.com/ej0cl6/deep-active-learning). PS. Can also look at our new [DeepAL+ toolkit](https://github.com/SineZHAN/deepALplus)!!!
3.  Mahalanobis [Source Paper](https://arxiv.org/pdf/1807.03888.pdf) and [Source Code](https://github.com/pokaxpoka/deep_Mahalanobis_detector).
