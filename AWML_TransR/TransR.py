#! /usr/bin/python
from exp import *

launch(op='TransR', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
    nbatches=100, totepochs=5000, test_all=10, neval=1000, savepath='TransR', datapath='../data_FB15k/')

