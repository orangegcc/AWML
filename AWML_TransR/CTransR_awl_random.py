#! /usr/bin/python
from expC_awl import *

launch(op='TransR', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
    nbatches=100, totepochs=3000, test_all=10, neval=1000, pretrain='False', beta=24.5, savepath='CTransR_awl_random', datapath='../data_FB15k/')

