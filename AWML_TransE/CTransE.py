#! /usr/bin/python
from expC import *
from time import time
t1 = time()

if theano.config.floatX == 'float32':
    sys.stderr.write("""WARNING: Detected floatX=float32 in the configuration.
This might result in NaN in embeddings after several epochs.
""")

launch(op='TransE', dataset='WN', simfn='L1', ndim=20, nhid=20, marge=2., lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=1000, test_all=10, neval=1000, savepath='WN_TransC',
    datapath='../data_WN18/', Nent=40991,  Nsyn=40943, Nrel=48)
t2 = time()
print "Training time:", t2-t1
