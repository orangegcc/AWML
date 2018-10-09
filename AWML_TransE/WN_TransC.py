#! /usr/bin/python
from WN_expC import *
from WN_evaluation import *
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

print "\n##### EVALUATION #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC_ap/best_valid_model.pkl')
t3 = time()
print "Testing time:", t3-t1
