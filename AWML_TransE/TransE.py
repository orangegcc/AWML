#! /usr/bin/python
from exp import *
# from WN_evaluation import *
from time import time
t1 = time()

if theano.config.floatX == 'float32':
    sys.stderr.write("""WARNING: Detected floatX=float32 in the configuration.
This might result in NaN in embeddings after several epochs.
""")

launch(op='TransE', dataset='WN', simfn='L1', ndim=20, nhid=20, marge=2., lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=1000, test_all=10, neval=1000, savepath='WN_TransE',
    datapath='../data_WN18/', Nent=40961,  Nsyn=40943, Nrel=18)
t2 = time()
print "Training time:", t2-t1

# print "\n##### EVALUATION #####\n"
# RankingEval(datapath='../data_WN18/', loadmodel='WN_TransE/best_valid_model.pkl')
# t3 = time()
# print "Testing time:", t3-t2
