#! /usr/bin/python
# from WN_expC import *
from evaluationC_lp_raw import *
from time import time
t1 = time()

'''if theano.config.floatX == 'float32':
    sys.stderr.write("""WARNING: Detected floatX=float32 in the configuration.
This might result in NaN in embeddings after several epochs.
""")'''

'''launch(op='TransE', dataset='WN', simfn='L1', ndim=20, nhid=20, marge=2., lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=1000, test_all=10, neval=1000, savepath='WN_TransC',
    datapath='../../data_WN18/', Nent=40961,  Nsyn=40943, Nrel=18)
t2 = time()
print "Training time:", t2-t1'''

fn = './FB15k_embedding/CTransR/best_valid_model.pkl'
print "\n##### EVALUATION @10 #####\n"
RankingEval(datapath='../data_FB15k/', loadmodel=fn, n=10)
print "\n##### EVALUATION @3 #####\n"
RankingEval(datapath='../data_FB15k/', loadmodel=fn, n=3)
print "\n##### EVALUATION @1 #####\n"
RankingEval(datapath='../data_FB15k/', loadmodel=fn, n=1)
# print "\n##### EVALUATION @1 #####\n"
# RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC/best_valid_model.pkl', n=1)'''
t3 = time()
print "Testing time:", t3-t1
