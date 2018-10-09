#! /usr/bin/python
from WN_expC_aml import *
# from WN_evaluation_ap import *
from time import time
t1 = time()

if theano.config.floatX == 'float32':
    sys.stderr.write("""WARNING: Detected floatX=float32 in the configuration.
This might result in NaN in embeddings after several epochs.
""")

launch(op='TransE', dataset='WN', simfn='L1', ndim=20, nhid=20, marge=1., lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=5000, test_all=10, neval=1000, savepath='WN_TransC_aml_random',
    datapath='../../data_WN18/', Nent=40991,  Nsyn=40943, Nrel=48, pretrain='False')
t2 = time()
print "Training time:", t2-t1

'''print "\n##### EVALUATION @10 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC_ap_den2/best_valid_model.pkl')
print "\n##### EVALUATION @5 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC_ap_den2/best_valid_model.pkl', n=5)
print "\n##### EVALUATION @2 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC_ap_den2/best_valid_model.pkl', n=2)
print "\n##### EVALUATION @1 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC_ap_den2/best_valid_model.pkl', n=1)
t3 = time()
print "Testing time:", t3-t2'''
