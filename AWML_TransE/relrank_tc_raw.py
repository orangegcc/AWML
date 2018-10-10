#! /usr/bin/python
# from WN_expC_relrank_ap import *
from evaluationC_tc_raw import *
from time import time
t1 = time()

print "\n##### EVALUATION @10 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC/best_valid_model.pkl', n=10)
print "\n##### EVALUATION @3 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC/best_valid_model.pkl', n=3)
print "\n##### EVALUATION @1 #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC/best_valid_model.pkl', n=1)
t3 = time()
print "Testing time:", t3-t1
