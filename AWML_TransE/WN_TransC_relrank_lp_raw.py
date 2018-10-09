#! /usr/bin/python
# from WN_expC_relrank_ap import *
from WN_evaluationC_lp_raw import *
from time import time

print "\n##### EVALUATION #####\n"
RankingEval(datapath='../data_WN18/', loadmodel='WN_TransC/best_valid_model.pkl', n=10)
