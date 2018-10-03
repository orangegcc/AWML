import cPickle
import os, string
import numpy as np
#import pdb

f = open('./cluster_wn/k.pkl','r')
k = cPickle.load(f)
f.close()

rel2subrel = {}
subrel2rel = {}

n = 40961
for i in range(18):
    subrel2rel[i+40943] = i+40943
    rel2subrel[i+40943] = [i+40943]
    if k[i] > 1:
	for j in range(k[i]-1):
	    subrel2rel[n] = i+40943
	    rel2subrel[i+40943] += [n]
	    n += 1
print n
#pdb.set_trace()
f = open('wn_rel2subrel_apC.pkl','w')
cPickle.dump(rel2subrel,f,-1)
f.close()
g = open('wn_subrel2rel_apC.pkl','w')
cPickle.dump(subrel2rel,g,-1)
g.close()
