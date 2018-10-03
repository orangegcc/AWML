import cPickle
import os, string
import numpy as np
#import pdb

f = open('./cluster/k.pkl','r')
k = cPickle.load(f)
f.close()

rel2subrel = {}
subrel2rel = {}

n = 16296
for i in range(1345):
    subrel2rel[i+14951] = i+14951
    rel2subrel[i+14951] = [i+14951]
    if k[i] > 1:
	for j in range(k[i]-1):
	    subrel2rel[n] = i+14951
	    rel2subrel[i+14951] += [n]
	    n += 1
print n
#pdb.set_trace()
f = open('rel2subrel_apC.pkl','w')
cPickle.dump(rel2subrel,f,-1)
f.close()
g = open('subrel2rel_apC.pkl','w')
cPickle.dump(subrel2rel,g,-1)
g.close()
