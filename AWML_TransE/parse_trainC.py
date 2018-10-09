import cPickle
import os,string
import numpy as np
import scipy
import scipy.sparse as sp
#import pdb

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

f = open('../data_WN18/WN-train-lhs.pkl','r')
g = open('../data_WN18/WN-train-rhs.pkl','r')
h = open('../data_WN18/WN-train-rel.pkl','r')
trainl = cPickle.load(f)
trainr = cPickle.load(g)
traino = cPickle.load(h)
f.close()
g.close()
h.close()

idxl = convert2idx(trainl)
idxr = convert2idx(trainr)
idxo = convert2idx(traino)

inpl = sp.lil_matrix((40943+48, 141442), dtype='float32')
inpr = sp.lil_matrix((40943+48, 141442), dtype='float32')
inpo = sp.lil_matrix((40943+48, 141442), dtype='float32')

f = open('rel2subrel_apC.pkl','r')
rel2subrel = cPickle.load(f)
f.close()
g = open('../data_WN18/WN_idx2synset.pkl','r')
idx2entity = cPickle.load(g)
g.close()
f = open('./cluster/k.pkl','r')
k = cPickle.load(f)
f.close()

count = {}
clustmat = {}
for i in range(18):
    count[i] = 0
    if k[i] > 1:
        #pdb.set_trace()
        f = open('./cluster/clustid/clustid'+str(i)+'.pkl','r')
        clustid = cPickle.load(f)
        f.close()
        clustmat[i] = clustid

for i in range(141442):
    inpl[idxl[i],i] = 1
    inpr[idxr[i],i] = 1
    if k[idxo[i]-40943] > 1:
	r = idxo[i]
	inpo[rel2subrel[r][clustmat[r-40943][count[r-40943]]],i] = 1
	count[r-40943] += 1
	#pdb.set_trace()
    else:
	inpo[idxo[i],i] = 1

f = open('../data_WN18/WN-train-inpl_C.pkl','w')
g = open('../data_WN18/WN-train-inpr_C.pkl','w')
h = open('../data_WN18/WN-train-inpo_C.pkl','w')
cPickle.dump(inpl.tocsr(),f,-1)
cPickle.dump(inpr.tocsr(),g,-1)
cPickle.dump(inpo.tocsr(),h,-1)
f.close()
g.close()
h.close()

