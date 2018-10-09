import cPickle
import os,string
from numpy import *
import numpy as np
import scipy
import scipy.sparse
#import pdb
import theano
import theano.sparse as S
import theano.tensor as T
from collections import OrderedDict

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

f = open('./WN_TransC_ap/embedding/750.pkl','r')
x = cPickle.load(f)
f.close()
x0 = x[0].normalize.input_storage[0].storage[0]
#pdb.set_trace()
x0 = x0.T

trainl = load_file('../data_WN18/WN-train-inpl_C.pkl')
trainr = load_file('../data_WN18/WN-train-inpr_C.pkl')
traino = load_file('../data_WN18/WN-train-inpo_C.pkl')
traino = traino[-48:, :]

trainlidx = convert2idx(trainl)
trainridx = convert2idx(trainr)
trainoidx = convert2idx(traino)

dif50 = {}
for i in range(48):
    dif50[i] = []

for j in range(141442):
    l = trainlidx[j]
    r = trainridx[j]
    o = trainoidx[j] #0~1451
    dif50[o] += [x0[r] - x0[l]]
    #pdb.set_trace()
    
os.mkdir('./dif50_ap_C')
for i in range(48):
    g = open('./dif50_ap_C/dif50_ap_C'+str(i)+'.pkl','w')
    cPickle.dump(dif50[i], g, -1)
    g.close()
