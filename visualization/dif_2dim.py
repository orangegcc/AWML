import cPickle
import os,string
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T

#import pdb

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

f = open('Y_E.pkl','r')
Y = cPickle.load(f)
f.close()

trainl = load_file('../../data_FB15k/FB15k-train-inpl_C.pkl')
trainr = load_file('../../data_FB15k/FB15k-train-inpr_C.pkl')
traino = load_file('../../data_FB15k/FB15k-train-inpo_C.pkl')
traino = traino[-1345:, :]

trainlidx = convert2idx(trainl)
trainridx = convert2idx(trainr)
trainoidx = convert2idx(traino)

dif = {}

for i in range(1345):
    dif[i] = []

for j in range(483142):
    l = trainlidx[j]
    r = trainridx[j]
    o = trainoidx[j] #0~1451
    dif[o] += [[Y[r,0]-Y[l,0], Y[r,1]-Y[l,1]]]
    #pdb.set_trace()

os.mkdir('./dif2_ap/')
for i in range(1345):
    f = open('./dif2_ap/dif2_ap'+str(i)+'.pkl','w')
    cPickle.dump(dif[i], f, -1)
    f.close()
