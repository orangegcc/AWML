import cPickle
import os,string
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T

import pdb

def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = scipy.sparse.lil_matrix((shape[0], shape[1]),
            dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

f = open('Y_E.pkl','r')
Y = cPickle.load(f)
f.close()
f = open('../../data_FB15k/freebase_mtr100_mte100-train.txt','r')
dat = f.readlines()
f.close()
#pdb.set_trace()
# Positives
trainl = load_file('../../data_FB15k/FB15k-train-inpl_C.pkl')
trainr = load_file('../../data_FB15k/FB15k-train-inpr_C.pkl')
traino = load_file('../../data_FB15k/FB15k-train-inpo_C.pkl')
traino = traino[-1345:, :]

# Negatives
trainln = create_random_mat(trainl.shape, np.arange(14951))
trainrn = create_random_mat(trainr.shape, np.arange(14951))


trainlidx = convert2idx(trainl)
trainridx = convert2idx(trainr)
trainoidx = convert2idx(traino)
trainlnidx = convert2idx(trainln)
trainrnidx = convert2idx(trainrn)


difl = {}
difr = {}
difl_real = {}
difr_real = {}

os.mkdir('realneg_ctranse')
for i in range(1345):
    difl[i] = []
    difr[i] = []
    difl_real[i] = []
    difr_real[i] = []

for j in range(483142):
    l = trainlidx[j]
    r = trainridx[j]
    o = trainoidx[j] #0~2350
    ln = trainlnidx[j]
    rn = trainrnidx[j]
    awln = np.argwhere(trainlidx[:]==ln)
    awrn = np.argwhere(trainridx[:]==rn)
    #pdb.set_trace()
    if awln.any():
        for aw in awln:
            i = aw[0]
            if trainridx[i] == r:
                if trainoidx[i] != o:
                    difl_real[o] += [[Y[r,0]-Y[ln,0], Y[r,1]-Y[ln,1]]]
                    f = open('./realneg_ctranse/h'+str(o)+'.txt', 'a')
                    f.write('\n'+dat[i]+str(trainoidx[i]))
                    f.close()
                    break
            else:
                if i == awln[-1]:
                    difl[o] += [[Y[r,0]-Y[ln,0], Y[r,1]-Y[ln,1]]]
    else:
         difl[o] += [[Y[r,0]-Y[ln,0], Y[r,1]-Y[ln,1]]]

    if awrn.any():
        for aw in awrn:
            i = aw[0]
            if trainlidx[i] == l:
                if trainoidx[i] != o:
                    difr_real[o] += [[Y[rn,0]-Y[l,0], Y[rn,1]-Y[l,1]]]
                    f = open('./realneg_ctranse/t'+str(o)+'.txt', 'a')
                    f.write('\n'+dat[i]+str(trainoidx[i]))
                    f.close()
                    break
            else:
                if i == awrn[-1]:
                    difr[o] += [[Y[rn,0]-Y[l,0], Y[rn,1]-Y[l,1]]]
    else:
         difr[o] += [[Y[rn,0]-Y[l,0], Y[rn,1]-Y[l,1]]]

    #pdb.set_trace()

os.mkdir('./dif2_negl_r_ctranse/')
os.mkdir('./dif2_negr_r_ctranse/')
os.mkdir('./dif2_negl_real_ctranse/')
os.mkdir('./dif2_negr_real_ctranse/')
for i in range(1345):
    f = open('./dif2_negl_r_ctranse/dif2_negl_r_ctranse'+str(i)+'.pkl','w')
    g = open('./dif2_negr_r_ctranse/dif2_negr_r_ctranse'+str(i)+'.pkl','w')
    cPickle.dump(difl[i], f, -1)
    cPickle.dump(difr[i], g, -1)
    f.close()
    g.close()
    f = open('./dif2_negl_real_ctranse/dif2_negl_real_ctranse'+str(i)+'.pkl','w')
    g = open('./dif2_negr_real_ctranse/dif2_negr_real_ctranse'+str(i)+'.pkl','w')
    cPickle.dump(difl_real[i], f, -1)
    cPickle.dump(difr_real[i], g, -1)
    f.close()
    g.close()

