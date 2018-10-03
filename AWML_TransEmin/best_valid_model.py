import cPickle
import os,string
import numpy as np

f = open('./fb15k_embeddings/TransE/best_valid_model.pkl','r')
x = cPickle.load(f)
f.close()

x0 = x[0].E.container.storage[0]
x1 = x[1].E.container.storage[0]

nbent = 14951
nbrel = 1345
initial_dims = 50

X0 = {}
X1 = {}

for j in range(initial_dims):
    X0[j] = np.asarray(x0[j])
    X1[j] = np.asarray(x1[j])

for i in range(nbent):
    for j in range(initial_dims):
        if j == initial_dims - 1:
            print ' ', X0[j][i]
        else:
            if j == 0:
                print '  ', X0[j][i],
            else:
                print ' ', X0[j][i],
for i in range(nbrel):
    for j in range(initial_dims):
        if j == initial_dims - 1:
            print ' ', X1[j][i]
        else:
            if j == 0:
                print '  ', X1[j][i],
            else:
                print ' ', X1[j][i],
