import pickle

nent = 14951
nrel = 1345

with open('FB15k_idx2entity.pkl','rb') as f:
    idx2ent = pickle.load(f)

with open('entity2id.txt','w') as f1:
    for i in range(nent):
        ent = idx2ent[i]
        f1.write(str(ent)+'\t'+str(i)+'\n')

with open('relation2id.txt','w') as f2:
    for i in range(nrel):
        rel = idx2ent[i+nent]
        f2.write(str(rel)+'\t'+str(i)+'\n')
