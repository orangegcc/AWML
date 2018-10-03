import pickle

nent = 40943
nrel = 18

with open('WN_idx2synset.pkl','rb') as f:
    idx2ent = pickle.load(f)

with open('entity2id.txt','w') as f1:
    for i in range(nent):
        ent = idx2ent[i]
        f1.write(str(ent)+'\t'+str(i)+'\n')

with open('relation2id.txt','w') as f2:
    for i in range(nrel):
        rel = idx2ent[i+nent]
        f2.write(str(rel)+'\t'+str(i)+'\n')
