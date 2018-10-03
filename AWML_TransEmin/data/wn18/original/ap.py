import pickle
import pdb
with open('../../../wn_rel2subrel_apC.pkl','rb') as f:
    r2subr = pickle.load(f)
with open('./WN_idx2synset.pkl','rb') as g:
    idx2ent = pickle.load(g)
# N = 14951
# pdb.set_trace()
for r in r2subr:
    rel = idx2ent[r]
    subr = r2subr[r]
    i = 0
    for s in subr:
        i += 1
        idx2ent[s] = rel+str(i)

with open('./WN_idx2synset_C.pkl','wb') as f:
    pickle.dump(idx2ent, f, -1)
# pdb.set_trace()
