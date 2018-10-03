import numpy as np
import cPickle
#import pdb

md = {}
for i in range(2430):
    f = open('./dif50_ap_C/dif50_ap_C'+str(i)+'.pkl','r')
    dif = cPickle.load(f)
    f.close()
    # pdb.set_trace()
    md[i] = 0
    for vec1 in dif:
        for vec2 in dif:
            #if i == 3:
                #pdb.set_trace()
            md[i] += np.sqrt(np.sum(np.square(vec1 - vec2)))
    # if i==3:
        # pdb.set_trace()
    md[i] /= (len(dif)*len(dif))
    print md[i]
f = open('density_rel_C.pkl','w')
cPickle.dump(md, f, -1)
f.close()
