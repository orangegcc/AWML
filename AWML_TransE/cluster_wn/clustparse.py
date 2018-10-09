import cPickle
import os ,string
#import pdb

# os.mkdir('./centid')
# os.mkdir('./centindex')
# os.mkdir('./clustid')
k = {}
for i in range(18):
    if os.path.exists('./id2index/id2index'+str(i)+'.txt'):
        f = open('./id2index/id2index'+str(i)+'.txt', 'r')
        id2index = f.readlines()
        f.close()
        for j in range(len(id2index)):
	    id2index[j] = int(id2index[j].split('\n')[0])
        if os.path.exists('./cluster/cluster'+str(i)+'.txt'):
	    centid = []
	    centindex = []
	    clustid = []
	    g = open('./cluster/cluster'+str(i)+'.txt','r')
	    cluster = g.readlines()
	    g.close()
	    #pdb.set_trace()
	    for j in range(len(cluster)):
                cluster[j] = int(cluster[j].split('\n')[0])
	        if cluster[j] not in centid:
		    centid += [cluster[j]]
		    centindex += [id2index[cluster[j]-1]]
	        clustid += [centid.index(cluster[j])]
	    k[i] = len(centid)
	    f = open('./centid/centid'+str(i)+'.pkl','w')
	    cPickle.dump(centid, f, -1)
	    f.close()
            f = open('./centindex/centindex'+str(i)+'.pkl','w')
            cPickle.dump(centindex, f, -1)
            f.close()
            f = open('./clustid/clustid'+str(i)+'.pkl','w')
            cPickle.dump(clustid, f, -1)
            f.close()
        else:
	    k[i] = 1
    else:
        k[i] = 1
f = open('k.pkl','w')
cPickle.dump(k, f, -1)
f.close()
    #pdb.set_trace()
