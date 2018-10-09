import cPickle
import string, os
import matplotlib
matplotlib.use('Agg')
import numpy as Math
import pylab as Plot
import pdb

#f = open('../FB15k/rel2subrel.pkl','r')
#g = open('../FB15k/subrel2rel.pkl','r')
#rel2subrel = cPickle.load(f)
#subrel2rel = cPickle.load(g)
#f.close()
#g.close()

p = open('rel2subrel_apC.pkl','r')
g = open('subrel2rel_apC.pkl','r')
rel2subrel = cPickle.load(p)
subrel2rel = cPickle.load(g)
p.close()
g.close()
# q = open("./raw_rel_sort.pkl",'r')
# relsort = cPickle.load(q)
# q.close()
#relsort = sorted(nbtri.iteritems(), key = lambda asd:asd[1], reverse = True)
#f = open('./nbtri_rel_ap.pkl','r')
#nb = cPickle.load(f)
#f.close()
#relsort = Math.argsort(nb)
#pdb.set_trace()

color = ['#ffff00', '#ff9933', '#ff6600', '#ff0000', '#ff0066', '#ff33cc', '#cc00cc', '#6600ff', '#0099ff', '#00ffff', '#66ff66', '#ccff66', '#999966', '#000000', '#003300', '#009933', '#00cc99', '#000066', '#660066', '#800000']

#pdb.set_trace()
f = open('../../data_FB15k/FB15k_entity2idx.pkl','r')
entity2idx = cPickle.load(f)
f.close()
h = open('Y_E.pkl','r')
Y = cPickle.load(h)
h.close()
#os.mkdir('./negplot/')
#pdb.set_trace()
#g = open('negplot_state.txt','w')
rel = '/film/film_job/films_with_this_crew_job./film/film_crew_gig/film'
r = entity2idx[rel]
sr = rel2subrel[r]

for i in range(len(sr)):
    if i >= 0:
        #os.mkdir('./negplot/'+rel.replace('/','_'))
        Plot.figure(figsize=(30,30), dpi=1000)
        # move spines
        ax = Plot.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('none')
        ax.spines['bottom'].set_color('none')
        # ax.yaxis.set_ticks_position('none')
        ax.spines['left'].set_color('none')

        g = open('negplot_state.txt','w')
        g.write('\nsubrel:'+str(len(sr)))

	f = open('./dif2_ap/dif2_ap'+str(sr[i]-14951)+'.pkl','r')
	dif = cPickle.load(f)
	f.close()
	for vec in dif:
            Plot.scatter(vec[0], vec[1], s = 10, c = '#ff0000', marker = '4', edgecolors = 'none')
        g.write('\npositive sample finished!')
        print len(dif)

        f = open('./dif2_negl_r_ctranse/dif2_negl_r_ctranse'+str(sr[i]-14951)+'.pkl','r')
        dif = cPickle.load(f)
        f.close()
        for vec in dif:
            Plot.scatter(vec[0], vec[1], s = 7, c = '#660066', marker = '2', edgecolors = 'none')
        g.write('\nleft negative sample finished!')

        f = open('./dif2_negr_r_ctranse/dif2_negr_r_ctranse'+str(sr[i]-14951)+'.pkl','r')
        dif = cPickle.load(f)
        f.close()
        for vec in dif:
            Plot.scatter(vec[0], vec[1], s = 7, c = '#000066', marker = '1', edgecolors = 'none')
        g.write('\nright negative sample finished!')


	Plot.scatter(Y[i,0], Y[i,1], s = 15000, c = '#000000', marker = '*', label = rel+str(i))

        Plot.legend(loc='upper left', frameon=False)
        Plot.savefig('./negplot_ctranse_r/'+rel.replace('/','_')+str(i)+'.png')
        g.close()
