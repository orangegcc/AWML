#! /usr/bin/python
import sys
import cPickle
from modelC_ap import *


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data_FB15k/', dataset='FB15k-test',
        loadmodel='best_valid_model.pkl', neval='all', Nsyn=14951, n=1,
        idx2synsetfile='FB15k_idx2entity.pkl'):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    f.close()
    leftop = LayerMat('lin', 50, 50)
    rightop = LayerMat('lin', 50, 50)
    simfn = eval('L2sim')

    # Load data
    testl = load_file(datapath + dataset + '-lhs.pkl')
    testr = load_file(datapath + dataset + '-rhs.pkl')
    testo = load_file(datapath + dataset + '-rel.pkl')
    if type(embeddings) is list:
        testo = testo[-1345:, :]
    # Convert sparse matrix to indexes
    if neval == 'all':
        idxtl = convert2idx(testl)
        idxtr = convert2idx(testr)
        idxto = convert2idx(testo)
    else:
        idxtl = convert2idx(testl)[:neval]
        idxtr = convert2idx(testr)[:neval]
        idxto = convert2idx(testo)[:neval]

    # Positives
    trainl = load_file(datapath + 'FB15k-train-inpl_C.pkl')
    trainr = load_file(datapath + 'FB15k-train-inpr_C.pkl')
    traino = load_file(datapath + 'FB15k-train-inpo_C.pkl')
    traino = traino[-2467:, :]

    # Valid set
    validl = load_file(datapath + 'FB15k-valid-lhs.pkl')
    validr = load_file(datapath + 'FB15k-valid-rhs.pkl')
    valido = load_file(datapath + 'FB15k-valid-rel.pkl')
    valido = valido[-1345:, :]

    idxl = convert2idx(trainl)
    idxr = convert2idx(trainr)
    idxo = convert2idx(traino)
    idxvl = convert2idx(validl)
    idxvr = convert2idx(validr)
    idxvo = convert2idx(valido)

    true_triples = np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

    rankofunc = RankRelFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn)

    with open('rel2subrel_apC.pkl','rb') as f:
        rel2subrel = cPickle.load(f)

    res = RankingScoreRelIdx_test(rankofunc, idxtl, idxtr, idxto, rel2subrel)

    dres = {}
    dres.update({'micromean': np.mean(res)})
    dres.update({'micromedian': np.median(res)})
    dres.update({'microhits@n': np.mean(np.asarray(res) <= n) * 100})

    print "### MICRO:"
    print "\t-- global   >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['micromean'], 5), round(dres['micromedian'], 5),
            n, round(dres['microhits@n'], 3))

    '''    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellrn = {}
    dictrelrrn = {}
    dictrelgrn = {}

    for i in listrel:
        dictrelres.update({i: [[], []]})

    for i, j in enumerate(res[0]):
        dictrelres[idxto[i]][0] += [j]

    for i, j in enumerate(res[1]):
        dictrelres[idxto[i]][1] += [j]

    for i in listrel:
        dictrellmean[i] = np.mean(dictrelres[i][0])
        dictrelrmean[i] = np.mean(dictrelres[i][1])
        dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        dictrellmedian[i] = np.median(dictrelres[i][0])
        dictrelrmedian[i] = np.median(dictrelres[i][1])
        dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
                                           dictrelres[i][1]) <= n) * 100

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrellmean': dictrellmean})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelgmean': dictrelgmean})
    dres.update({'dictrellmedian': dictrellmedian})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelgmedian': dictrelgmedian})
    dres.update({'dictrellrn': dictrellrn})
    dres.update({'dictrelrrn': dictrelrrn})
    dres.update({'dictrelgrn': dictrelgrn})

    dres.update({'macrolmean': np.mean(dictrellmean.values())})
    dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
    dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
    dres.update({'macrogmean': np.mean(dictrelgmean.values())})
    dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
    dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

    print "### MACRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
            n, round(dres['macrolhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
            n, round(dres['macrorhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
            n, round(dres['macroghits@n'], 3))

    idx2synset = cPickle.load(open(datapath + idx2synsetfile))
    offset = 0
    if type(embeddings) is list:
        idxto = idxto[-embeddings[1].N:, :]
        offset = l.shape[0] - embeddings[1].N
    for i in np.sort(list(listrel)):
        print "### RELATION %s:" % idx2synset[offset + i]
        print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
                round(dictrellmean[i], 5), round(dictrellmedian[i], 5),
                n, round(dictrellrn[i], 3), len(dictrelres[i][0]))
        print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
                round(dictrelrmean[i], 5), round(dictrelrmedian[i], 5),
                n, round(dictrelrrn[i], 3), len(dictrelres[i][1]))
        print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%, N: %s" % (
                roGund(dictrelgmean[i], 5), round(dictrelgmedian[i], 5),
                n, round(dictrelgrn[i], 3),
                len(dictrelres[i][0] + dictrelres[i][1]))
    '''
    return dres


def ClassifEval(datapath='../data_WN18/', validset='WN-valid', testset='WN-test',
        loadmodel='best_valid_model.pkl', seed=647):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    np.random.seed(seed)

    # Load data
    lv = load_file(datapath + validset + '-lhs.pkl')
    lvn = lv[:, np.random.permutation(lv.shape[1])]
    rv = load_file(datapath + validset + '-rhs.pkl')
    rvn = rv[:, np.random.permutation(lv.shape[1])]
    ov = load_file(datapath + validset + '-rel.pkl')
    ovn = ov[:, np.random.permutation(lv.shape[1])]
    if type(embeddings) is list:
        ov = ov[-embeddings[1].N:, :]
        ovn = ovn[-embeddings[1].N:, :]

    # Load data
    lt = load_file(datapath + testset + '-lhs.pkl')
    ltn = lt[:, np.random.permutation(lv.shape[1])]
    rt = load_file(datapath + testset + '-rhs.pkl')
    rtn = rt[:, np.random.permutation(lv.shape[1])]
    ot = load_file(datapath + testset + '-rel.pkl')
    otn = ot[:, np.random.permutation(lv.shape[1])]
    if type(embeddings) is list:
        ot = ot[-embeddings[1].N:, :]
        otn = otn[-embeddings[1].N:, :]

    simfunc = SimFn(simfn, embeddings, leftop, rightop)

    resv = simfunc(lv, rv, ov)[0]
    resvn = simfunc(lvn, rvn, ovn)[0]
    rest = simfunc(lt, rt, ot)[0]
    restn = simfunc(ltn, rtn, otn)[0]

    # Threshold
    perf = 0
    T = 0
    for val in list(np.concatenate([resv, resvn])):
        tmpperf = (resv > val).sum() + (resvn <= val).sum()
        if tmpperf > perf:
            perf = tmpperf
            T = val
    testperf = ((rest > T).sum() + (restn <= T).sum()) / float(2 * len(rest))
    print "### Classification performance : %s%%" % round(testperf * 100, 3)

    return testperf


if __name__ == '__main__':
    #ClassifEval()
    RankingEval(loadmodel=sys.argv[1])
