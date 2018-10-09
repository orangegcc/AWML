#coding:utf-8
#! /usr/bin/python

from modelC_aml import *
import pdb

# Utils ----------------------------------------------------------------------
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
    rows, cols = spmat.nonzero()  #返回输入值中非零元素的信息（以矩阵的形式）这些信息中包括 两个矩阵， 包含了相应维度上非零元素所在的行标号，与列标标号。
    return rows[np.argsort(cols)]  #argsort函数返回的是数组值从小到大的索引值


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z

# ----------------------------------------------------------------------------


# Experiment function --------------------------------------------------------
def FB15kexp(state, channel):

    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)

    # Positives
    trainl = load_file(state.datapath + state.dataset + '-train-inpl_C.pkl')
    trainr = load_file(state.datapath + state.dataset + '-train-inpr_C.pkl')
    traino = load_file(state.datapath + state.dataset + '-train-inpo_C.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        traino = traino[-state.Nrel:, :]
    #elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
        #trainl = trainl[:state.Nsyn, :]
        #trainr = trainr[:state.Nsyn, :]
        #traino = traino[-state.Nrel:, :]

    # Valid set
    validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')
    validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')
    valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        valido = valido[-18:, :]
    #elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
        #validl = validl[:state.Nsyn, :]
        #validr = validr[:state.Nsyn, :]
        #valido = valido[-state.Nrel:, :]


    # Test set
    testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')
    testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')
    testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        testo = testo[-18:, :]
    #elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
        #testli = testl[:state.Nsyn, :]
        #testr = testr[:state.Nsyn, :]
        #testo = testo[-state.Nrel:, :]

    # Index conversion
    trainlidx = convert2idx(trainl)[:state.neval]  #按训练三元组的顺序返回头实体的字典序号，并选出前1000个（neval）,下同
    trainridx = convert2idx(trainr)[:state.neval]
    trainoidx = convert2idx(traino)[:state.neval]
    validlidx = convert2idx(validl)[:state.neval]
    validridx = convert2idx(validr)[:state.neval]
    validoidx = convert2idx(valido)[:state.neval]
    testlidx = convert2idx(testl)[:state.neval]
    testridx = convert2idx(testr)[:state.neval]
    testoidx = convert2idx(testo)[:state.neval]
    #pdb.set_trace()

    idxl = convert2idx(trainl)  #按训练三元组的顺序返回头实体的字典序号，下同
    idxr = convert2idx(trainr)
    idxo = convert2idx(traino)
    idxtl = convert2idx(testl)
    idxtr = convert2idx(testr)
    idxto = convert2idx(testo)
    idxvl = convert2idx(validl)
    idxvr = convert2idx(validr)
    idxvo = convert2idx(valido)
    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T
    #reshape重构数组的形状，此处将一维数组变成3×(59071+50000+483142)的二维数组，第一维的三个列表分别表示头实体、关系、尾实体，第二维先后表示测试、验证、训练三元组
    #.T为转置，因此true_triples为一个(59071+50000+483142)×3的二维数组

    # Model declaration
    if not state.loadmodel:
        # operators
        if state.op == 'Unstructured':
            leftop  = Unstructured()
            rightop = Unstructured()
        elif state.op == 'SME_lin':
            leftop  = LayerLinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'left')
            rightop = LayerLinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'right')
        elif state.op == 'SME_bil':
            leftop  = LayerBilinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'left')
            rightop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'right')
        elif state.op == 'SE':
            leftop  = LayerMat('lin', state.ndim, state.nhid)
            rightop = LayerMat('lin', state.ndim, state.nhid)
        elif state.op == 'TransE':
            leftop  = LayerTrans()  #该函数实现加和的功能
            rightop = Unstructured()  #该函数实现返回自身的功能
        elif state.op == 'Bi':
            leftop = LayerMat('lin', state.ndim, 1)
            rightop = LayerdMat()
        elif state.op == 'Tri':
            leftop = LayerMat('lin', state.ndim, 1)
            rightop = LayerMat('lin', state.ndim, state.ndim)
        # embeddings
        if not state.loademb:
            embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')
        else:
            f = open(state.loademb)
            embeddings = cPickle.load(f)
            f.close()
        if state.op == 'SE' and type(embeddings) is not list:
            relationl = Embeddings(np.random, state.Nrel, state.ndim * state.nhid, 'rell')
            relationr = Embeddings(np.random, state.Nrel, state.ndim * state.nhid, 'relr')
            embeddings = [embeddings, relationl, relationr]
        if state.op == 'TransE' and type(embeddings) is not list:
            relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
            embeddings = [embeddings, relationVec, relationVec]
        if state.op == 'Bi' and type(embeddings) is not list:
            embeddings = Embeddings(np.random, state.Nsyn, state.ndim, 'emb')
            W = Embeddings(np.random, 1, state.ndim, 'W')
            rel_matricesl = Embeddings(np.random, state.Nrel, state.ndim, 'relmatL')
            rel_matricesr = Embeddings(np.random, state.Nrel, state.ndim, 'relmatR')
            embeddings = [embeddings, W, rel_matricesl, rel_matricesr]
        if state.op == 'Tri' and type(embeddings) is not list:
            embeddings = Embeddings(np.random, state.Nsyn, state.ndim, 'emb')
            rel_matrices = Embeddings(np.random, state.Nrel, state.ndim*state.ndim, 'relmat')
            embeddings = [embeddings, rel_matrices]
        simfn = eval(state.simfn + 'sim')
    else:
        if state.op == 'TATEC':
            f = open(state.loadmodelBi)
            embbi = cPickle.load(f)
            leftopbi = cPickle.load(f)
            rightopbi = cPickle.load(f)
            f.close()
            f = open(state.loadmodelTri)
            embtri = cPickle.load(f)
            leftoptri = cPickle.load(f)
            rightoptri = cPickle.load(f)
            f.close()
            embeddings = [embbi[0], embbi[1], embbi[2], embbi[3], embtri[0], embtri[1]]
        else:
            f = open(state.loadmodel)
            embeddings = cPickle.load(f)
            leftop = cPickle.load(f)
            rightop = cPickle.load(f)
            simfn = cPickle.load(f)
            f.close()
    if state.pretrain == 'True':
        f = open('./WN_TransC/best_valid_model.pkl','r')
        embeddings = cPickle.load(f)
        f.close()

    marw = Embeddings(np.random, 1, 1, 'margeweight')
    marb = Embeddings(np.random, state.Nrel, 1, 'margebias')

    f = open('density_rel_C.pkl','r')
    dens = cPickle.load(f)
    f.close()
    # den = np.array(density)
    # pdb.set_trace()
    density = Embeddings(np.random, state.Nrel, 1, 'emb')
    for r in range(state.Nrel):
        density.E.container.storage[0][0,r] = dens[r]
    # pdb.set_trace()
    # Function compilation
    if state.op == 'Bi':
        trainfunc = TrainFn1MemberBi(embeddings, leftop, rightop, marge=state.marge)
        ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=state.Nsyn)
        rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=state.Nsyn)
    elif state.op == 'Tri':
        trainfunc = TrainFn1MemberTri(embeddings, leftop, rightop, marge=state.marge)
        ranklfunc = RankLeftFnIdxTri(embeddings, leftop, rightop, subtensorspec=state.Nsyn)
        rankrfunc = RankRightFnIdxTri(embeddings, leftop, rightop, subtensorspec=state.Nsyn)
    elif state.op == 'TATEC':
        trainfunc = TrainFn1MemberTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, marge=state.marge)
        ranklfunc = RankLeftFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, subtensorspec=state.Nsyn)
        rankrfunc = RankRightFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, subtensorspec=state.Nsyn)
    else:
	#pdb.set_trace()
        trainfunc = TrainFn1Member(simfn, embeddings, density, marw, marb, leftop, rightop, marge=state.marge, rel=True)
        ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)
        rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)
	#ranklfuncvt = RankLeftFnIdxvt(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)
	#rankrfuncvt = RankRightFnIdxvt(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)

    out = []
    outb = []
    os.mkdir(state.savepath+'/margel/')
    os.mkdir(state.savepath+'/marger/')
    os.mkdir(state.savepath+'/embedding/')
    state.bestvalid = -1

    batchsize = trainl.shape[1] / state.nbatches  #每次训练的三元组个数483142/100

    f = open('./subrel2rel_apC.pkl','r')
    g = open('./rel2subrel_apC.pkl','r')
    subrel2rel = cPickle.load(f)
    rel2subrel = cPickle.load(g)
    f.close()
    g.close()

    # f = open('density_rel_C.pkl','r')
    # density = cPickle.load(f)
    # f.close()
    # den = np.array(density)
    # density = Embeddings(np.random, state.Nrel, 1, 'emb')
    # for r in range(state.Nrel):
    #     density.E.container.storage[0][0,r] = density[r]

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):#训练500遍
        # Shuffling 将训练三元组顺序重排
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        #pdb.set_trace()
        # Negatives
        trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
        trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))
	trainon = create_random_mat(traino.shape, np.arange(state.Nrel))

        for i in range(state.nbatches): #100
	    #pdb.set_trace()
            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
            tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
	    tmpno = trainon[:, i * batchsize:(i + 1) * batchsize]
            # training iteration
            outtmp = trainfunc(state.lremb, state.lrparam / float(batchsize), tmpl, tmpr, tmpo, tmpnl, tmpnr, tmpno) #lremb=0.01,lrparam=0.01,
            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]
            # embeddings normalization
            if type(embeddings) is list and state.op == 'Bi':
                auxE = embeddings[0].E.get_value()
                idx=np.where(np.sqrt(np.sum(auxE ** 2, axis=0)) > state.rhoE)
                auxE[:, idx] = (state.rhoE*auxE[:, idx]) / np.sqrt(np.sum(auxE[:, idx] ** 2, axis=0))
                embeddings[0].E.set_value(auxE)
            elif type(embeddings) is list and state.op == 'Tri':
                auxE = embeddings[0].E.get_value()
                idx=np.where(np.sqrt(np.sum(auxE ** 2, axis=0)) > state.rhoE)
                auxE[:, idx] = (state.rhoE*auxE[:, idx]) / np.sqrt(np.sum(auxE[:, idx] ** 2, axis=0))
                embeddings[0].E.set_value(auxE)
                auxR = embeddings[1].E.get_value()
                idx=np.where(np.sqrt(np.sum(auxR ** 2, axis=0)) > state.rhoL)
                auxR[:, idx] = (state.rhoL*auxR[:, idx]) / np.sqrt(np.sum(auxR[:, idx] ** 2, axis=0))
                embeddings[1].E.set_value(auxR)
            elif type(embeddings) is list and state.op == 'TATEC':
                auxEb = embeddings[0].E.get_value()
                idxb=np.where(np.sqrt(np.sum(auxEb ** 2, axis=0)) > state.rhoE)
                auxEb[:, idxb] = (state.rhoE*auxEb[:, idxb]) / np.sqrt(np.sum(auxEb[:, idxb] ** 2, axis=0))
                embeddings[0].E.set_value(auxEb)
                auxEt = embeddings[4].E.get_value()
                idxt=np.where(np.sqrt(np.sum(auxEt ** 2, axis=0)) > state.rhoE)
                auxEt[:, idxt] = (state.rhoE*auxEt[:, idxt]) / np.sqrt(np.sum(auxEt[:, idxt] ** 2, axis=0))
                embeddings[4].E.set_value(auxEt)
                auxR = embeddings[5].E.get_value()
                idxr=np.where(np.sqrt(np.sum(auxR ** 2, axis=0)) > state.rhoL)
                auxR[:, idxr] = (state.rhoL*auxR[:, idxr]) / np.sqrt(np.sum(auxR[:, idxr] ** 2, axis=0))
                embeddings[5].E.set_value(auxR)
            elif type(embeddings) is list:
                embeddings[0].normalize()
            else:
                embeddings.normalize()

        marw.normalize()
        marb.normalize()

        if (epoch_count % state.test_all) == 0:
            # model evaluation
            print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                    epoch_count,
                    round(time.time() - timeref, 3) / float(state.test_all))
            timeref = time.time()
            print >> sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                    round(np.mean(out), 4), round(np.std(out), 4),
                    round(np.mean(outb) * 100, 3))
            out = []
            outb = []
            resvalid = FilteredRankingScoreIdx_test(ranklfunc, rankrfunc, validlidx, validridx, validoidx, true_triples, rel2subrel)
            state.valid = np.mean(resvalid[0] + resvalid[1])
            restrain = FilteredRankingScoreIdx_train(ranklfunc, rankrfunc, trainlidx, trainridx, trainoidx, true_triples, subrel2rel)
            state.train = np.mean(restrain[0] + restrain[1])
            print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (state.valid, state.train)
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                restest = FilteredRankingScoreIdx_test(ranklfunc, rankrfunc, testlidx, testridx, testoidx, true_triples, rel2subrel)
                state.bestvalid = state.valid
                state.besttrain = state.train
                state.besttest = np.mean(restest[0] + restest[1])
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                if state.op == 'TATEC':
                    cPickle.dump(embeddings, f, -1)
                    cPickle.dump(leftopbi, f, -1)
                    cPickle.dump(leftoptri, f, -1)
                    cPickle.dump(rightopbi, f, -1)
                    cPickle.dump(rightoptri, f, -1)
                else:
                    cPickle.dump(embeddings, f, -1)
                    cPickle.dump(leftop, f, -1)
                    cPickle.dump(rightop, f, -1)
                    cPickle.dump(simfn, f, -1)
                f.close()
                print >> sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (
                        state.besttest)
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            if state.op == 'TATEC':
                cPickle.dump(embeddings, f, -1)
                cPickle.dump(leftopbi, f, -1)
                cPickle.dump(leftoptri, f, -1)
                cPickle.dump(rightopbi, f, -1)
                cPickle.dump(rightoptri, f, -1)
            else:
                cPickle.dump(embeddings, f, -1)
                cPickle.dump(leftop, f, -1)
                cPickle.dump(rightop, f, -1)
                cPickle.dump(simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()

        if (epoch_count % 200) == 0:
            f = open(state.savepath+'/margel/'+str(epoch_count)+'.pkl','w')
            cPickle.dump(outtmp[2], f, -1)
            f.close()
            g = open(state.savepath+'/marger/'+str(epoch_count)+'.pkl','w')
            cPickle.dump(outtmp[3], g, -1)
            g.close()
            p = open(state.savepath+'/embedding/'+str(epoch_count)+'.pkl','w')
            cPickle.dump(embeddings, p, -1)
            p.close()

            print outtmp[2]
            print outtmp[3]

    test = FilteredRankingScoreIdx_test(ranklfunc, rankrfunc, idxtl, idxtr, idxto, true_triples, rel2subrel)
    mr = np.mean(test[0] + test[1])
    vr = np.var(test[0] + test[1])
    hits = 0
    for i in (test[0]+test[1]):
        if i < 11:
            hits += 1
    hits = float(hits)/10000
    print "mean rank: ", mr
    print "variance: ", vr
    print "hits@10: ", hits*100,'%'

    return channel.COMPLETE


def launch(datapath='data/', dataset='WN', Nent=40991,
        Nsyn=40943, Nrel=48, loadmodel=False, loademb=False, op='Unstructured',
        simfn='Dot', ndim=20, nhid=20, marge=1., lremb=0.1, lrparam=1.,
        nbatches=100, totepochs=5000, test_all=1, neval=50, seed=123,
        savepath='.', pretrain='True'):

    # Argument of the experiment script
    state = DD()
    state.datapath = datapath
    state.dataset = dataset
    state.Nent = Nent
    state.Nsyn = Nsyn
    state.Nrel = Nrel
    state.loadmodel = loadmodel
    # state.loadmodelBi = loadmodelBi
    # state.loadmodelTri = loadmodelTri
    state.loademb = loademb
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.marge = marge
    # state.rhoE = rhoE
    # state.rhoL = rhoL
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.neval = neval
    state.seed = seed
    state.savepath = savepath
    state.pretrain = pretrain

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    # Jobman channel remplacement
    class Channel(object):
        def __init__(self, state):
            self.state = state
            f = open(self.state.savepath + '/orig_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()
            self.COMPLETE = 1

        def save(self):
            f = open(self.state.savepath + '/current_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()

    channel = Channel(state)

    FB15kexp(state, channel)

if __name__ == '__main__':
    launch()
