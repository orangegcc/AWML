#!/usr/bin/python -uB
# -*- coding: utf-8 -*-
# import pdb
import numpy as np
import theano
# from time import time
import random, pickle, datetime, time
import os, sys, types, socket, getopt, logging

import data.util as util

import energy.sparse.learning_den as learning

import energy.activation as activation
import energy.similarity as similarity
import energy.loss as loss
import energy.evaluation as evaluation

import energy.model as model

import persistence.layer as persistence

# Classes of methods
base_vers = ['TransE', 'ScalE', 'NTransE', 'NScalE'] + ['BiTransE', 'BiScalE', 'BiNTransE', 'BiNScalE']
xi_vers = ['XiTransE', 'XiScalE', 'XiNTransE', 'XiNScalE']
xiscaltrans_vers = ['XiScalTransE', 'XiNScalTransE', 'XiTransScalE', 'XiNTransScalE']
semixi_vers = ['XiScalTransSE', 'XiTransScalSE', 'XiN1ScalTransSE']
lc_vers = ['CeTransE', 'CrTransE', 'CerTransE']
scaltrans_vers = ['ScalTransE', 'NScalTransE', 'BiScalTransE', 'BiNScalTransE']
aff_vers = ['AffinE', 'NAffinE', 'BiAffinE', 'BiNAffinE']
xiaff_vers = ['XiAffinE', 'XiNAffinE']

# Experiment function
#@profile
def learn(state):
    np.random.seed(state.seed)

    c = util.configuration()
    layer, exp, exp_id = None, {}, None

    dataset = util.ExpDataSet(train_path=state.train_path, valid_path=state.valid_path, test_path=state.test_path)

    # Training set
    trainl, trainr, traino = dataset.train()

    # Validation set
    if dataset.has_valid is True:
        validl, validr, valido = dataset.valid()

    # Test set
    if dataset.has_test is True:
        testl, testr, testo = dataset.test()

    logging.info('Shape for training set: %s' % (str(trainl.shape)))

    if state.use_db:
        is_fast = trainl.shape[1] > 10000000 # if the dataset is not small-sized (> 10m triples), switch to fast mode
        layer = persistence.PickleLayer(dir=c.get('Persistence', 'path'), is_fast=is_fast)
        exp = {
            'start_time': datetime.datetime.utcnow()
        }
        exp_id = layer.create(state.name, exp)

    NE, NP, NR = 14951, 2430, 1345
    state.Nrel = NP
    state.Nent = NE + NP

    if state.Nsyn is None:
        state.Nsyn = NE

    if dataset.specs is not None:
        state.Nleft, state.Nright, state.Nshared = dataset.specs['Nleft'], dataset.specs['Nright'], dataset.specs['Nshared']

    exp['best'] = {} # use the validation set (if available) to pick the best model
    exp['state'] = { k: (state[k].__name__ if isinstance(state[k], types.FunctionType) else str(state[k])) for k in state.keys() }
    exp['producer'] = util.producer(c)

    # Show experiment parameters
    logging.info('State: %s', exp['state'])

    if state.op in ['SE'] + base_vers + semixi_vers + xi_vers + xiscaltrans_vers + lc_vers + scaltrans_vers + aff_vers + xiaff_vers:
        traino = traino[-state.Nrel:, :] # last elements of traino
        if dataset.has_valid is True:
            valido = valido[-NR:, :]
        if dataset.has_test is True:
            testo = testo[-NR:, :]

    logging.debug('Converting sparse matrices to indexes ..')
    # pdb.set_trace()
    # Convert sparse matrices to indexes
    trainlidx, trainridx, trainoidx = util.convert2idx(trainl), util.convert2idx(trainr), util.convert2idx(traino)

    if dataset.has_valid is True:
        validlidx, validridx, validoidx = util.convert2idx(validl), util.convert2idx(validr), util.convert2idx(valido)

    if dataset.has_test is True:
        testlidx, testridx, testoidx = util.convert2idx(testl), util.convert2idx(testr), util.convert2idx(testo)

    true_triples = None
    if (dataset.has_valid is True) and (dataset.has_test is True) and state.filtered:
        true_triples = np.concatenate([testlidx, validlidx, trainlidx, testoidx, validoidx, trainoidx, testridx, validridx, trainridx]).reshape(3, testlidx.shape[0] + validlidx.shape[0] + trainlidx.shape[0]).T

    # Operators
    leftop, rightop = model.op(state.op, state.ndim, state.nhid)

    logging.debug('Initializing the embeddings ..')

    # Embeddings
    embeddings = learning.Embeddings(np.random, state.Nent, state.ndim, tag='emb')

    relationVec = None

    if (state.op in ['SE']) and type(embeddings) is not list:
        relationl = learning.Embeddings(np.random, state.Nrel, state.ndim * state.nhid, tag='rell')
        relationr = learning.Embeddings(np.random, state.Nrel, state.ndim * state.nhid, tag='relr')
        embeddings = [embeddings, relationl, relationr]

    elif (state.op in base_vers + lc_vers) and type(embeddings) is not list:
        relationVec = learning.Embeddings(np.random, state.Nrel, state.ndim, tag='relvec')
        embeddings = [embeddings, relationVec, relationVec]

    elif (state.op in xi_vers) and type(embeddings) is not list:
        relationVec = learning.Embeddings(np.random, state.Nrel, state.ndim * 2, tag='relvec')
        embeddings = [embeddings, relationVec, relationVec]

    elif (state.op in scaltrans_vers) and type(embeddings) is not list:
        scaleTranslateVec = learning.Embeddings(np.random, state.Nrel, state.ndim * 2, tag='scaleTranslateVec')
        embeddings = [embeddings, scaleTranslateVec, scaleTranslateVec] # x, w, d

    elif (state.op in xiscaltrans_vers) and type(embeddings) is not list:
        scaleTranslateVec = learning.Embeddings(np.random, state.Nrel, state.ndim * 4, tag='scaleTranslateVec')
        embeddings = [embeddings, scaleTranslateVec, scaleTranslateVec] # x, w, d

    elif (state.op in semixi_vers) and type(embeddings) is not list:
        scaleTranslateVec = learning.Embeddings(np.random, state.Nrel, state.ndim * 3, tag='scaleTranslateVec')
        embeddings = [embeddings, scaleTranslateVec, scaleTranslateVec] # x, w, d

    elif (state.op in aff_vers) and type(embeddings) is not list:
        affineVec = learning.Embeddings(np.random, state.Nrel, (state.ndim * state.nhid), tag='affineVec')
        embeddings = [embeddings, affineVec, affineVec]

    elif (state.op in xiaff_vers) and type(embeddings) is not list:
        affineVec = learning.Embeddings(np.random, state.Nrel, (state.ndim * state.nhid) * 2, tag='affineVec')
        embeddings = [embeddings, affineVec, affineVec]

    with open('./fb15k_margin/fb15k_TransC_ap/best_valid_model.pkl','rb') as f:
        embeddings = pickle.load(f)

    simfn = state.simfn

    marw = learning.Embeddings(np.random, 1, 1, 'margeweight')
    marb = learning.Embeddings(np.random, state.Nrel, 1, 'margebias')

    f = open('density_rel_C.pkl','rb')
    dens = pickle.load(f)
    f.close()
    # den = np.array(density)
    # pdb.set_trace()
    density = learning.Embeddings(np.random, state.Nrel, 1, 'emb')
    for r in range(state.Nrel):
        density.E.container.storage[0][0,r] = dens[r]

    logging.debug('Initializing the training function ..')

    # Functions compilation
    trainfunc = learning.TrainFn1Member(simfn, embeddings, density, marw, marb, leftop, rightop, rel=True,
                                        method=state.method, op=state.op, loss=loss.hinge, loss_margin=state.loss_margin,
                                        decay=state.decay, epsilon=state.epsilon, max_learning_rate=state.max_lr,
                                        weight_L1_param_regularizer=state.l1_param_weight, weight_L2_param_regularizer=state.l2_param_weight)

    # FB has some specific parameters for RankRightFnIdx:
    l_subtensorspec = state.Nsyn
    r_subtensorspec = state.Nsyn
    if dataset.specs is not None:
        r_subtensorspec = dataset.specs['Nright'] + dataset.specs['Nshared']

    ranklfunc = evaluation.RankLeftFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=l_subtensorspec)
    rankrfunc = evaluation.RankRightFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=r_subtensorspec)
    rankofunc = evaluation.RankRelFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=r_subtensorspec)

    ranklfunc_filtered = evaluation.RankLeftFnIdx_filtered(simfn, embeddings, leftop, rightop, subtensorspec=l_subtensorspec)
    rankrfunc_filtered = evaluation.RankRightFnIdx_filtered(simfn, embeddings, leftop, rightop, subtensorspec=r_subtensorspec)

    out, outb = [], []
    os.mkdir(state.name)
    os.mkdir(state.name+'margel')
    os.mkdir(state.name+'marger')
    os.mkdir(state.name+'embedding')

    train_mrs, train_hits = [], []              # Mean Rank and Hits@10 for every state.test_all Epoch
    valid_mrs, valid_hits = [], []              # Mean Rank and Hits@10 for every state.test_all Epoch
    test_mrs, test_hits = [], []                # Mean Rank and Hits@10 for every state.test_all Epoch

    state.bestvalid, state.besttest = None, None
    state.bestepoch = None

    with open('./subrel2rel_apC.pkl','rb') as f:
        subrel2rel = pickle.load(f)
    with open('./rel2subrel_apC.pkl','rb') as g:
        rel2subrel = pickle.load(g)

    batchsize = trainl.shape[1] / state.nbatches

    logging.info("Starting the Experiment ..")
    timeref = time.time()

    average_costs_per_epoch = []                                        # X
    ratios_violating_examples_per_epoch = []                            # X

    for epoch_count in range(1, state.totepochs + 1):

        logging.debug('Running epoch %d of %d ..' % (epoch_count, state.totepochs))

        # Shuffling
        order = np.random.permutation(trainl.shape[1])

        # Note: this is painfully slow when (trainl, trainr, traino) are lil_matrix
        trainl, trainr, traino = trainl[:, order], trainr[:, order], traino[:, order]

        logging.debug('Creating negative examples ..')

        trainln_arange = np.arange(state.Nsyn)
        trainrn_arange = np.arange(state.Nsyn)
        trainon_arange = np.arange(traino.shape[0])
        # the FB dataset has some specific settings
        if dataset.specs is not None:
            trainln_arange = np.arange(dataset.specs['Nright'] + dataset.specs['Nshared'])
            trainrn_arange = np.arange(dataset.specs['Nright'], dataset.specs['Nright'] + dataset.specs['Nshared'] + dataset.specs['Nleft'])

        trainln = util.create_random_mat(trainl.shape, trainln_arange)
        trainrn = util.create_random_mat(trainr.shape, trainrn_arange)
        trainon = util.create_random_mat(traino.shape, trainon_arange)

        epoch_average_costs = []                                        # X
        epoch_ratios_violating_examples = []                            # X

        for i in range(state.nbatches): # Iterate over Batches

            logging.debug('Running on batch %d of %d ..' % (i, state.nbatches))

            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]

            tmpln = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmprn = trainrn[:, i * batchsize:(i + 1) * batchsize]
            tmpon = trainon[:, i * batchsize:(i + 1) * batchsize]

            logging.debug('Executing the training function ..')

            _lrparam = state.lrparam / float(batchsize)
            if state.no_rescaling is True:
                _lrparam = state.lrparam

            # training iteration
            outtmp = trainfunc(state.lremb, _lrparam, tmpl, tmpr, tmpo, tmpln, tmprn, tmpon)

            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]

            average_cost = outtmp[0]                                        # X
            ratio_violating_examples = outtmp[1]                            # X

            epoch_average_costs += [average_cost]                           # X
            epoch_ratios_violating_examples += [ratio_violating_examples]   # X

            logging.debug('Normalizing the embeddings ..')

            # embeddings normalization
            if type(embeddings) is list:
                embeddings[0].normalize() # normalize e
            else:
                embeddings.normalize()
        marw.normalize()
        marb.normalize()

        # End of Epoch

        logging.info("-- EPOCH %s (%s seconds):" % (epoch_count, round(time.time() - timeref, 3)))

        average_costs_per_epoch += [epoch_average_costs]                                    # X
        ratios_violating_examples_per_epoch += [epoch_ratios_violating_examples]            # X

        exp['average_costs_per_epoch'] = average_costs_per_epoch                            # X
        exp['ratios_violating_examples_per_epoch'] = ratios_violating_examples_per_epoch    # X

        # Model Evaluation

        logging.info("COST >> %s +/- %s, %% updates: %s%%" % (round(np.mean(out), 4), round(np.std(out), 4), round(np.mean(outb) * 100, 3)))

        # Check if NaN
        if np.isnan(np.mean(out)):
            logging.error('NaN propagation detected!')
            return

        out, outb = [], []

        # Evaluate the Ranking Score each test_all epochs
        if (state.test_all is not None) and ((epoch_count % state.test_all) == 0):

            valid_summary = None
            state.valid = None

            # Evaluation on the Validation Set
            if dataset.has_valid is True:

                if state.ranking_score_right:
                    resvalid = evaluation.RankingScoreRightIdx(rankrfunc, validlidx, validridx, validoidx)
                    valid_summary = evaluation.ranking_summary_right(resvalid, idxo=validoidx, tag='raw valid')
                    state.valid = np.mean(resvalid)
                else:
                    resvalid = evaluation.RankingScoreIdx_test(ranklfunc, rankrfunc, validlidx, validridx, validoidx, rel2subrel)
                    valid_summary = evaluation.ranking_summary(resvalid, idxo=validoidx, tag='raw valid')
                    state.valid = np.mean(resvalid[0] + resvalid[1])

                    if (state.filtered):
                        resvalid_filtered = evaluation.FilteredRankingScoreIdx_test(ranklfunc, rankrfunc, validlidx, validridx, validoidx, true_triples, rel2subrel)
                        valid_summary_filtered = evaluation.ranking_summary(resvalid_filtered, idxo=validoidx, tag='filtered valid')

            test_summary = None
            state.test = None

            # Evaluation on the Test Set
            if dataset.has_test is True:

                if state.ranking_score_right:
                    restest = evaluation.RankingScoreRightIdx(rankrfunc, testlidx, testridx, testoidx)
                    test_summary = evaluation.ranking_summary_right(restest, idxo=testoidx, tag='raw test')
                    state.test = np.mean(restest)
                else:
                    # entity prediction
                    restest = evaluation.RankingScoreIdx_test(ranklfunc, rankrfunc, testlidx, testridx, testoidx, rel2subrel)
                    for en in [1,3,10]:
                        test_summary = evaluation.ranking_summary(restest, idxo=testoidx, n=en, tag='raw test')
                    state.test = np.mean(restest[0] + restest[1])

                    if (state.filtered):
                        restest_filtered = evaluation.FilteredRankingScoreIdx_test(ranklfunc, rankrfunc, testlidx, testridx, testoidx, true_triples, rel2subrel)
                        for en in [1,3,10]:
                            valid_summary_filtered = evaluation.ranking_summary(restest_filtered, idxo=testoidx, n=en, tag='filtered test')
                    # relation prediction
                    restest1 = evaluation.RankingScoreRelIdx_test(rankofunc, testlidx, testridx, testoidx, rel2subrel)
                    for en in [1,3,10]:
                        test_summary = evaluation.ranking_summary_rel(restest1, n=en, tag='raw test')
                    if (state.filtered):
                        restest_filtered1 = evaluation.FilteredRankingScoreRelIdx_test(rankofunc, testlidx, testridx, testoidx, true_triples, rel2subrel)
                        for en in [1,3,10]:
                            valid_summary_filtered = evaluation.ranking_summary_rel(restest_filtered1, n=en, tag='filtered test')


            save_model = True
            if dataset.has_valid is True:
                save_model = False
                if state.bestvalid == None or state.valid < state.bestvalid:
                    save_model = True

            if save_model is True:
                with open(state.name+'best_valid_model.pkl','wb') as f:
                    pickle.dump(embeddings, f, -1)
                if (epoch_count % 200) == 0:
                    f = open(state.name+'margel/'+str(epoch_count)+'.pkl','wb')
                    pickle.dump(outtmp[2], f, -1)
                    f.close()
                    g = open(state.name+'marger/'+str(epoch_count)+'.pkl','wb')
                    pickle.dump(outtmp[3], g, -1)
                    g.close()
                    p = open(state.name+'embedding/'+str(epoch_count)+'.pkl','wb')
                    pickle.dump(embeddings, p, -1)
                    p.close()

                if dataset.has_valid is True:
                    state.bestvalid = state.valid
                    exp['best_valid'] = state.bestvalid

                if dataset.has_test is True:
                    state.besttest = state.test
                    exp['best_test'] = state.besttest

                state.bestepoch = epoch_count
                exp['best_epoch'] = state.bestepoch

                # Save the Best Model (on the Validation Set) using the Persistence Layer
                embs = [e.E for e in embeddings] if (type(embeddings) == list) else [embeddings.E]
                model_params = embs + leftop.params + rightop.params + (simfn.params if hasattr(simfn, 'params') else [])

                model_param_values = {}
                for model_param in set(model_params):
                    value = {
                        'value': model_param.get_value().tolist(),
                        'shape': model_param.get_value().shape
                    }
                    model_param_values[str(model_param)] = value

                best_model = {
                    'parameters': model_param_values,
                    'epoch': epoch_count,
                    'entities': dataset.entities,
                    'predicates': dataset.predicates,

                    'valid_summary': valid_summary,
                    'test_summary': test_summary
                }

                if dataset.resources is not None:
                    best_model['resources'] = dataset.resources
                    best_model['bnodes'] = dataset.bnodes
                    best_model['literals'] = dataset.literals

                exp['best'] = best_model


        if state.use_db:
            layer.update(exp_id, exp)

        timeref = time.time()
    return


def launch(op='TransE', simfn=similarity.dot, ndim=20, nhid=20, Nsyn=None,
        test_all=1, use_db=False, seed=666,
        method='SGD', lremb=0.01, lrparam=0.1, no_rescaling=False, filtered=False,
        loss_margin=1.0, decay=0.999, epsilon=1e-6, max_lr=None, nbatches=100, totepochs=2000, C=1.0, name='tmp',
        l1_param_weight=None, l2_param_weight=None, ranking_score_right=False,
        train_path=None, valid_path=None, test_path=None):

    # Argument of the experiment script
    state = util.DD()

    state.name = name
    state.train_path = train_path
    state.valid_path = valid_path
    state.test_path = test_path

    state.method = method
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.Nsyn = Nsyn

    state.loss_margin = loss_margin
    state.test_all = test_all
    state.use_db = use_db

    state.lremb = lremb
    state.lrparam = lrparam
    state.no_rescaling = no_rescaling
    state.filtered = filtered

    state.decay = decay
    state.epsilon = epsilon
    state.max_lr = max_lr
    state.C = C

    state.l1_param_weight = l1_param_weight
    state.l2_param_weight = l2_param_weight
    state.ranking_score_right = ranking_score_right

    state.nbatches = nbatches
    state.totepochs = totepochs
    state.seed = seed

    learn(state)


def main(argv):

    name = 'eval'

    train_path = 'data/fb15k/FB15k-train.pkl'
    valid_path = None
    test_path = None

    lr = 1.0
    lremb, lrparam = None, None
    no_rescaling = False
    filtered = False

    margin = 1.0
    decay, epsilon = 0.999, 1e-6
    C, max_lr = 1.0, None

    sim_str = None
    method = 'SGD'
    op = 'TransE'

    ndim, nhid = 50, 50
    nbatches = 10
    totepochs = 500

    test_all, use_db = None, False
    seed = 123

    l1_param_weight, l2_param_weight = None, None
    ranking_score_right = False

    usage_str = ("""Usage: %s [-h]
                    [--name=<name>] [--train=<path>] [--valid=<path>] [--test=<path>] [--use_db]
                    [--sim=<sim>] [--op=<op>] [--strategy=<strategy>] [--ndim=<ndim>] [--nhid=<nhid>]
                    [--lr=<lr>] [--lremb=<lremb>] [--lrparam=<lrparam>] [--no_rescaling] [--filtered]
                    [--margin=<margin>] [--decay=<decay>] [--epsilon=<epsilon>] [--max_lr=<max_lr>] [--C=<C>]
                    [--l1_param=<weight>] [--l2_param=<weight>] [--ranking_score_right]
                    [--nbatches=<nbatches>] [--test_all=<test_all>] [--totepochs=<totepochs>]  [--seed=<seed>]
                    """ % (sys.argv[0]))

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', [ 'name=', 'train=', 'valid=', 'test=', 'use_db',
                                                'sim=', 'op=', 'strategy=', 'ndim=', 'nhid=',
                                                'lr=', 'lremb=', 'lrparam=', 'no_rescaling', 'filtered',
                                                'margin=', 'decay=', 'epsilon=', 'max_lr=', 'C=',
                                                'l1_param=', 'l2_param=', 'ranking_score_right',
                                                'nbatches=', 'test_all=', 'totepochs=', 'seed=' ])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            logging.info(usage_str)
            logging.info('\t--name=<name> (default: %s)' % (name))
            logging.info('\t--train=<path> (default: %s)' % (train_path))
            logging.info('\t--valid=<path> (default: %s)' % (valid_path))
            logging.info('\t--test=<path> (default: %s)' % (test_path))
            logging.info('\t--use_db (use a persistence layer -- default: %s)' % (use_db))

            logging.info('\t--sim=<sim> (default: %s)' % (sim_str))
            logging.info('\t--op=<op> (default: %s)' % (op))
            logging.info('\t--strategy=<strategy> (default: %s)' % (method))
            logging.info('\t--ndim=<ndim> (default: %s)' % (ndim))
            logging.info('\t--nhid=<nhid> (default: %s)' % (nhid))

            logging.info('\t--lr=<lr> (default: %s)' % (lr))
            logging.info('\t--lremb=<lremb> (default: %s)' % (lr))
            logging.info('\t--lrparam=<lrparam> (default: %s)' % (lr))
            logging.info('\t--no_rescaling (default: %s)' % (no_rescaling))
            logging.info('\t--filtered (default: %s)' % (filtered))

            logging.info('\t--margin=<margin> (default: %s)' % (margin))
            logging.info('\t--decay=<decay> (default: %s)' % (decay))
            logging.info('\t--epsilon=<epsilon> (default: %s)' % (epsilon))
            logging.info('\t--max_lr=<max_lr> (default: %s)' % (max_lr))
            logging.info('\t--C=<C> (default: %s)' % (C))

            logging.info('\t--l1_param=<weight> (default: %s)' % (l1_param_weight))
            logging.info('\t--l2_param=<weight> (default: %s)' % (l2_param_weight))
            logging.info('\t--ranking_score_right (default: %s)' % (ranking_score_right))

            logging.info('\t--nbatches=<nbatches> (default: %s)' % (nbatches))
            logging.info('\t--test_all=<test_all> (default: %s)' % (test_all))
            logging.info('\t--totepochs=<totepochs> (default: %s)' % (totepochs))
            logging.info('\t--seed=<seed> (default: %s)' % (seed))
            return

        if opt == '--name':
            name = arg
        elif opt == '--train':
            train_path = arg
        elif opt == '--valid':
            valid_path = arg
        elif opt == '--test':
            test_path = arg
        elif opt == '--use_db':
            use_db = True

        elif opt == '--sim':
            sim_str = arg
        elif opt == '--op':
            op = arg
        elif opt == '--strategy':
            method = arg
        elif opt == '--ndim':
            ndim = int(arg)
        elif opt == '--nhid':
            nhid = int(arg)

        elif opt == '--lr':
            lr = float(arg)
        elif opt == '--lremb':
            lremb = float(arg)
        elif opt == '--lrparam':
            lrparam = float(arg)
        elif opt == '--no_rescaling':
            no_rescaling = True
        elif opt == '--filtered':
            filtered = True

        elif opt == '--margin':
            margin = float(arg)
        elif opt == '--decay':
            decay = float(arg)
        elif opt == '--epsilon':
            epsilon = float(arg)
        elif opt == '--max_lr':
            max_lr = float(arg)
        elif opt == '--C':
            C = float(arg)

        elif opt == '--l1_param':
            l1_param_weight = float(arg)
        elif opt == '--l2_param':
            l2_param_weight = float(arg)
        elif opt == '--ranking_score_right':
            ranking_score_right = True

        elif opt == '--nbatches':
            nbatches = int(arg)
        elif opt == '--test_all':
            test_all = int(arg)
        elif opt == '--totepochs':
            totepochs = int(arg)
        elif opt == '--seed':
            seed = int(arg)

    if lremb is None:
        lremb = lr
    if lrparam is None:
        lrparam = lr

    if sim_str is None:
        sim_str = 'dot'
        if op in base_vers + lc_vers + scaltrans_vers + aff_vers + xiaff_vers:
            # In TransE and other, d(x, y) = ||x - y||_1
            sim_str = 'L1'

    simfn = getattr(similarity, sim_str)
    # t1 = time()
    launch(op=op, simfn=simfn, method=method, seed=seed, totepochs=totepochs,
            name=name, train_path=train_path, valid_path=valid_path, test_path=test_path,
            test_all=test_all, use_db=use_db,
            ndim=ndim, nhid=nhid, nbatches=nbatches, Nsyn=None,
            lremb=lremb, lrparam=lrparam, no_rescaling=no_rescaling, filtered=filtered,
            loss_margin=margin, epsilon=epsilon, decay=decay, max_lr=max_lr, C=C,
            l1_param_weight=l1_param_weight, l2_param_weight=l2_param_weight,
            ranking_score_right=ranking_score_right)
    # t2 = time()
    # print "Training time:", t2-t1

if __name__ == '__main__':
    t1 = time.time()
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
    t2 = time.time()
    print "Training time:", t2-t1
