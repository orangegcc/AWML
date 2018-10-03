# -*- coding: utf-8 -*-

import numpy as np
import pickle
import theano
import theano.tensor as T

import logging

from sklearn import metrics
from sparse.learning import parse_embeddings


def auc_pr(predictions=[], labels=[]):
    '''Computes the Area Under the Precision-Recall Curve (AUC-PR)'''
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions)
    auc = metrics.auc(recall, precision)
    return auc

def auc_pr_orig(predictions=[], labels=[]):
    pred = np.asarray(predictions)
    lab = np.asarray(labels)

    order = np.argsort(pred)
    lab_ordered = lab[order]
    pred_ordered = pred[order]

    precision = {}
    recall = {}
    # All examples are classified 1
    precision[np.min(pred_ordered) - 1.0] = (np.sum(lab_ordered) /
            float(len(lab)))
    recall[np.min(pred_ordered) - 1.0] = 1.
    for i in range(len(lab)):
        if len(lab) - i - 1 == 0:
            # No examples are classified 1
            precision[pred_ordered[i]] = 1
        else:
            precision[pred_ordered[i]] = (np.sum(lab_ordered[i + 1:]) /
                    float(len(lab) - i - 1))
        recall[pred_ordered[i]] = (np.sum(lab_ordered[i + 1:]) /
                float(np.sum(lab_ordered)))

    # Precision-Recall curve points
    points = []
    for i in np.sort(precision.keys())[::-1]:
        points += [(float(recall[i]), float(precision[i]))]
    # Compute area
    auc = sum((y0 + y1) / 2. * (x1 - x0) for (x0, y0), (x1, y1) in
            zip(points[:-1], points[1:]))
    return auc

def auc_roc(predictions=[], labels=[]):
    '''Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC)'''
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    precision, recall, threshold = metrics.roc_curve(labels, predictions)
    auc = metrics.auc(recall, precision)
    return auc

#
# COMPUTING PERFORMANCE METRICS ON RANKINGS
#

#
# Evaluation summary (as in FB):
#
def ranking_summary_right(res, idxo=None, n=10, tag='raw'):
    dres = {}
    dres.update({'micrormean': np.mean(res)})
    dres.update({'micrormedian': np.median(res)})
    dres.update({'microrr@n': np.mean(np.asarray(res) <= n) * 100})

    logging.info('### MICRO (%s):' % (tag))
    logging.info('\t-- right  >> mean: %s, median: %s, r@%s: %s%%' % (
                    round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                    n, round(dres['microrr@n'], 3)))

    if idxo is not None:

        listrel = set(idxo)
        dictrelres = {}
        dictrelrmean = {}
        dictrelrmedian = {}
        dictrelrrn = {}

        for i in listrel:
            dictrelres.update({i: []})

        for i, j in enumerate(res):
            dictrelres[idxo[i]] += [j]

        for i in listrel:
            dictrelrmean[i] = np.mean(dictrelres[i])
            dictrelrmedian[i] = np.median(dictrelres[i])
            dictrelrrn[i] = np.mean(np.asarray(dictrelres[i]) <= n) * 100

        dres.update({'dictrelres': dictrelres})
        dres.update({'dictrelrmean': dictrelrmean})
        dres.update({'dictrelrmedian': dictrelrmedian})
        dres.update({'dictrelrrn': dictrelrrn})

        dres.update({'macrormean': np.mean(dictrelrmean.values())})
        dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
        dres.update({'macrorr@n': np.mean(dictrelrrn.values())})

        logging.info('### MACRO (%s):' % (tag))
        logging.info('\t-- right  >> mean: %s, median: %s, r@%s: %s%%' % (
                        round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
                        n, round(dres['macrorr@n'], 3)))

    return dres

#
# Evaluation summary (as in FB15k):
# relation prediction
def ranking_summary_rel(res, idxo=None, n=10, tag='raw'):
    resg = res
    dres = {}
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})
    logging.info('### MICRO (%s):' % (tag))
    logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' % (                                                 round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                    n, round(dres['microghits@n'], 3)))
    return dres


# entity prediction
def ranking_summary(res, idxo=None, n=10, tag='raw'):
    resg = res[0] + res[1]
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    logging.info('### MICRO (%s):' % (tag))
    logging.info('\t-- left   >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
                    n, round(dres['microlhits@n'], 3)))
    logging.info('\t-- right  >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                    n, round(dres['microrhits@n'], 3)))
    logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                    n, round(dres['microghits@n'], 3)))

    if idxo is not None:
        listrel = set(idxo)
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
            dictrelres[idxo[i]][0] += [j]

        for i, j in enumerate(res[1]):
            dictrelres[idxo[i]][1] += [j]

        for i in listrel:
            dictrellmean[i] = np.mean(dictrelres[i][0])
            dictrelrmean[i] = np.mean(dictrelres[i][1])
            dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
            dictrellmedian[i] = np.median(dictrelres[i][0])
            dictrelrmedian[i] = np.median(dictrelres[i][1])
            dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
            dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
            dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
            dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] + dictrelres[i][1]) <= n) * 100

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

        logging.info('### MACRO (%s):' % (tag))
        logging.info('\t-- left   >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
                        n, round(dres['macrolhits@n'], 3)))
        logging.info('\t-- right  >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
                        n, round(dres['macrorhits@n'], 3)))
        logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
                        n, round(dres['macroghits@n'], 3)))

    return dres

#
# RANKING FUNCTIONS
#
def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl, idxo = T.iscalar('idxl'), T.iscalar('idxo')

    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))              # lhs: 1xD vector containing the embedding of idxl

    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T                                             # rhs: NxD embedding matrix

    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))             # rell: 1xD vector containing the embedding of idxo (relationl)
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))             # relr: 1xD vector containing the embedding of idxo (relationr)

    tmp = leftop(lhs, rell)                                             # a = rell(lhs)
                                                                        # b = relr(rhs)

    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))    # simi = fnsim(a, b)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')

def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr, idxo = T.iscalar('idxr'), T.iscalar('idxo')

    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))

    tmp = rightop(rhs, relr)

    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi], on_unused_input='ignore')

def RankRelFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    #embedding, relationl, relationr = parse_embeddings(embeddings)
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxr')
    idxl = T.iscalar('idxl')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    if subtensorspec is not None:
    # We compute the score only for a subset of entities
        rell = (relationl.E[:, :subtensorspec]).T
        relr = (relationr.E[:, :subtensorspec]).T
    else:
        rell = embedding.E.T
        relr = embedding.E.T
    # hack to prevent a broadcast problem with the Bilinear layer
    if hasattr(leftop, 'forwardrankrel'):
        tmpleft = leftop.forwardrankrel(lhs, rell)
    else:
        tmpleft = leftop(lhs, rell)
    if hasattr(rightop, 'forwardrankrel'):
        tmpright = rightop.forwardrankrel(rhs, relr)
    else:
        tmpright = rightop(rhs, relr)
    simi = fnsim(tmpleft, tmpright)
    return theano.function([idxl, idxr], [simi], on_unused_input='ignore')


def RankingScoreIdx(sl, sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []

    for l, o, r in zip(idxl, idxo, idxr):
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

    
def RankingScoreIdx_test(sl, sr, idxl, idxr, idxo, rel2subrel):
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr): #tuple   #此处的r是0~1344原始关系字典中的序号
        subrel = rel2subrel[o+40943]
        j = 0
        for so in subrel:
            scores_l = (sl(r, so-40943)[0]).flatten()
            el = np.argsort(np.argsort(-scores_l)).flatten()[l] + 1
            if j == 0:
                errorl = el
            else:
                if el < errorl:
                    errorl = el
            scores_r = (sr(l, so-40943)[0]).flatten()
            er = np.argsort(np.argsort(-scores_r)).flatten()[r] + 1
            if j == 0:
                errorr = er
            else:
                if er < errorr:
                    errorr = er
            j += 1
     
        errl += [errorl]
        errr += [errorr]
    return errl, errr


def RankingScoreRelIdx(so, idxl, idxr, idxo):
    erro = []

    for l, o, r in zip(idxl, idxo, idxr):

        erro += [np.argsort(np.argsort((so(l, r)[0]).flatten())[::-1]).flatten()[o] + 1]

    return erro


def RankingScoreRelIdx_test(so, idxl, idxr, idxo, rel2subrel):
    erro = []
    for l, o, r in zip(idxl, idxo, idxr): #tuple   #此处的r是0~1344原始关系字典中的序号
        subrel = rel2subrel[o+40943]
        j = 0
        scores_o = (so(l, r)[0]).flatten()
        for s in subrel:
            eo = np.argsort(np.argsort(-scores_o)).flatten()[s-40943] + 1
            if j == 0:
                erroro = eo
            else:
                if eo < erroro:
                    erroro = eo
            j += 1
        erro += [erroro]
    return erro


def RankLeftFnIdx_filtered(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr, idxo = T.iscalar('idxr'), T.iscalar('idxo')
    leftparts = T.ivector('leftparts')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    lhs = lhs[leftparts, :]                                             # select the left parts not appearing
                                                                        # in the train/valid/test sets

    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))

    tmp = rightop(rhs, relr)

    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo, leftparts], [simi], on_unused_input='ignore')

def RankRightFnIdx_filtered(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)
    # Inputs
    idxl, idxo = T.iscalar('idxl'), T.iscalar('idxo')
    rightparts = T.ivector('rightparts')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))              # lhs: 1xD vector containing the embedding of idxl
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T                                             # rhs: NxD embedding matrix

    rhs = rhs[rightparts, :]                                            # select the right parts not appearing
                                                                        # in the train/valid/test sets

    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))             # rell: 1xD vector containing the embedding of idxo (relationl)
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))             # relr: 1xD vector containing the embedding of idxo (relationr)

    tmp = leftop(lhs, rell)                                             # a = rell(lhs)
                                                                        # b = relr(rhs)

    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))    # simi = fnsim(a, b)
    return theano.function([idxl, idxo, rightparts], [simi], on_unused_input='ignore')

def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.
    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)

        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr

def FilteredRankingScoreIdx_test(sl, sr, idxl, idxr, idxo, true_triples, rel2subrel):

    errl = []
    errr = []
    f = open('./cluster_wn/k.pkl','rb')
    k = pickle.load(f)
    f.close()
    for l, o, r in zip(idxl, idxo, idxr): #tuple   #此处的r是0~1344原始关系字典中的序号
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,) #返回需要测试的三元组的头实体、关系、尾实体>分别在所有正例三元组中的序号，得到il、io、ir
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
        subrel = rel2subrel[o+40943]
        j = 0
        for so in subrel:
            io1 = np.argwhere(true_triples[10000:,1]==so-40943).reshape(-1,)
            io2 = np.argwhere(true_triples[0:10000,1]==o).reshape(-1,)
            io = np.concatenate((io1, io2), axis = 0)
            inter_l = [i for i in ir if i in io]
            rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
            scores_l = (sl(r, so-40943)[0]).flatten()
            scores_l[rmv_idx_l] = -np.inf
            el = np.argsort(np.argsort(-scores_l)).flatten()[l] + 1
            if j == 0:
                errorl = el
            else:
                if el < errorl:
                    errorl = el
            inter_r = [i for i in il if i in io]
            rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
            scores_r = (sr(l, so-40943)[0]).flatten()
            scores_r[rmv_idx_r] = -np.inf
            er = np.argsort(np.argsort(-scores_r)).flatten()[r] + 1
            if j == 0:
                errorr = er
            else:
                if er < errorr:
                    errorr = er
            j += 1

        errl += [errorl]
        errr += [errorr]

    return errl, errr


    
def FilteredRankingScoreRelIdx(so, idxl, idxr, idxo, true_triples):
    erro = []
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)

        inter_o = [i for i in ir if i in il]
        rmv_idx_o = [true_triples[i,1] for i in inter_o if true_triples[i,1] != o]
        scores_o = (so(l, r)[0]).flatten()
        scores_o[rmv_idx_o] = -np.inf
        erro += [np.argsort(np.argsort(-scores_o)).flatten()[o] + 1]
    return erro


def FilteredRankingScoreRelIdx_test(so, idxl, idxr, idxo, true_triples, rel2subrel):
    erro = []
    f = open('./cluster_wn/k.pkl','rb')
    k = pickle.load(f)
    f.close()
    for l, o, r in zip(idxl, idxo, idxr): #tuple   #此处的o是0~1344原始关系字典中的序号
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,) #返回需要测试的三元组的头实体、关系、尾实体>分别在所有正例三元组中的序号，得到il、io、ir
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
        subrel = rel2subrel[o+40943]
        j = 0
        inter_o = [i for i in ir if i in il]
        rmv_idx_o = [true_triples[i,1] for i in inter_o if (true_triples[i,1]+40943) not in subrel]
        scores_o = (so(l, r)[0]).flatten()
        scores_o[rmv_idx_o] = -np.inf
        for s in subrel:
            eo = np.argsort(np.argsort(-scores_o)).flatten()[s-40943] + 1
            if j == 0:
                erroro = eo
            else:
                if eo < erroro:
                    erroro = eo
            j += 1
        erro += [erroro]
    return erro


def RankingScoreIdx_filtered(sl, sr, idxl, idxr, idxo,
                                l_subtensorspec, r_subtensorspec,
                                true_triples=[]):
    errl, errr = [], []

    for l, o, r in zip(idxl, idxo, idxr):

        # Remove the current triple from the true triples list
        _true_triples = [x for x in true_triples if x is not (l, o, r)]
        #_true_triples = true_triples

        # Generate new possible left/right parts, removing those appearning in the true triples list
        leftparts = np.array([i for i in range(l_subtensorspec) if (i, o, r) not in _true_triples], dtype='int32')
        rightparts = np.array([i for i in range(r_subtensorspec) if (l, o, i) not in _true_triples], dtype='int32')

        sl_ro, sr_lo = sl(r, o, leftparts), sr(l, o, rightparts)

        errl += [np.argsort(np.argsort((sl_ro[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr_lo[0]).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

def RankingScoreIdx_sub(sl, sr, idxl, idxr, idxo, selection=[]):
    """
    Similar to RankingScoreIdx, but works on a subset of examples, defined in
    the 'selection' parameter.
    """
    errl, errr = [], []

    #for l, o, r in zip(idxl, idxo, idxr):
    for l, o, r in [(idxl[i], idxo[i], idxr[i]) for i in selection]:
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

def RankingScoreRightIdx(sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the rhs, over a list of lhs, rhs
    and rel indexes.

    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
    return errr
