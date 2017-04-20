'''
Description - This script implements siamese networks in theano, providing dynamic time warping as the distance measure

Author - James O' Neill
'''
import numpy as np
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances

def siamese_euclidean(y_true, y_pred):
    a = y_pred[0::2]
    b = y_pred[1::2]
    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
    y_true = y_true[0::2]
    return ((diff - y_true)**2).mean()

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def siamese_dtw(y_true, y_pred,dist=manhattan_distances):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(y_true)
    assert len(y_pred)
    if ndim(y_true)==1:
        x = y_true.reshape(-1,1)
    if ndim(y_pred)==1:
        y = y_pred.reshape(-1,1)
    r, c = len(y_true), len(y_pred)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(y_true,y_pred,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(y_true)==1:
        path = zeros(len(y_pred)), range(len(y_pred))
    elif len(y_pred) == 1:
        path = range(len(y_true)), zeros(len(y_true))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape)
