import numpy as np
from operator import add

def sum(blocked_rdd):
    return blocked_rdd.map(np.sum).reduce(add)

def mean(blocked_rdd):
    """
    Done this way to avoid overflow from summing everything before dividing. Though
    not sure if that's an issue?
    """
    pavgs = blocked_rdd.map(lambda b: (np.average(b, axis=0), b.shape[0]))
    avgs, weights = zip(*pavgs.collect())
    return np.average(np.array(avgs), axis=0, weights=weights)

def cov(blocked_rdd):
    """
    Calculated the covariance matrix for the given blocked RDD.
    Unlike numpy.cov, expects each row to represent an observation.
    """
    avg = mean(blocked_rdd)
    covs = blocked_rdd.map(lambda x: x - avg).map(lambda x: (x.T.dot(x), x.shape[0]))
    prod, count = covs.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    return prod / count

