import numpy as np
import scipy.sparse as sp


def block_rdd(data, block_size=None):
    """Block an RDD

    :param data: RDD of data points
    :param block_size: Size of blocks in new RDD
    :return data: RDD of blocks
    """

    import pandas as pd
    try:
        entry = data.first()
    except IndexError:
        # empty RDD: do not block
        return data

    # do different kinds of block depending on the type
    if isinstance(entry, tuple):
        return data.mapPartitions(_block_tuple, block_size)

    elif isinstance(entry, dict):
        return data.mapPartitions(
            lambda x: _block_collection(x, pd.DataFrame, block_size))

    elif sp.issparse(entry):
        return data.mapPartitions(
            lambda x: _block_collection(x, sp.vstack, block_size))

    else:
        # Fallback to array packing
        return data.mapPartitions(
            lambda x: _block_collection(x, np.array, block_size))


def _pack_accumulated(accumulated):
    if len(accumulated) > 0 and sp.issparse(accumulated[0]):
        return sp.vstack(accumulated)
    else:
        return np.array(accumulated)


def _block_tuple(iterator, block_size=None):
    """Pack rdd of tuples as tuples of arrays or scipy.sparse matrices"""
    i = 0
    tuple_size = None
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))

        if block_size is not None and i >= block_size:
            yield tuple(_pack_accumulated(x) for x in blocked_tuple)
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            i = 0
        for x_j, x in zip(tuple_i, blocked_tuple):
            x.append(x_j)
        i += 1
    yield tuple(_pack_accumulated(x) for x in blocked_tuple)


def _block_collection(iterator, collection_type, block_size=None):

    i = 0
    accumulated = []
    for a in iterator:
        if block_size is not None and i >= block_size:
            yield collection_type(accumulated)
            accumulated = []
            i = 0
        accumulated.append(a)
        i += 1
    yield collection_type(accumulated)