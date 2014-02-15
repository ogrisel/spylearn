def block_data(data, block_size=None):
    """Block an RDD

    :param data: RDD of data points
    :param block_size: Size of blocks in new RDD
    :return data: RDD of blocks
    """

    import pandas as pd
    import numpy as np

    entry = data.first()

    if type(entry) is tuple:
        data = data.mapPartitions(_block_tuple, block_size)

    if type(entry) is np.ndarray:
        data = data.mapPartitions(lambda x: _block_collection(x, np.array, block_size))

    if type(entry) is dict:
        data = data.mapPartitions(lambda x: _block_collection(x, pd.DataFrame, block_size))

    return data


def _block_tuple(iterator, block_size=None):

    import numpy as np

    i = 0
    tuple_size = None
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))

        if block_size is not None and i >= block_size:
            yield tuple(np.array(x) for x in blocked_tuple)
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            i = 0
        for x_j, x in zip(tuple_i, blocked_tuple):
            x.append(x_j)
        i += 1
    yield tuple(np.array(x) for x in blocked_tuple)


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