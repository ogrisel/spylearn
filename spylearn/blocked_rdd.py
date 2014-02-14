import numpy as np
import pandas as pd


def _block_tuple(iterator, block_size=None):
    i = 0
    tuple_size = None
    blocked_tuple = None
    for tuple_i in iterator:
        if blocked_tuple is None:
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))

        if block_size is not None and i > block_size:
            yield tuple(np.array(x) for x in blocked_tuple)
            blocked_tuple = tuple([] for _ in range(len(tuple_i)))
            i = 0
        for x_j, x in zip(tuple_i, blocked_tuple):
            x.append(x_j)
        i += 1
    yield tuple(np.array(x) for x in blocked_tuple)


def _block_array(iterator, block_size=None):
    i = 0
    arrays = []
    for a in iterator:
        if block_size is not None and i > block_size:
            yield np.array(arrays)
            arrays = []
        arrays.append(a)
        i += 1
    yield np.array(arrays)


def _block_dataframe(iterator, block_size=None):
    i = 0
    rows = []
    for d in iterator:
        if block_size is not None and i > block_size:
            yield pd.DataFrame(rows)
            rows = []
        rows.append(d)
        i += 1
    yield pd.DataFrame(rows)
