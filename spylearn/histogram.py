import numpy as np
import collections
import types

def histogram(rdd, range=None, bins=10):
  """
  Compute the histogram of an RDD.
  """
  def _bin(num, bin_edges):
    """
    Given a number and set of bins defined by edges, computes which bin the number
    lies in.  Lower edges are inclusive, higher are exclusive.
    """
    if num < bin_edges[0]:
      return []

    for i, edge in enumerate(bin_edges[1:]):
      if num < edge:
        return [i]

    return []

  if isinstance(bins, collections.Iterable):
    bin_edges = bins
  elif type(bins) is types.IntType:
    if range is None:
      raise TypeError("range argument required when bins is an int")
    bin_edges = np.linspace(range[0], range[1], bins+1)
  else:
    raise TypeError("bins required to be an int or iterable")

  return (rdd.flatMap(lambda x: _bin(x, bin_edges)).countByValue(), bin_edges)

~                                                                                 
