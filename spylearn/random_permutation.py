import numpy as np

def random_permutation(rdd, seed=None):
    """
    Shuffles, i.e., randomly reorders, the elements of the given rdd. This is
    more efficient than sorting by a random key.  It functions by assigning
    each element to a random partition and then ordering within each partition
    using the Fisher-Yates algorithm.
    """
    num_partitions = rdd._jrdd.splits().size()
    def partition_partition(split_index, iterable):
        if seed != None:
            np.random.seed(seed + split_index)
        for el in iterable:
            yield (np.random.randint(num_partitions), el)
    
    rdd = rdd.mapPartitionsWithIndex(partition_partition)
    repartitioned = rdd.partitionBy(num_partitions, partitionFunc=lambda x: x)
    
    def fisher_yates(split_index, iterable):
        """
        Order randomly within a partition and strip off keys
        """
        if seed != None:
            np.random.seed(seed + num_partitions + split_index) 

        out = []
        for el in iterable:
            j = np.random.randint(len(out)+1)
            if j == len(out):
                out.append(el[1])
            else:
                out.append(out[j])
                out[j] = el[1]
        return out
    
    return repartitioned.mapPartitionsWithIndex(fisher_yates)

