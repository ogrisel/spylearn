from pyspark.mllib.clustering import KMeans
from pyspark.rdd import RDD
from operator import add
from numpy.linalg import norm
import itertools

class ParallelKMeans:
    """K-Means clustering

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default: 10
        Maximum number of iterations of the k-means algorithm for a
        single run.
    
    init : string, optional, default: k-means||
        Method for coming up with the initial clusters centers.  Either 'k-means||'
        for the algorithm described by Bahmani et al. (Bahmani et al.,
        Scalable K-Means++, VLDB 2012) or 'random' for initial centers chosen from
        random input points.

    """
    def __init__(self, n_clusters=8, max_iter=10, init='k-means||'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def fit(self, rdd):
        rdd.cache()
        self.model = KMeans.train(rdd, self.n_clusters, self.max_iter,
            runs=1, initializationMode=self.init)
        self.cluster_centers_ = self.model.centers
        self.inertia_ = self.score_rdd(rdd)

    def error(self, point):
        center = self.cluster_centers_[self.model.predict(point)]
        return norm(point - center)
    
    def predict(self, data):
        if isinstance(data, RDD):
            return self.predict_rdd(data)
        else:
            return self.predict_array(data)

    def predict_rdd(self, rdd):
        return rdd.map(lambda x: self.model.predict(x))

    def predict_array(self, arr):
        return [self.model.predict(x) for x in arr]

    def score(self, data):
        if isinstance(data, RDD):
            return self.score_rdd(data)
        else:
            return self.score_array(data)

    def score_rdd(self, rdd):
        return -rdd.map(self.error).sum()

    def score_array(self, arr):
        return -sum(itertools.imap(self.error, arr))


