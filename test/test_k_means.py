import numpy as np

from spylearn.k_means import ParallelKMeans

from common import SpylearnTestCase

from nose.tools import assert_greater, assert_less
from numpy.testing import assert_array_equal

class KMeansTestCase(SpylearnTestCase):

    data = None
    n_partitions = 2

    def setUp(self):
        super(KMeansTestCase, self).setUp()
        if self.data is None:
            rng = np.random.RandomState(42)
            self.center1 = rng.randint(10, size=50)
            self.center2 = rng.randint(10, size=50)
            self.cluster1 = rng.normal(size=(int(1e3), 50)) + self.center1
            self.cluster2 = rng.normal(size=(int(1e3), 50)) + self.center2
            X = np.concatenate([self.cluster1, self.cluster2])
            rng.shuffle(X)
            self.data = self.sc.parallelize(X, numSlices=self.n_partitions)
            self.expected_error = sum([np.linalg.norm(rng.randn(50) -
                rng.randn(50)) for _ in range(int(1e3))])

    def test_clustering(self):
        model = ParallelKMeans(2, 7)
        model.fit(self.data)
        cluster1_predictions = model.predict(self.cluster1)
        cluster2_predictions = model.predict(self.cluster2)
        assert_array_equal(np.repeat(cluster1_predictions[0], len(self.cluster1)), 
            cluster1_predictions)
        assert_array_equal(np.repeat(cluster2_predictions[0], len(self.cluster2)), 
            cluster2_predictions)

        score1 = model.score(self.cluster1)
        assert_less(score1, 0)
        assert_greater(score1, -self.expected_error * 1.5)
        score2 = model.score(self.cluster2)
        assert_less(score2, 0)
        assert_greater(score2, -self.expected_error * 1.5)

