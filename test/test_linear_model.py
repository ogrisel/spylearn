import numpy as np

from sklearn.linear_model import SGDClassifier

from spylearn.linear_model import parallel_train
from spylearn.block_rdd import block_rdd

from common import SpylearnTestCase

from nose.tools import assert_greater
from nose import SkipTest
from numpy.testing import assert_array_almost_equal


class SumModel(object):

    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def partial_fit(self, X, y, **kwargs):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.coef_ += X.sum(axis=0)
        return self


class LinearModelTestCase(SpylearnTestCase):

    data = None
    n_partitions = 2

    def setUp(self):
        super(LinearModelTestCase, self).setUp()
        if self.data is None:
            rng = np.random.RandomState(42)
            X = rng.normal(size=(int(1e3), 50))
            coef = rng.normal(size=50)
            y = (np.dot(X, coef) > 0.01).astype(np.int)
            self.X = X
            self.y = y
            self.classes = np.unique(y)
            self.data = self.sc.parallelize(list(zip(X, y)),
                numSlices=self.n_partitions).cache()
            self.blocked_data = block_rdd(self.data, block_size=171)

    def test_parallel_train_sum_model_non_blocked(self):
        n_iter = 2
        model = parallel_train(SumModel(), self.data, self.classes, n_iter)
        expected_coef = self.X.sum(axis=0) * n_iter / self.n_partitions
        assert_array_almost_equal(model.coef_, expected_coef , 5)

    def test_parallel_train(self):
        if not hasattr(SGDClassifier, 'partial_fit'):
            raise SkipTest('sklearn >= 0.13 is required to run this test')
        model = SGDClassifier(loss='log', alpha=1e-5, random_state=2)
        model = parallel_train(model, self.blocked_data, self.classes)
        assert_greater(model.score(self.X, self.y), 0.90)
