import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from spylearn.linear_model import parallel_train

from common import SpylearnTestCase

from nose.tools import assert_greater
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
            X, y = make_classification(n_samples=int(1e3), n_features=50,
                                       n_informative=30, random_state=2)
            self.X = X
            self.y = y
            self.classes = np.unique(y)
            self.data = self.sc.parallelize(list(zip(X, y)),
                numSlices=self.n_partitions).cache()

    def test_parallel_train_sum_model_non_blocked(self):
        n_iter = 2
        model = parallel_train(SumModel(), self.data, self.classes, n_iter)
        expected_coef = self.X.sum(axis=0) * n_iter / self.n_partitions
        assert_array_almost_equal(model.coef_, expected_coef , 5)
