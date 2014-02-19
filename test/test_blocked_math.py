from common import SpylearnTestCase

from spylearn.blocked_math import count, cov, svd, svd_em
from spylearn.block_rdd import block_rdd

import numpy as np
import scipy.linalg as ln
from numpy.testing import assert_array_almost_equal


def match_sign(a, b):
    a_sign = np.sign(a)
    b_sign = np.sign(b)
    if np.array_equal(a_sign, -b_sign):
        return -b
    elif np.array_equal(a_sign, b_sign):
        return b
    else:
        raise AssertionError("inconsistent matching of sign")


class TestUtils(SpylearnTestCase):
    def setUp(self):
        super(TestUtils, self).setUp()

    def tearDown(self):
        super(TestUtils, self).tearDown()


class TestBlockedMath(TestUtils):

    def test_count(self):
        n_samples = 100
        n_partitions = 10
        mat = [np.array([1]) for i in range(n_samples)]
        data = block_rdd(self.sc.parallelize(mat, n_partitions))
        assert_array_almost_equal(n_samples, count(data))

    def test_cov(self):
        rng = np.random.RandomState(42)
        true_cov = np.array([[3., 2., 4.], [2., 2., 5.], [4., 5., 6.]])
        mat = rng.multivariate_normal(np.array([1., 2., 3.]), size=int(1e3),
            cov=true_cov)
        data = block_rdd(self.sc.parallelize(mat, 4))
        rdd_cov = cov(data)
        assert_array_almost_equal(np.cov(mat.T), rdd_cov, decimal=1)

    def test_svd(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(1e3, 10)
        data = block_rdd(self.sc.parallelize(list(mat), 10))
        u, s, v = svd(data, 1)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(mat)
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]))
        assert_array_almost_equal(s[0], s_true[0])
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]))

    def test_svd_em(self):
        rng = np.random.RandomState(42)
        mat = rng.randn(10, 3)
        data = block_rdd(self.sc.parallelize(list(mat), 2)).cache()
        u, s, v = svd_em(data, 1, seed=42)
        u = np.squeeze(np.concatenate(np.array(u.collect()))).T
        u_true, s_true, v_true = ln.svd(mat)
        tol = 1
        assert_array_almost_equal(v[0], match_sign(v[0], v_true[0, :]), tol)
        assert_array_almost_equal(s[0], s_true[0], tol)
        assert_array_almost_equal(u, match_sign(u, u_true[:, 0]), tol)

