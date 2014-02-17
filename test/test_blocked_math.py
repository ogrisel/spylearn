from common import SpylearnTestCase

from spylearn.blocked_math import cov
from spylearn.block_rdd import block_rdd
import numpy as np

from numpy.testing import assert_array_almost_equal

class TestUtils(SpylearnTestCase):
    def setUp(self):
        super(TestUtils, self).setUp()

    def tearDown(self):
        super(TestUtils, self).tearDown()

class TestBlockedMath(TestUtils):

    def test_cov(self):
        rng = np.random.RandomState(42)
        true_cov = np.array([[3., 2., 4.], [2., 2., 5.], [4., 5., 6.]])
        mat = rng.multivariate_normal(np.array([1., 2., 3.]), size=int(1e3),
            cov=true_cov)
        data = block_rdd(self.sc.parallelize(mat, 4))
        rdd_cov = cov(data)
        assert_array_almost_equal(np.cov(mat.T), rdd_cov, decimal=1)

