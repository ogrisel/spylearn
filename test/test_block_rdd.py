import shutil
import tempfile
import numpy as np

from common import SpylearnTestCase

from spylearn.block_rdd import block_rdd

from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal


class TestUtils(SpylearnTestCase):
    def setUp(self):
        super(TestUtils, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestUtils, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestBlockRDD(TestUtils):

    def test_block_rdd_tuple(self):
        pass

    def test_block_rdd_array(self):
        n_partitions = 10
        n_samples = 100
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
            n_partitions)
        blocked_data = block_rdd(data)
        assert_array_almost_equal(np.ones((10, 1)), blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_array_almost_equal(np.ones((10, 1)), blocks[-1])
        assert_equal(sum(len(b) for b in blocks),  n_samples)

        n_partitions = 17
        data = self.sc.parallelize([np.array([1]) for i in range(n_samples)],
            n_partitions)
        blocked_data = block_rdd(data)
        assert_array_almost_equal(np.ones((n_samples / n_partitions, 1)),
                                  blocked_data.first())
        blocks = blocked_data.collect()
        assert_equal(len(blocks), n_partitions)
        assert_equal(sum(len(b) for b in blocks),  n_samples)

    def test_block_rdd_dict(self):
        pass
