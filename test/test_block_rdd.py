import shutil
import tempfile
import numpy as np

from common import SpylearnTestCase

from spylearn.block_rdd import block_rdd
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
        data = self.sc.parallelize([np.array([1]) for i in range(0, 100)], 10)
        blocked_data = block_rdd(data)
        assert_array_almost_equal(np.ones((10, 1)), blocked_data.first())

    def test_block_rdd_dict(self):
        pass
