from common import SpylearnTestCase
import shutil
import tempfile

from spylearn.histogram import histogram
import numpy as np

from nose.tools import assert_equals
from numpy.testing import assert_array_almost_equal


class TestUtils(SpylearnTestCase):
    def setUp(self):
        super(TestUtils, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestUtils, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestHistogram(TestUtils):

    def test_bins_as_number(self):
        data = self.sc.parallelize([1, 2, 3, 4, 5])
        hist, bin_edges = histogram(data, range=(0, 6), bins=2)
        assert_equals(2, hist[0])
        assert_equals(3, hist[1])
        assert_array_almost_equal(np.array([0, 3, 6]), bin_edges)

    def test_bins_as_array(self):
        data = self.sc.parallelize([1, 2, 3, 4, 5])
        hist, bin_edges = histogram(data, bins=[0, 3, 6])
        assert_equals(2, hist[0])
        assert_equals(3, hist[1])
        assert_array_almost_equal(np.array([0, 3, 6]), bin_edges)

    def test_ignore_out_of_range(self):
        data = self.sc.parallelize([1, 2, 3, 4, 5])
        hist, bin_edges = histogram(data, range=(2, 5), bins=2)
        assert_equals(2, hist[0])
        assert_equals(1, hist[1])
        assert_array_almost_equal(np.array([2, 3.5, 5]), bin_edges)

