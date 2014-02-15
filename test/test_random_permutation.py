from common import SpylearnTestCase
import shutil
import tempfile

from spylearn.random_permutation import random_permutation
import numpy as np

from nose.tools import assert_equals, assert_not_equals
from numpy.testing import assert_array_almost_equal


class TestUtils(SpylearnTestCase):
    def setUp(self):
        super(TestUtils, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestUtils, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRandomPermutation(TestUtils):

    def test_random_permutation(self):
        data = self.sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        shuffled = random_permutation(data, seed=5).collect()
        assert_equals([10, 7, 6, 8, 2, 5, 9, 4, 3, 1], shuffled)
        assert_not_equals(shuffled, random_permutation(data, seed=6).collect())

