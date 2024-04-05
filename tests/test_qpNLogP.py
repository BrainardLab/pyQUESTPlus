import unittest
import numpy as np
from qpLogLikelihood import qpNLogP

class TestQpNLogP(unittest.TestCase):
    def test_basic(self):
        n = np.array([1, 2, 3])
        p = np.array([0.1, 0.2, 0.3])
        expected = np.array([-2.3026,   -3.2189,  -3.6119])
        np.testing.assert_almost_equal(qpNLogP(n, p), expected, decimal=4)

    def test_zero_n(self):
        n = np.array([0, 0, 0])
        p = np.array([0.1, 0.2, 0.3])
        expected = np.array([0, 0, 0])
        np.testing.assert_almost_equal(qpNLogP(n, p), expected, decimal=5)

    def test_zero_p(self):
        n = np.array([1, 2, 3])
        p = np.array([0, 0, 0])
        expected = np.array([-np.finfo(float).max, -np.finfo(float).max, -np.finfo(float).max])
        np.testing.assert_almost_equal(qpNLogP(n, p), expected, decimal=5)

    def test_zero_n_p(self):
        n = np.array([0, 0, 0])
        p = np.array([0, 0, 0])
        expected = np.array([0, 0, 0])
        np.testing.assert_almost_equal(qpNLogP(n, p), expected, decimal=5)

if __name__ == '__main__':
    unittest.main()