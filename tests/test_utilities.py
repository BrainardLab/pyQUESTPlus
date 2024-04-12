import pytest
import numpy as np
from pyQUESTPlus.utilities import qpLogLikelihood, qpNLogP

class TestQpLogLikelihood:
    def test_invalid_stimCounts(self):
        with pytest.raises(TypeError):
            qpLogLikelihood('invalid', lambda x: x, [1, 2, 3])

    def test_invalid_qpPF(self):
        with pytest.raises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], 'invalid', [1, 2, 3])

    def test_invalid_psiParams(self):
        with pytest.raises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, 'invalid')

    def test_invalid_check(self):
        with pytest.raises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, [1, 2, 3], 'invalid')

    def test_valid_input(self):
        stimCounts = [{'stim': [1, 2, 3], 'outcomeCounts': [0, 2, 2]}]
        # qpPF = lambda x: x(0)  # Fix: Correctly define the input parameter and return the desired value
        psiParams = [1]
        result = qpLogLikelihood(stimCounts, qpPFTest, psiParams)
        assert isinstance(result, float)  # Assuming the function returns a float
        
class TestQpNLogP:
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

# This has the right input/output format for a QP function. Enough for an arg test.
def qpPFTest(stim,x):
    result = np.full(stim.shape,0.1) #np.array([0.1 for _ in stim])
    #print(result.shape)
    #breakpoint()
    return result
    