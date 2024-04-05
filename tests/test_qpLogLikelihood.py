import pytest
import numpy as np
from pyQUESTPlus.qpLogLikelihood import qpLogLikelihood

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
        stimCounts = [{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}]
        qpPF = lambda x: x(0)  # Fix: Correctly define the input parameter and return the desired value
        psiParams = [1]
        result = qpLogLikelihood(stimCounts, qpPFTest, psiParams)
        assert isinstance(result, float)  # Assuming the function returns a float

# This has the right input/output format for a QP function. Enough for an arg test.
def qpPFTest(stim,x):
    
    return np.array([0.1 for _ in stim])
    