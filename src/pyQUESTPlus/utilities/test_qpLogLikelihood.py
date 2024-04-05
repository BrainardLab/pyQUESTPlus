import unittest
import numpy as np
from qpLogLikelihood import qpLogLikelihood

class TestQpLogLikelihood(unittest.TestCase):
    def test_invalid_stimCounts(self):
        with self.assertRaises(TypeError):
            qpLogLikelihood('invalid', lambda x: x, [1, 2, 3])

    def test_invalid_qpPF(self):
        with self.assertRaises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], 'invalid', [1, 2, 3])

    def test_invalid_psiParams(self):
        with self.assertRaises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, 'invalid')

    def test_invalid_check(self):
        with self.assertRaises(TypeError):
            qpLogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, [1, 2, 3], 'invalid')

    def test_valid_input(self):
        stimCounts = [{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}]
        qpPF = lambda x: x(0)  # Fix: Correctly define the input parameter and return the desired value
        psiParams = [1]
        result = qpLogLikelihood(stimCounts, qpPFTest, psiParams)
        self.assertIsInstance(result, float)  # Assuming the function returns a float

# This has the right input/output format for a QP function. Enough for an arg test.
def qpPFTest(stim,x):
    
    return np.array([0.1 for _ in stim])
    
if __name__ == '__main__':
    unittest.main()