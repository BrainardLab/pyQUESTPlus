import pytest
import numpy as np
from pyQUESTPlus.utilities import ArrayEntropy, DrawFromDomainList, GetBoundsFromDomainList, LogLikelihood, NLogP, StimIndexToStim
class TestArrayEntropy:
    def test_ArrayEntropy_valid_input(self):
        probArray = np.array([0.1, 0.2, 0.7]).T
        result = ArrayEntropy(probArray)
        expected_result = 1.1567796494470395  # This is the expected entropy for the given probArray
        assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    def test_ArrayEntropy_invalid_input(self):
        probArray = np.array([0.1, 0.2, 0.8]).T  # This array does not sum to 1
        with pytest.raises(AssertionError):
            ArrayEntropy(probArray)      
class TestDrawFromDomainList:
    def test_valid_input(self):
        domain_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        v = DrawFromDomainList(domain_list)
        assert len(v) == len(domain_list), f"Expected length {len(domain_list)}, but got {len(v)}"
        for i in range(len(v)):
            assert domain_list[i][0] <= v[i] <= domain_list[i][-1], f"Expected value between {domain_list[i][0]} and {domain_list[i][-1]}, but got {v[i]}"
    def test_empty_input(self):
        domain_list = []
        with pytest.raises(ValueError):
            DrawFromDomainList(domain_list)
    def test_invalid_input(self):
        domain_list = "invalid input"
        with pytest.raises(TypeError):
            DrawFromDomainList(domain_list)

class TestGetBoundsFromDomainList:
    def test_valid_input(self):
        domain_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        vlb, vub = GetBoundsFromDomainList(domain_list)
        assert vlb == [1, 4, 7], f"Expected [1, 4, 7], but got {vlb}"
        assert vub == [3, 6, 9], f"Expected [3, 6, 9], but got {vub}"

    def test_empty_input(self):
        domain_list = []
        with pytest.raises(ValueError):
            GetBoundsFromDomainList(domain_list)

    def test_invalid_input(self):
        domain_list = "invalid input"
        with pytest.raises(TypeError):
            GetBoundsFromDomainList(domain_list)
class TestGetBoundsFromDomainList:
    def test_GetBoundsFromDomainList_valid_input(self):
        domain_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        vlb, vub = GetBoundsFromDomainList(domain_list)
        assert vlb == [1, 4, 7], f"Expected [1, 4, 7], but got {vlb}"
        assert vub == [3, 6, 9], f"Expected [3, 6, 9], but got {vub}"
    
    def test_GetBoundsFromDomainList_empty_input(self):
        domain_list = []
        with pytest.raises(Exception):
            GetBoundsFromDomainList(domain_list)
    
    def test_GetBoundsFromDomainList_invalid_input(self):
        domain_list = "invalid input"
        with pytest.raises(TypeError):
            GetBoundsFromDomainList(domain_list)
class TestLogLikelihood:
    def test_invalid_stimCounts(self):
        with pytest.raises(TypeError):
            LogLikelihood('invalid', lambda x: x, [1, 2, 3])

    def test_invalid_PF(self):
        with pytest.raises(TypeError):
            LogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], 'invalid', [1, 2, 3])

    def test_invalid_psiParams(self):
        with pytest.raises(TypeError):
            LogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, 'invalid')

    def test_invalid_check(self):
        with pytest.raises(TypeError):
            LogLikelihood([{'stim': [1, 2, 3], 'outcomeCounts': [4, 5, 6]}], lambda x: x, [1, 2, 3], 'invalid')

    def test_valid_input(self):
        stimCounts = [{'stim': [1, 2, 3], 'outcomeCounts': [0, 2, 2]}]
        psiParams = [1]
        result = LogLikelihood(stimCounts, PFTest, psiParams)
        assert isinstance(result, float)  # Assuming the function returns a float    
class TestNLogP:
    def test_basic(self):
        n = np.array([1, 2, 3])
        p = np.array([0.1, 0.2, 0.3])
        expected = np.array([-2.3026,   -3.2189,  -3.6119])
        np.testing.assert_almost_equal(NLogP(n, p), expected, decimal=4)

    def test_zero_n(self):
        n = np.array([0, 0, 0])
        p = np.array([0.1, 0.2, 0.3])
        expected = np.array([0, 0, 0])
        np.testing.assert_almost_equal(NLogP(n, p), expected, decimal=5)

    def test_zero_p(self):
        n = np.array([1, 2, 3])
        p = np.array([0, 0, 0])  

def TestStimIndexToStim():
    # Test with a 2D array
    stimDomain = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stimIndex = 1
    expected_result = np.array([4, 5, 6])
    assert np.array_equal(StimIndexToStim(stimIndex, stimDomain), expected_result)

    # Test with a 1D array
    stimDomain = np.array([[1], [2], [3]])
    stimIndex = 1
    expected_result = np.array([2])
    assert np.array_equal(StimIndexToStim(stimIndex, stimDomain), expected_result)

    # Test with an out-of-bounds index
    stimDomain = np.array([1, 2, 3])
    stimIndex = 4
    with pytest.raises(IndexError):
        StimIndexToStim(stimIndex, stimDomain)

# This has the right input/output format for a PF function. Enough for an arg test.
def PFTest(stim,x):
    result = np.full(stim.shape,0.1) #np.array([0.1 for _ in stim])
    return result
    