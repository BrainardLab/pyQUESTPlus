import numpy as np

def ArrayEntropy(probArray):
    """
    Compute the entropy of the probability values in the passed array,
    with respect to base 2 (i.e. entropy in bits).

    Each column is handled separately.

    Parameters:
    probArray : np.array
        An array of probabilities for the possible outcomes.
        These should sum to 1.

    Returns:
    np.array
        The computed entropy of the array.
    """
    # Check that probabilities sum to something close to 1
    tolerance = 1e-7
    assert abs(np.sum(probArray) - 1) < tolerance, "Probabilities do not sum to 1"

    # Compute the log probs
    logProbs = np.log2(probArray)

    # Compute the entropy
    # Using np.nansum skips adding in any terms where the
    # probability is zero, where np.log2(0) returns NaN.
    arrayEntropy = -np.nansum(probArray * logProbs, axis=0)

    return arrayEntropy

def DrawFromDomainList(domainList):
    """
    Draw parameters from list of domain for each parameter

    Parameters:
    domainList : list
        List where each entry is the domain for the corresponding parameter

    Returns:
    v : list
        The random draw in row vector form
    """
    if not domainList:
        raise ValueError("domainList must not be empty")
    if not isinstance(domainList, list):
        raise TypeError("domainList must be a list")
    vlb, vub = GetBoundsFromDomainList(domainList)
    v = [np.random.uniform(vlb[i], vub[i]) for i in range(len(domainList))]
    return v

def GetBoundsFromDomainList(domainList):
    """
    Get parameter bounds from list of domain for each parameter

    Parameters:
    domainList : list
        List where each entry is the domain for the corresponding parameter

    Returns:
    vlb : list
        Lower bound in row vector form
    vub : list
        Upper bound in row vector form
    """
    if not domainList:
        raise ValueError("domainList must not be empty")
    if not isinstance(domainList, list):
        raise TypeError("domainList must be a list")
    
    vlb = [min(domain) for domain in domainList]
    vub = [max(domain) for domain in domainList]
    return vlb, vub

def LogLikelihood(stimCounts, qpPF, psiParams, check=False):
    """
    Compute log likelihood of a stimulus count array

    Args:
        stimCounts: A list of dictionaries with each stimulus value presented
                    in sorted order, and a vector of the counts of each possible
                    outcome type that happened on trials for that stimulus value:
                    stimCounts[i]['stim'] - List of stimulus parameters
                    stimCounts[i]['outcomeCounts'] - List of length
                        nOutcomes with the number of times each outcome
                        happened for the given stimulus.
        qpPF: Callable function (e.g. PFWeibull).
        psiParams: List of parameters for the passed psychometric function.
        check: boolean (False) - Run some checks on the data unpacking. Slows things down.

    Returns:
        logLikelihood: Log likelihood of the data.  If the psychometric function
                    returns NaN for any of its inputs, the logLikelihood is returned
                    as NaN.
    """
    # Check input
    if not isinstance(stimCounts, list):
        raise TypeError("stimCounts must be a list")
    if not callable(qpPF):
        raise TypeError("qpPF must be a callable function")
    if not isinstance(psiParams, list):
        raise TypeError("psiParams must be a list")
    if not isinstance(check, bool):
        raise TypeError("check must be a boolean")

    # Get number of stimuli, stimulus parameter dimension and number of outcomes
    nStim = len(stimCounts)
    stimDim = len(stimCounts[0]["stim"])
    nOutcomes = len(stimCounts[0]["outcomeCounts"])

    # Get stimulus matrix with parameters along each column.
    stimMat = np.array([stim["stim"] for stim in stimCounts])

    # Get predicted proportions for each stimulus
    predictedProportions = qpPF(stimMat, psiParams)

    # Get the outcomes
    outcomeCounts = np.array([stim["outcomeCounts"] for stim in stimCounts])

    # Check
    if check:
        outcomeCounts1 = np.zeros((nStim, nOutcomes))
        for i in range(nStim):
            outcomeCounts1[i, :] = stimCounts[i]["outcomeCounts"]
        if np.any(outcomeCounts != outcomeCounts1):
            raise ValueError("Two ways of unpacking outcome counts do not match.")

    # Compute the log likelihood
    if np.any(np.isnan(predictedProportions)):
        logLikelihood = np.nan
    else:
        nLogP = NLogP(outcomeCounts, predictedProportions)
        logLikelihood = np.sum(nLogP)

    return logLikelihood

def NLogP(n, p):
    """
    Compute n*log(p) and handle cases where one or both args are zero

    Args:
        n: Number of trials. Can be a number, vector or matrix.
        p: Probability. Needs to be the same size as n.

    Both n and p must contain non-negative values.

    Returns:
        nLogP: n*log(p). Same size as n and p. For each entry:
            Returns -1*real max if p == 0 && n > 0.
            Returns 0 if p == 0 && n == 0.
    """
    # Check input
    if not isinstance(n, (int, float, np.ndarray)):
        raise TypeError("n must be a number, vector or matrix")
    if not isinstance(p, (int, float, np.ndarray)):
        raise TypeError("p must be a number, vector or matrix")
    if np.any(n < 0):
        raise ValueError("Each passed n must be non-negative")
    if np.any(p < 0):
        raise ValueError("Each passed p must be non-negative")
    if np.shape(n) != np.shape(p):
        raise ValueError("Passed n and p must have the same size")

    # Enforce type to ensure proper elementwise operations
    n = np.array(n)
    p = np.array(p)

    if n.shape != p.shape:
        raise ValueError("n and p must have the same shape")

    # Compute nLogP
    epsilon = 1e-10
    nLogP = n * np.log(p + epsilon)
    nLogP[(p == 0) & (n > 0)] = -1 * np.finfo(float).max
    nLogP[(p == 0) & (n == 0)] = 0

    return nLogP


def StimIndexToStim(stimIndex, stimDomain):
    """
    Find stimulus in stimDomain corresponding to stimIndex

    Parameters:
    stimIndex : int
        Row index into stimDomain where stimulus lives.
    stimDomain : np.array
        Matrix where each row describes one of the possible
        stimuli that quest is dealing with.

    Returns:
    np.array
        Row vector of stimulus parameters.
    """
    # Python uses 0-based indexing, so we subtract 1 from stimIndex
    stim = stimDomain[stimIndex, :]
    return stim
