# Common utility functions
#
import math
import random
# import sets
import sys
from operator import mul

MAX_SAMPLE_SIZE = 200


def bucketize(l, buckets):
    """
        Bucketize a list of two-tuples where the first element of each pair
        is the dimension along which the list should be bucketized and the
        second element of the pair is the associated data object.

        The buckets data structure is a list of (low,high) pairs (inclusive).

        Return a list of pairs where the first item in each pair is the
        range for the bucket (inclusive) and the second item is a list of the
        data objects that fall in the bucket
    """
    results = {}
    for (low, high) in  buckets:
        results[(low, high)] = []
    for (val, el) in l:
        for (low, high) in  buckets:
            if low <= val and val <= high:
                results[(low, high)].append(el)
    items = results.items()
    items.sort()
    return items

def histogram(l, buckets):
    """
        Given a list and a buckets data structure consisiting of
        (low,high) pairs (inclusive), return a list of
        (lower_bound, upper_bound, number_of_entries)
    """
    histogram = []
    for (minRange, maxRange) in buckets:
        nMatches = 0
        for el in l:
            if minRange <= el and el <= maxRange:
                nMatches += 1
        histogram.append((minRange, maxRange, nMatches))
    return histogram


def getBuckets(l, nBuckets, maxSampleSize=MAX_SAMPLE_SIZE):
    """Calculate a maximum entropy partitioning of the given list.
       Return a list of (minVal, maxVal) ranges for the partitioning."""
    l2 = sampleUpTo(l, maxSampleSize)
    l2.sort()
    (partitions, entropy) = __getBuckets(l2, nBuckets, 0, len(l2)-1, {})
    if partitions:
        l3 = list(l)
        l3.sort()
        partitions = refinePartitions(partitions, l3)
    return partitions


def refinePartitions(buckets, l):
    newBuckets = []
    for i in xrange(len(buckets)):
        (low, high) = buckets[i]
        if i == 0:
            low = min(l)
        if i == len(buckets)-1:
            high = max(l)
        else:
            nextLow = buckets[i+1][0]
            high = largestElemLessThan(l, nextLow)
        newBuckets.append((low, high))
    return newBuckets


def largestElemLessThan(l, x):
    assert(len(l) > 0)
    y = l[0]
    for el in l:
        if el >= x:
            break
        y = el
    return y


INVALID_ENTROPY = -100000000.0
def __getBuckets(l, n, minIndex, maxIndex, cache):
    """Try to get the buckets for a
       Return a list of (minVal, maxVal) ranges for the partitioning."""

    # check if we have enough room for all partitions
    if len(uniquify(l)) < n:
        return ([], INVALID_ENTROPY)

    # check if we hit the cache
    k = (minIndex, maxIndex, n)
    if cache.has_key(k):
        return cache[k]

    # check if we only have one partition
    maxPartitions = []
    if n == 1:
        p = (maxIndex-minIndex+1.0) / len(l)
        maxEntropy = -p*math.log(p)
        maxPartitions.append((l[minIndex], l[maxIndex]))
    else:
        (maxPartitions, maxEntropy) = ([], INVALID_ENTROPY)
        for partitionIndex in xrange(minIndex+1, maxIndex+1):
            # make sure we threshold at value changes only
            if l[partitionIndex] == l[partitionIndex-1]:
                continue
            (partitions1, entropy1) = __getBuckets(l, 1, minIndex,
                                                   partitionIndex-1, cache)
            (partitions2, entropy2) = __getBuckets(l, n-1, partitionIndex,
                                                   maxIndex, cache)
            if partitions1 and partitions2: # found a valid partitioning
                #print k, partitionIndex, partitions1, partitions2
                assert(partitions1[-1][1] < partitions2[0][0])
                if entropy1+entropy2 > maxEntropy:
                    maxEntropy = entropy1+entropy2
                    maxPartitions = partitions1 + partitions2


    # sanity check.
    for i in xrange(len(maxPartitions)-1):
        assert(maxPartitions[i][1] < maxPartitions[i+1][0])

    cache[k] = (maxPartitions, maxEntropy)
    return (maxPartitions, maxEntropy)


def warn(message):
    print(message + '\n')


def die(message, exitValue=1):
    print(message + '\n')
    sys.exit(exitValue)


def sampleUpTo(l, n):
    if len(l) <= n:
        return list(l)
    else:
        return random.sample(l, n)


def uniquify(l):
    """ Return a deduped version of a list where all elements are unique.
       This is a stable algorithm - only the first element is retained."""
    s = sets.Set()
    r = []
    for el in l:
        if el not in s:
            s.add(el)
            r.append(el)
    return r


def distinct(l):
    return uniquify(l)

def mean(l):
    #if len(l) == 0:
    #    return 0.0
    #else:
    return float(sum(l))/len(l)

def pearson(X, Y):
    mx = mean(X)
    my = mean(Y)
    sx = math.sqrt(variance(X))
    sy = math.sqrt(variance(Y))

    #    s = 0.0
    #    for (x, y) in zip(X, Y):
    #        s += (x-mx)/sx * (y-my)/sy

    return sum([(x-mx)/sx * (y-my)/sy for x,y in zip(X,Y)]) / (len(X)-1.0)

def variance(l):
    if len(l) < 2:
        return 0.0
    m = mean(l)
    return 1.0 * sum([(x-m)*(x-m) for x in l]) / (len(l) - 1.0)

def removeMissingValues(X,Y):
    newX = []
    newY = []
    for i in range(len(X)):
        if X[i]!='?' and Y[i]!='?':
            newX.append(X[i])
            newY.append(Y[i])
    return newX, newY

def mae(predicted, actual):
    return mean([abs(predicted - actual) for predicted, actual in zip(predicted, actual)])


# Solves Y = aX + b

def linreg(X, Y, verbose=False):
    from math import sqrt
    if len(X) != len(Y):
        raise ValueError #, 'unequal length'

    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
        det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

    meanerror = residual = 0.0
    for x, y in zip(X, Y):
        meanerror = meanerror + (y - Sy/N)**2
        residual = residual + (y - a * x - b)**2
    RR = 1 - residual/meanerror
    ss = residual / (N-2)
    Var_a, Var_b = ss * N / det, ss * Sxx / det

    if verbose:
        print("y=ax+b")
        print("N= %d" % N)
        print("a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N-2, sqrt(Var_a)))
        print("b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N-2, sqrt(Var_b)))
        print("R^2= %g" % RR)
        print("s^2= %g" % ss)

    return a, b


'''
def replaceMissingValues(X):
    mx=mean([val for x in X if x!='?'])
    for i in range(len(X)):
        if X[i]=='?':
            val = mx
        else:
            val = X[i]
        newX.append(val)
    return newX
'''

def binarySearch(l, x):
    """
        Returns a two tuple.
        The first element is a 1 if x was found and a 0 otherwise.
        If x was found, the second number is the largest index for x.
        If x was not found, the second number is the index of the
        largest element < x
    """
    low = 0
    high = len(l) - 1
    mid = (low+high) / 2
    while low <= high:
        if l[mid] > x:
            high = mid - 1
        elif l[mid] < x:
            low = mid + 1
        else:
            while mid < len(l)-2 and l[mid+1] == x:
                mid += 1
            return (1, mid)
        mid = (low+high) / 2

    # the orders are now reversed
    newLow = high
    newHigh = low
    if newHigh <= len(l)-1 and l[newHigh] <= x:
        return (0, newHigh)
    elif newLow >= 0 and l[newLow] <= x:
        return (0, newLow)
    elif newLow-1 >= 0 and l[newLow-1] <= x:
        return (0, newLow-1)
    else:
        return (0, -1)

def length(X):
    return math.sqrt(sum(x*x for x in X))

def normalize(X):
    meanX = mean(X)
    meanAdjusted = [x - meanX for x in X]
    lengthMeanAdjusted = length(meanAdjusted)
    return [x/lengthMeanAdjusted for x in meanAdjusted]

def cosSim(X, Y):
    """
    :param X: the first vector
    :param Y: the second vector
    :return: the cosine similarity between two vectors X and Y
    """
    try:
        # Avoids doing zip which can be expensive if repeated often
        return sum(X[i]*Y[i] for i in range(max(len(X), len(Y))))/float(length(X)*length(Y))
    except ArithmeticError:
        return 0.0


def weightedCosSim(X, Y, w):
    xw = weightedLength(X,w)
    yw = weightedLength(Y,w)
    try:
        listProduct = lambda a,b: map(mul,a,b)
        numerator = sum(listProduct(listProduct(X,Y), w))
        denominator = float(xw*yw)
        return numerator / denominator
    except StandardError:
        return 0.0



def weightedLength(X,w):
    listProduct = lambda a,b: map(mul,a,b)
    return math.sqrt(sum(listProduct(listProduct(X,X), w)))

def weightedCorrelationSim(X, Y, w):
    meanX = weightedMean(X,w)
    meanY = weightedMean(Y,w)
    return weightedCosSim(map(lambda x: x - meanX, X),map(lambda y: y - meanY,Y), w)

def weightedMean(X, w):
    return sum(map(mul,X, w))/float(sum(w))

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [population[_int(_random() * n)] for i in xrange(k)]

def shannonEntropy(pVals, base=2):
    return -1*sum(p * math.log(p, base) for p in pVals if p!= 0)

def logLikelihood(probTrue, actual):
    # For binary classification problems (y=0 or 1)
    probFalse = [1 - p for p in probTrue]
    return sum([math.log(p) for p, y in zip(probTrue, actual) if y==1]) + \
           sum([math.log(p) for p, y in zip(probFalse, actual) if y==0])

def getAccuracy(predicted, actual):
    numCorrect = len([1 for p, a in zip(predicted, actual) if p==a])
    n = len(predicted)
    return float(numCorrect)/n

def getPrecision(predicted, actual):
    numCorrectPositive = len([1 for p, a in zip(predicted, actual) if p==1 and a==1])
    numGuessedPositive = len([1 for p in predicted if p==1])
    return float(numCorrectPositive)/numGuessedPositive

def getRecall(predicted, actual):
    numCorrectPositive = len([1 for p, a in zip(predicted, actual) if p==1 and a==1])
    nPositive = len([1 for a in actual if a==1])
    return float(numCorrectPositive)/nPositive

def posmax(seq, key=lambda x:x):
    """Return the position of the first maximum item of a sequence.
    It accepts the usual key parameter too."""
    #Grabbed from web.  Not thoroughly tested
    return max(enumerate(seq), key=lambda k: key(k[1]))[0]


if __name__ == '__main__':

    probTrue = (.8, .3, .4, .5)
    actual = [1,0,0,1,0]
    assert round(logLikelihood(probTrue, actual), 4) == -1.7838

    predicted = [1,0,0,0,1,1,0,0,1,1]
    actual =    [0,0,1,0,1,1,0,0,0,1]
    assert getAccuracy(predicted,actual ) == .7
    assert getPrecision(predicted,actual ) == .6
    assert getRecall(predicted, actual) == .75

