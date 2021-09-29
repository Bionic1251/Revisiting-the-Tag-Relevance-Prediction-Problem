import math
#from corpus import imdbcorpus
#from src.data.imdb import IMDBCorpus


# Transformation options
from src.common import *
from src.data import utils


def read(fileName='imdb'):
    """
    :return: the un-pickled object
    """
    full_path = os.path.join(DIR_IMDB_SCRAPE, fileName + '.pickle')
    return cPickle.load(open(full_path, 'rb'))


class FeatureOption():

    def __init__(self):
        pass

    def getTypeName(self):
        pass

    def getName(self):
        pass
    
    def getDesc(self):
        return '%s: %s'%(self.getTypeName(), self.getName())


class Transform(FeatureOption):
    def getTypeName(self):
        return 'Transform'
    def transformVector(self, l, tags, movies, corpus):
        return [self.transformValue(x, tag, corpus) for x, tag, movieId in zip(l, tags, movieId)]
    def transformValue(self, x):
        pass
class IdentityTransform(Transform):
    def getName(self):
        return 'none'
    def transformValue(self, x, tag, movieId, corpus):
        return x
class LogTransform(Transform):
    def __init__(self, c):
        self.c = c
    def getName(self):
        return 'log ' + str(self.c)
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            return math.log(x + self.c)
class LogTransformRelative(Transform):
    def __init__(self, c):
        self.c = c
    def getName(self):
        return 'log_relative_' + str(self.c)
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            return math.log(x + self.c * corpus.getBackgroundWordFreq(tag))
class LogTransformBeta(Transform):
    def getName(self):
        return 'log_beta'
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            if x==0:
                return math.log(corpus.getBetaSmoothedZeroFreq(tag, movieId))
            else:
                return math.log(x)
class LogTransformRelativeBeta(Transform):
    def getName(self):
        return 'log_relative_beta'
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            if x==0:
                return math.log(corpus.getBetaSmoothedZeroFreq(tag, movieId) + corpus.getBackgroundWordFreq(tag))
            else:
                return math.log(x + corpus.getBackgroundWordFreq(tag))

class LogTransformRelativeBeta2(Transform):
    def getName(self):
        return 'log_relative_beta2'
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            if x==0:
                backgroundFreq = corpus.getBackgroundWordFreq(tag)
                if backgroundFreq == 0:
                    return 0
                return math.log(corpus.getBetaSmoothedZeroFreq(tag, movieId) + backgroundFreq) - math.log(corpus.getBackgroundWordFreq(tag))
            else:
                return math.log(x + corpus.getBackgroundWordFreq(tag)) - math.log(corpus.getBackgroundWordFreq(tag))            

class LogTransformRelativeBeta3(Transform):
    def getName(self):
        return 'log_relative_beta3_' + str(self.c)
    def __init__(self, corpus, tags, c=1):
        self.c = c
        self.avgVal = {}
        self.stdDev = {}
        movies = corpus.docVectors.keys()
        for tag in tags:
            vals = [self.transformValue(corpus.getWordFreq(movieId, tag), tag, movieId, corpus, zScore=False) for movieId in movies]
            self.avgVal[tag] = utils.mean(vals)
            self.stdDev[tag] = math.sqrt(utils.variance(vals))

    def transformValue(self, x, tag, movieId, corpus, zScore=True):
        if (x==None):
            return None
        avgTagFreq = corpus.getTagFreqMeanStdDev(tag)[0]
        if avgTagFreq==0:  # Does not appear at all in the corpus
            return 0
        if x==0:
            x = corpus.getBetaSmoothedZeroFreq(tag, movieId)
        
        val = math.log(x + self.c * avgTagFreq)
        if zScore:
            return (val - self.avgVal[tag]) / self.stdDev[tag]   
        else:
            return val
class LogTransformMax(Transform):
    def getName(self):
        return 'log max '
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            return math.log(max(x, corpus.getBackgroundWordFreq(tag)))
class SquareRootTransform(Transform):
    def getName(self):
        return 'square root'
    def transformValue(self, x, tag, movieId, corpus):
        if (x==None):
            return None
        else:
            return math.sqrt(x)
        
    

# Assorted common methods

def normalizeCorpus(corpus, smoother, tfidf):
    freq, betaSmooth, smoothingWords=smoother.getSmoothingParams()
    tfidf = tfidf.getTfidf()
    corpus.smoothingWords = smoothingWords
    corpus.normalize(freq=freq, betaSmooth=betaSmooth, tfidf=tfidf)
    
def getCorpus(filename, smoother, tfidf, termFilter):
    imdbCorpus = read(fileName = filename)
    termFilter.filterCorpus(imdbCorpus)
    normalizeCorpus(imdbCorpus, smoother, tfidf)
    return imdbCorpus

# Smoothing options

class Smoother(FeatureOption):
    def getTypeName(self):
        return 'Smoothing'
    def getSmoothingParams(self, corpus):
        pass
class BetaSmoother(Smoother):
    def getName(self):
        return 'beta'
    def getSmoothingParams(self):
        freq=False
        betaSmooth=True
        smoothingWords = 0
        return (freq, betaSmooth, smoothingWords)
class ConstantSmoother(Smoother):
    def __init__(self, c):
        self.c = c
    def getName(self):
        return 'constant ' + str(self.c)
    def getSmoothingParams(self):
        freq=True
        betaSmooth=False
        smoothingWords = self.c
        return (freq, betaSmooth, smoothingWords)
        
# TFIDF options

class TFIDF(FeatureOption):
    def getTypeName(self):
        return 'tfidf'
    def getTfidf(self):
        pass        
class TFIDFtrue(TFIDF):
    def getName(self):
        return 'true'
    def getTfidf(self):
        return True        
class TFIDFfalse(TFIDF):
    def getName(self):
        return 'false'
    def getTfidf(self):
        return False        
        
# Term filtering options
        
class TermFilter(FeatureOption):
    def getTypeName(self):
        return 'Term filter'
    def filterCorpus(self, corpus):
        pass
class TopNTermFilter(TermFilter):
    def __init__(self,n):
        self.n = n
    def getName(self):
        return 'top ' + str(self.n)
    def filterCorpus(self, corpus):
        corpus.setMinDocRank(self.n)
class NullFilter(TermFilter):
    def getName(self):
        return 'none'

