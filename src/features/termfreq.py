"""
This feature represents the frequency of a tag being referenced in the (IMDB) corpus

For a <tag,movie_id> pair the value is a float, the (transformed) frequency
"""

from src.features import featuregenerator


class TermFreq(featuregenerator.FeatureGenerator):
    
    def __init__(self, corpus, transform=None, qualifier=""):
        self.corpus = corpus
        self.transform = transform
        self.qualifier = qualifier
        if self.qualifier != "":
            self.qualifier = "_" + self.qualifier       

    def getName(self):
        return "term_freq_" + self.corpus.getName() + self.qualifier

    def getDescription(self):
        return "Returns term frequency in document(s) associated with a movie"

    def getFeatures(self, movieId, tag, userId=None):
        featureVal = self.corpus.getWordFreq(movieId, tag)
        if self.transform is not None:
            featureVal = self.transform.transformValue(featureVal, tag, movieId, self.corpus)
        return [(self.getName(), featureVal)]
        
     
        
