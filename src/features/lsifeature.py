"""
This feature represents the (float) lsi similarity between a tag and a content vector (tags, text, etc.) associated with the movie

Also see the lsi module for the computation
"""

from src.features import featuregenerator

class LsiFeature(featuregenerator.FeatureGenerator):
    
    def __init__(self, svd, qualifier=""):
        self.svd = svd
        
        self.qualifier = qualifier
        if self.qualifier != "":
            self.qualifier = "_" + self.qualifier
        
    def getName(self):
        return "lsi" + self.qualifier

    def getDescription(self):
        return "Returns lsi similarity between a tag and a content vector (tags, text, etc.) associated with movie"

    def getFeatures(self, movieId, tag, userId=None):
        try:
            featureVal = self.svd.getSimByNameAndVector(vectorName=movieId, vector2=self.svd.vectorSpace.makeVectorForTag(tag))
        except LookupError:
            featureVal = 0
        
        return [ ( self.getName(), featureVal) ]
        
     
        
