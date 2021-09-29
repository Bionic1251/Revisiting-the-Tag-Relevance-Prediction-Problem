"""
For a <tag,movie_id> pair, this feature represents the (integer) number of distinct applications of the tag to the movie
"""

from src.features import featuregenerator

class TagCount(featuregenerator.FeatureGenerator):

    def __init__(self, tagEvents):
        self.tagEvents = tagEvents

    def getName(self):
        return 'tag_count'

    def getDescription(self):
        return 'Number of times tag has been applied to movie'
    
    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), self.tagEvents.getEventCount(movieId, tag))]    
        
     
        
