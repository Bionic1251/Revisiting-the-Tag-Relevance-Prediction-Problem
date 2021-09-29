"""
This feature represents/creates whether a certain tag event actually occurred

The value of this feature for a <tag,movie_id pair> is:
    1   if the tag has been applied to the movie
    0   otherwise
"""

from src.features import featuregenerator


class TagExists(featuregenerator.FeatureGenerator):
    
    def __init__(self, tagEvents):
        self.tagEvents = tagEvents
    
    def getName(self):
        return 'tag_exists'

    def getDescription(self):
        return 'Whether or not tag has been applied to movie'
    
    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), int(self.tagEvents.getEventExists(movieId, tag)))]    
        
        
     
        
