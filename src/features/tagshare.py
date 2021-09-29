"""
NOTE: This is not currently being used
"""

from src.features import featuregenerator


class TagShare(featuregenerator.FeatureGenerator):
    
    def __init__(self, tagEvents, smoothingConstant=0):
        self.tagEvents = tagEvents
        self.smoothingConstant = smoothingConstant

    def getName(self):
        return 'tag_share'

    def getDescription(self):
        return 'Tag share of tag for movie'
    
    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), self.tagEvents.getTagShare(movieId, tag, self.smoothingConstant))]    

    
    
