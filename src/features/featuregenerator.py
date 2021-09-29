# A generator for features used to predict tags for a movie. 

class FeatureGenerator:
    def __init__(self):
        pass

    def getName(self):
        """Returns the human-interpretable name of the feature class"""

    def getDescription(self):
        """Returns the human-interpretable description of the feature class"""

    def __repr__(self):
        return self.getName()
        
    def __str__(self):
        return self.getName()

    def getFeatureNames(self):
        return [self.getName()]
    #Override this if multiple features, or different name

    def getFeatures(self, movieId, tag, userId):
        """
            Takes as input movieId, tag, and (optionally) userId.  Return a list of the features.
            Each features consist of a (feature name, feature value) pair.
        """
        pass
