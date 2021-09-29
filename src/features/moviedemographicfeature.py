"""
Age and gender of raters of movies

NOTE: This is not currently being used
"""

from src.features import featuregenerator


class MovieDemographicFeature(featuregenerator.FeatureGenerator):
    
    def __init__(self, movieDemographic):
        self.movieDemographic = movieDemographic
        
    def getName(self):
        return "movie demographic" 
    
    def getFeatureNames(self):
        return ['avg_age', 'proportion_female']

    def getDescription(self):
        return "Returns avg age and female proportion of raters of movies"

    def getFeatures(self, movieId, tag, userId=None):   
        return [(self.getFeatureNames()[0], self.movieDemographic.getAvgAge(movieId)),
                (self.getFeatureNames()[1], self.movieDemographic.getProportionFemale(movieId)) ]

