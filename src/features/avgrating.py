"""
This feature represents the average rating of the movie

The value is a float between 0 and 5, with higher values indicating that users enjoyed the movie more
"""

from src.features import featuregenerator


class AvgRating(featuregenerator.FeatureGenerator):
    
    def __init__(self, ratings):
        self.ratings = ratings
        
    def getName(self):
        return "avg_movie_rating" 

    def getDescription(self):
        return "Returns average rating of movie"

    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), self.ratings.getAvgMovieRating(movieId))]
        
     
        
