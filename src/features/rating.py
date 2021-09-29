"""
Returns user's rating of movie

NOTE: This is not currently being used
"""

from src.features import featuregenerator


class Rating(featuregenerator.FeatureGenerator):
    
    def __init__(self, ratings):
        self.ratings = ratings
        
    def getName(self):
        return "movie_rating" 

    def getDescription(self):
        return "Returns user's rating of movie"

    def getFeatures(self, movieId, tag, userId):
        return [(self.getName(), self.ratings.getMovieRating(movieId, userId))]
        
     
        
