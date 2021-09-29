"""
Returns user's average rating of movie

NOTE: This is not currently being used
"""

from src.features import featuregenerator

class UserAvgRating(featuregenerator.FeatureGenerator):
    
    def __init__(self, ratings):
        self.ratings = ratings
        
    def getName(self):
        return "user_avg_rating" 

    def getDescription(self):
        return "Returns user's average movie rating"

    def getFeatures(self, movieId, tag, userId):
        return [(self.getName(), self.ratings.getAvgUserRating(userId))]
        
     
        
