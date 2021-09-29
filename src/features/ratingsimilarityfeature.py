"""
This feature represents the similarity between a movie and a tag based on the cosine similarity of ratings vectors

For a <tag,movie_id> pair, the value of this feature is a float between -1 and 1, with more positive numbers
    indicating greater similarity

See ratingsim.ratingsimilarity
"""

from src.features import featuregenerator


class RatingSimilarityFeature(featuregenerator.FeatureGenerator):
    
    def __init__(self, ratingSimilarity):
        self.ratingSimilarity = ratingSimilarity
        self.warningGiven = False
        
    def getName(self):
        return "tag_movie_rating_similarity" 

    def getDescription(self):
        return "Returns cosine similarity between movie ratings and centroid of rating vectors for movies with tag."

    def getFeatures(self, movieId, tag, userId=None):
        try:
            featureVal = self.ratingSimilarity.getTagMovieSimilarity(tag, movieId)
        except LookupError:
            featureVal = 0
            if not self.warningGiven:
                self.warningGiven = True
                print('WARNING: Cannot calculate rating similarity, returning zero')
        return [(self.getName(), featureVal)]
