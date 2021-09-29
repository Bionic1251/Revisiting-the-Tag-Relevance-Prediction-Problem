"""
The class TagVectorSpace is used by application.tasks.write_r_extracts, in combination with lsi.svd.SVD, for computing
    computing tag features to be used in the R analysis during application.tasks.run_predict_tag_relevance
"""

import src.features.lsi.vectorspace
#from ml import moviedb
from src.features.lsi.vectorspace import VectorSpace
from src.features.lsi import moviedb


class TagVectorSpace(VectorSpace):

    def __init__(self, tagEvents, tagShare=False, smoothingConstant=0):
        self.tagEvents = tagEvents
        self.tagShare = tagShare
        self.smoothingConstant = smoothingConstant
        tags = self.tagEvents.getTags()
        self.tagToIndex = dict([(tag, index) for index, tag in enumerate(tags)])
        self.indexToTag = dict([(index, tag) for index, tag in enumerate(tags)])
        self.vectors = []
        self.vectorIndexToName = {}
        self.vectorNameToIndex = {}
        self.movieDb = None
        vectorIndex = 0
        for movieId in tagEvents.getMovies():
            self.vectors.append(self.makeVectorForMovie(movieId))
            self.vectorIndexToName[vectorIndex] = movieId
            self.vectorNameToIndex[movieId] = vectorIndex
            vectorIndex +=1

    def makeVectorForTag(self, tag):
        return {self.tagToIndex[tag]:1}
    
    def makeVectorForMovie(self, movieId):
        tagsForMovie = self.tagEvents.getTagsForMovie(movieId)
        if self.tagShare:
            return dict([(self.tagToIndex[tag], self.tagEvents.getTagShare(movieId, tag, self.smoothingConstant)) for tag in tagsForMovie])
        else:
            return dict([(self.tagToIndex[tag],1) for tag in tagsForMovie])
    
    def getVectors(self):
        return self.vectors
    
    def getVectorName(self, vectorIndex):
        return self.vectorIndexToName[vectorIndex]
    
    def getVectorIndex(self, vectorName):
        return self.vectorNameToIndex[vectorName]
    
    def getVectorDesc(self, vectorIndex):
        if self.movieDb==None:
            self.movieDb = moviedb.MovieDb()
        movieId = self.getVectorName(vectorIndex)
        try:
            return self.movieDb.getTitle(movieId)
        except LookupError:
            return  movieId
