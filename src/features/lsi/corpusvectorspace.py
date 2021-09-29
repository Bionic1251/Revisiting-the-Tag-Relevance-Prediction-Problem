"""
The class CorpusVectorSpace is used by application.tasks.write_r_extracts, in combination with lsi.svd.SVD, for
    computing IMDB corpus features to be used in the R analysis during application.tasks.run_predict_tag_relevance
"""

import src.features.lsi.vectorspace
from src.features.lsi.vectorspace import VectorSpace
from src.features.lsi import moviedb


class CorpusVectorSpace(VectorSpace):

    def __init__(self, corpus):
        self.corpus = corpus
        self.movieDb = None

    def makeVectorForTag(self, tag):
        return self.corpus.makeDocVectorFromText(tag)
    
    def getVectors(self):
        return [self.corpus.docVectors[self.corpus.indexToDocName[index]] for index in sorted(self.corpus.indexToDocName.keys())]
    
    def getVectorName(self, vectorIndex):
        return self.corpus.indexToDocName[vectorIndex]
    
    def getVectorIndex(self, vectorName):
        return self.corpus.docNameToIndex[vectorName]
    
    def getVectorDesc(self, vectorIndex):
        if self.movieDb==None:
            self.movieDb = moviedb.MovieDb()
        movieId = self.getVectorName(vectorIndex)
        try:
            return self.movieDb.getTitle(movieId)
        except LookupError:
            return  movieId
