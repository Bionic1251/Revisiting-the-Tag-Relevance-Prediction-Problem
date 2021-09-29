"""
Subclasses of VectorSpace (see corpusvectorspace and tagvectorspace) are used by application.tasks.write_r_extracts, in
    combination with lsi.svd.SVD, for creating features to use in the R analysis during application.tasks.run_predict_tag_relevance
"""

class VectorSpace():

    def __init__(self):
        pass
    
    def getVectors(self):
        pass
    
    def getVectorName(self, vectorIndex):
        pass

    def getVectorIndex(self, vectorName):
        pass
    
    def getVectorDesc(self, vectorIndex):
        return self.getVectorName(vectorIndex)
    
    def makeVectorForTag(self, tag):
        pass
