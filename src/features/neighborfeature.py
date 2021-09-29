#Note: this has not been tested yet

import featuregenerator

class NeighborFeature(featuregenerator.FeatureGenerator):
    
    def __init__(self, feature, neighbors, weight=True, numNeighbors=10):
        self.feature = feature
        self.neighbors = neighbors
        self.weight = weight
        self.numNeighbors = numNeighbors
        
    def getName(self):
        return "neighbors_" + self.feature.getName()

    def getDescription(self):
        return "Returns feature value (%s) averaged over %d neighbors"%(self.feature.getName(), self.numNeighbors)

    def getFeatures(self, movieId, tag):
        sims = []
        vals = []
        neighbors = self.neighbors.getNeighbors(movieId, self.numNeighbors)
        if neighbors==None:
            return self.feature.getFeatures(movieId,tag)
        for sim, neighbor in neighbors:
            vals.append(self.feature.getFeatures(neighbor,tag)[0][1])
            sims.append(sim)
        if self.weight:
            featureVal = sum(sim*val for sim, val in zip(sims, vals)) / sum(sims)
        else:
            featureVal = sum(vals)/len(vals)
        return [(self.getName(), featureVal)]
        
     
        
