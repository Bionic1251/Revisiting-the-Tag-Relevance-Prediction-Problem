# TODO: This module appears to be unused

import featuregenerator
import ml.sql_connections as sql_cnx
from simcalc.sc_lib import utils


class RunTime(featuregenerator.FeatureGenerator):
    
    def __init__(self):
        mlcnx = sql_cnx.getMLConnection()
        mlcursor = mlcnx.cursor()
        sql = """
            select movieId, runTime from rt
        """
        mlcursor.execute(sql)
        self.runTimes = {}
        for movieId, runTimeString in mlcursor.fetchall():
            words = runTimeString.split()
            minutes = 0
            hours = 0
            for i, word in enumerate(words):
                if word[0]=='h' and i > 0:
                    hours = int(words[i-1])
                elif word[0]=='m' and i > 0:
                    minutes = int(words[i-1])
            if hours > 0 or minutes > 0:
                self.runTimes[movieId] = hours * 60 + minutes

        self.avgRunTime = utils.mean(self.runTimes.values())
     
    def getName(self):
        return "runtime"

    def getDescription(self):
        return "Returns run time of movie"

    def getFeatures(self, movieId, tag):
        return [(self.getName(), self.runTimes.get(movieId, None))]
        
     
        
