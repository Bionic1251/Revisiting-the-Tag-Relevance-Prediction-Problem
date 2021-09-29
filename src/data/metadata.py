import re
#import src.data.sql_connections as sql_cnx
from src.data.data import *

NAME_SUFFIXES = re.compile(r"\bjr\b|\bjr.\b|\bsr\b|\bsr.\b|\bii\b|\biii\b")


class MetaData:
    def __init__(self, includeMovies=None):
        self.firstNameCounts = {}
        self.lastNameCounts = {}

        self.directors=set([])
        self.actors=set([])
        self.directorsByMovie = {}
        self.actorsByMovie = {}
        # sql = """ select movieId, starring, directedBy from movie_data """
        #values = df_movie_data[['movie_id', 'starring', 'directed_by']].to_numpy().tolist()
        #values = df_movie_data[['movie_id', 'starring', 'directed_by']].to_numpy().tolist()
        values = df_movie_data[['item_id', 'starring', 'directed_by']].to_numpy().tolist()
        for movieId, starring, directedBy in values:
            if includeMovies != None and movieId not in includeMovies:
                continue
            try:
                directors = set([s.strip() for s in directedBy.lower().split(',')])
            except AttributeError:
                directors = ''
            for director in directors:
                self.directors.add(director)
                self.processName(director)
            try:
                actors = set([s.strip() for s in starring.lower().split(',')])
            except AttributeError:
                actors = ''  # todo: missing actors
            for actor in actors:
                self.actors.add(actor)
                self.processName(actor)
            self.actorsByMovie[movieId] = actors
            self.directorsByMovie[movieId] = directors


    def processName(self, name):
        name = NAME_SUFFIXES.sub('',name)
        wordsInName = name.split()
        if len(wordsInName) < 2:
            return
        firstName = wordsInName[0]
        lastName = wordsInName[-1]
        self.firstNameCounts[firstName] = self.firstNameCounts.get(firstName, 0) + 1
        self.lastNameCounts[lastName] = self.lastNameCounts.get(lastName, 0) + 1

    def isActor(self, text):
        if text.lower() in self.actors:
            return True
    def isDirector(self, text):
        if text.lower() in self.directors:
            return True

    def getAllActors(self):
        return self.actors

    def getAllDirectors(self):
        return self.directors

    def getActors(self, movieId):
        return  self.actorsByMovie[movieId]

    def getDirectors(self, movieId):
        return self.directorsByMovie[movieId]

    def getNames(self, minCount):
        return set([name for name, count in self.firstNameCounts.items() if count >= minCount]) | set([name for name, count in self.lastNameCounts.items() if count >= minCount])

if __name__ == '__main__':
    md = MetaData()
    #print md.directors
    #print md.actors
    print(md.getActors(296))
#    print 'first names'
#    for count, name in reversed(sorted([(count, name) for name, count in md.firstNameCounts.items()])):
#        print '%d\t%s'%(count,name)
#
#    print 'last names'
#    for count, name in reversed(sorted([(count, name) for name, count in md.lastNameCounts.items()])):
#        print '%d\t%s'%(count,name)
#
#    print 'min 20'
#    print md.getNames(20)
#
