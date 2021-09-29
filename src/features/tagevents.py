"""
This module contains the TagEvents class, and handles the `tag_events` db table
"""
# import ml.sql_connections as sql_cnx
import csv
import pandas as pd
from src.common import Input
from src.data.paths import *


# DELETE: This class is in build_features.py




class DeleteTagEvents:
    """
    This class handles the `tag_events` database table

    Instance variables:
        eventCountByMovieTag: a dictionary from <movieId,tag> to number of applications
            (ie: eventCountByMovieTag[movieId][tag])
        eventCountByTagMovie: a dictionary from <tag,movieId> to number of applications
            (ie: eventCountByTagMovie[tag][movieId])
        numEventsByMovie: a dictionary from movieId to the total number of tag application events on the movie

    The following instance variables will only be set if the getNumDistinctTaggers flag in the constructor is true
        numDistinctTaggers: a dictionary from tag to the number of distinct users who have applied that tag
        maxTaggers: the number of distinct users who have applied the most-applied tag
    """

    def __init__(self, includeMovies=None, includeTags=None, getNumDistinctTaggers=False):
        """
        The constructor queries info from the `tag_events` database to fill instance variables
        :param includeMovies: a list of movies to consider (default to None, meaning include all tags)
        :param includeTags: a list of tags to consider (default to None, meaning include all tags)
        :param getNumDistinctTaggers: if True, query for, and set, numDistinctTaggers and maxTaggers
        """
        # Get a database connection and cursor
        #mlcnx = sql_cnx.getMLConnection()
        #mlcursor = mlcnx.cursor()

        # Query for the number of times each tag has been applied to each movie
        sql = """
            select movieId, lower(tag), count(*) from tag_events group by movieId, lower(tag)
        """
        #mlcursor.execute(sql)

        # Handle the data returned (counts of tag applications to movies)
        self.eventCountByMovieTag = {}
        self.eventCountByTagMovie = {}
        self.numEventsByMovie = {}
        # for movieId, tag, count in mlcursor.fetchall():
        
        for movieId, tag, count in sim_sql_query_tag_events():
            if includeMovies != None and movieId not in includeMovies:
                continue
            if includeTags != None and tag not in includeTags:
                continue
            if movieId not in self.eventCountByMovieTag:
                self.eventCountByMovieTag[movieId] = {}
            self.eventCountByMovieTag[movieId][tag] = count 
            self.numEventsByMovie[movieId] = self.numEventsByMovie.get(movieId, 0) + count
            if tag not in self.eventCountByTagMovie:
                self.eventCountByTagMovie[tag]={}
            self.eventCountByTagMovie[tag][movieId]= count
            
        if getNumDistinctTaggers:
            # Query for the number of distinct users who have tagged
            sql = """
                select lower(tag), count(distinct(userId)) from tag_events group by lower(tag)
            """
            mlcursor.execute(sql)
            self.numDistinctTaggers = dict(mlcursor.fetchall())
            self.maxTaggers = max(self.numDistinctTaggers.itervalues())


    def getMovies(self):
        """
        :return: a list of the movieIds that resulted from the queries in the constructor
        """
        return self.eventCountByMovieTag.keys()


    def getTags(self):
        """
        :return:a list of the tags that resulted from the queries in the constructor
        """
        return self.eventCountByTagMovie.keys()


    def getEventCount(self, movieId, tag):
        """
        :return: the number of times that tag has been applied to that movie (from the constructor query)
        """
        try:
            return self.eventCountByMovieTag[movieId][tag]
        except LookupError:
            # No applications found
            return 0


    def getEventCountsForMovie(self, movieId):
        """
        :return: a dictionary from tag to the number of applications on this movie
        """
        try:
            return self.eventCountByMovieTag[movieId]
        except LookupError:
            # No tags found for this movieId by the constructor query
            return {}


    def getEventExists(self, movieId, tag):
        """
        :return: True if the tag has been applied to the movie (according to the constructor query)
        """
        try:
            return (tag in self.eventCountByMovieTag[movieId])
        except LookupError:
            # The movie wasn't in the dictionary, so the tag wasn't applied
            return False


    def getTagsForMovie(self, movieId):
        """
        :return: a list of the tags that have been applied to this movie (empty if the movie isn't found)
        """
        try:
            return self.eventCountByMovieTag[movieId].keys()
        except LookupError:
            return []


    def getMoviesForTag(self, tag):
        """
        :return: a list of the movieIds to which the tag has been applied (empty if the tag isn't found)
        """
        try:
            return self.eventCountByTagMovie[tag].keys()
        except LookupError:
            return []


    def getTagShare(self, movieId, tag, smoothingConstant=0):
        """
        :return: the portion of tag events on the movie which are of the parameter tag, normalized by smoothingConstant
        """
        try:
            return float(self.getEventCount(movieId, tag)) / (self.numEventsByMovie[movieId] + smoothingConstant)
        except LookupError:
            return 0


    def getNumDistinctTaggers(self, tag):
        """
        :return: the number of distinct users who have applied this tag
        :raises AttributeError if getNumDistinctTaggers was False during instance construction
        """
        try:
            return self.numDistinctTaggers[tag]
        except LookupError:
            return 0


    def getMaxTaggers(self):
        """
        Simple accessor for maxTaggers
        :raises AttributeError if getNumDistinctTaggers was False during instance construction
        """
        return self.maxTaggers



if __name__ == '__main__':
    pass

