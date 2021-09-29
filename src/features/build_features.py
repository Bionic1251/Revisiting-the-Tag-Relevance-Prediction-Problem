#!/usr/bin/env python
"""
from: write_r_extracts.py
Build features from pickle files data/interim/ into processed/
"""
import click
import logging
import pdb
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.paths import *
import time
import gc
import pandas as pd
import pickle as cPickle
from src.data.data import *
from src.common import Input
from src.features.termfreq import *
from src.features.featureoptions import *
import src.features.lsi as lsi
import src.features as features
from src.features.mainsurvey.select_tags import getTagsFromFile
from src.data.make_books_interim import *
from src.input_data import *


def run(fpath_mainanswers,
        fpath_ratings=EXTRACT_FILE_SURVEY_MOVIES,
        fpath_ids=EXTRACT_FILE_MOVIES):

    tags_survey = getSurveyTagsFromFile(fpath=fpath_ratings)

    movies = getSurveyMoviesFromFile(fpath=fpath_ids)

    main_survey_data = SurveyData(fpath_mainanswers,
                                  excludeAdditionalRounds=False,
                                  excludeNotSure=True,
                                  excludeAllEqual=True,
                                  )

    logger.info(f"Mainanswers file: {fpath_mainanswers}")
    logger.info(f"Ratings file: {fpath_ratings}")
    logger.info(f"Items IDs file: {fpath_ids} ")

    # # Start setting up features
    featureExtract = FeatureExtract(tags_survey, movies)
    featureResponseExtract = FeatureResponseExtract(main_survey_data)

    # # Set up tag features (exists and counts)
    tagEvents = TagEvents(includeTags=tags_survey, m_or_b=input_folder)
    tagExistsFeature = TagExists(tagEvents)
    tagCountFeature = TagCount(tagEvents)
    featureExtract.addFeatures([tagExistsFeature, tagCountFeature])
    featureResponseExtract.addFeatures([tagExistsFeature, tagCountFeature])
    del tagExistsFeature, tagCountFeature
    del tagEvents
    gc.collect()

    # # Set up rating similarity feature
    ratingSim = cPickle.load(open(FILE_RATING_SIM, 'rb'))
    logger.info(f"Load ratings sim pickle file: {FILE_RATING_SIM}")
    ratingSimFeature = RatingSimilarityFeature(ratingSim)
    featureExtract.addFeature(ratingSimFeature)
    featureResponseExtract.addFeature(ratingSimFeature)
    del ratingSimFeature
    del ratingSim
    gc.collect()

    # Set up IMDB log term frequency feature
    imdbCorpus = cPickle.load(open(FILE_IMDB_CORPUS, 'rb'),  encoding='latin1')
    logger.info(f"Load imdb corpus pickle file: {FILE_IMDB_CORPUS}")
    logImdbFeature = TermFreq(imdbCorpus,
                              transform=LogTransformRelativeBeta3(imdbCorpus, tags_survey, c=1),
                              qualifier="log")
    featureExtract.addFeature(logImdbFeature)
    featureResponseExtract.addFeature(logImdbFeature)
    del logImdbFeature
    gc.collect()

    # Set up IMDB corpus SVD features
    cvs = lsi.CorpusVectorSpace(imdbCorpus)
    for svdRank in [25, 175]:
        # Compute SVD of IMDB Corpus
        corpusSVD = lsi.SVD(cvs, svdRank=svdRank, name="_corpus_%d"%svdRank)
        corpusSVD.load()

        # Add feature for this
        lsiFeature = features.LsiFeature(corpusSVD, qualifier="imdb_" + str(svdRank))
        featureExtract.addFeature(lsiFeature)
        featureResponseExtract.addFeature(lsiFeature)
        del lsiFeature
        del corpusSVD
        gc.collect()
    del cvs
    del imdbCorpus
    gc.collect()

    # # Set up IMDB no-stem log term frequency feature
    imdbNoStemCorpus = cPickle.load(open(FILE_IMDB_NOSTEM_CORPUS, 'rb'),  encoding='latin1')
    logger.info(f"Load no stem imdb corpus pickle file: {FILE_IMDB_NOSTEM_CORPUS}")
    imdbNoStemFeature = TermFreq(imdbNoStemCorpus,
                                 transform=LogTransformRelativeBeta3(imdbNoStemCorpus, tags_survey, c=1),
                                 qualifier="log_nostem")
    featureExtract.addFeature(imdbNoStemFeature)
    featureResponseExtract.addFeature(imdbNoStemFeature)
    del imdbNoStemFeature
    del imdbNoStemCorpus
    gc.collect()

    # # Set up average rating feature
    ratingsData = cPickle.load(open(FILE_RATINGS, 'rb'),  encoding='latin1')
    logger.info(f"Load ratings pickle file: {FILE_RATINGS}")
    avgRatingFeature = features.AvgRating(ratingsData)
    featureExtract.addFeature(avgRatingFeature)
    featureResponseExtract.addFeature(avgRatingFeature)
    del avgRatingFeature
    del ratingsData
    gc.collect()

    # # Compute tag event SVD, and set up tag lsi feature
    tagEvents = TagEvents(includeTags=getTagsFromFile(file_with_tags=TAGS_SURVEY_BOOKS),
                          m_or_b=Input.BOOKS)

    tvs = lsi.TagVectorSpace(tagEvents)
    svdRank = 75
    tagSVD = lsi.SVD(tvs, svdRank=svdRank, name="_tags_%d"%svdRank)
    tagSVD.load()
    tagLsiFeature = features.LsiFeature(tagSVD, qualifier='tags_' + str(svdRank))
    featureExtract.addFeature(tagLsiFeature)
    featureResponseExtract.addFeature(tagLsiFeature)
    del tagLsiFeature
    del tagSVD
    del tagEvents
    gc.collect()

    # # Write the extract files for the R process
    featureExtract.writeExtract(FILE_DATA_PREDICT_RELEVANCE)
    featureResponseExtract.writeExtract(FILE_DATA_TRAIN_RELEVANCE)
    logger.info(f"Save features into {FILE_DATA_PREDICT_RELEVANCE}")
    logger.info(f"Save features into {FILE_DATA_TRAIN_RELEVANCE}")


def getSurveyTagsFromFile(fpath):
    """
    :return: The set of tags from EXTRACT_FILE_SURVEY
    """
    f = open(fpath, 'r')
    tags = set([line.strip() for line in f])
    return tags


def getSurveyMoviesFromFile(fpath):
    """
    :return: The set of movies from EXTRACT_FILE
    """
    f = open(fpath, 'r')
    movies = set([int(line.strip()) for line in f])
    return movies


EXCLUDE_USERS = []

EXCLUDE_TAGS = []


class SurveyData(object):

    def __init__(self,
                 importFileName,
                 excludeAdditionalRounds,
                 excludeNotSure,
                 excludeAllEqual,
                 include_tags=None,
                 ):
        self.excludeAdditionalRounds = excludeAdditionalRounds
        self.excludeNotSure = excludeNotSure
        self.responses = []
        self.distinctTags = set([])
        self.distinctUsers = set([])
        self.distinctMovies = set([])
        self.numResponses = {}
        self.numResponsesByRound = {}
        distinctResponsesByUser = {}
        responsesTemp = []

        # Parse the file
        with open(importFileName, 'r') as f:
            for i, line in enumerate(f):
                # uid, roundIndex, movieId, tag, response
                # 173626	0	3108	mission from god	3
                vals = line.strip().split('\t')
                userId = int(vals[0])
                if userId in EXCLUDE_USERS:
                    continue
                roundIndex = int(vals[1])
                movieId = int(vals[2])
                tag = vals[3]
                if include_tags is not None:
                    if tag not in include_tags:
                        continue
                if tag in EXCLUDE_TAGS:
                    continue
                response = int(vals[4])
                if excludeAdditionalRounds and roundIndex > 0:
                    continue
                if excludeNotSure and response==-1:
                    continue
                if userId not in distinctResponsesByUser:
                    distinctResponsesByUser[userId] = set([])
                distinctResponsesByUser[userId].add(response)
                responsesTemp.append((movieId, tag, userId, response, roundIndex))

        for movieId, tag, userId, response, roundIndex in responsesTemp:
            if excludeAllEqual and len(distinctResponsesByUser[userId])==1:
                continue
            self.responses.append((movieId, tag, userId, response))
            self.distinctTags.add(tag)
            self.distinctUsers.add(userId)
            self.distinctMovies.add(movieId)
            self.numResponses[tag] = self.numResponses.get(tag,0) + 1
            self.numResponsesByRound[roundIndex] = self.numResponsesByRound.get(roundIndex, 0) + 1

    def getResponses(self):
        return self.responses

    def getResponsesAsDict(self):
        uniqueMovieTagPairs = set([(movieId, tag) for movieId, tag, userId, response in self.responses])
        responsesAsDict = dict([((movieId, tag),[]) for movieId, tag in uniqueMovieTagPairs])
        for movieId, tag, userId, response in self.responses:
            responsesAsDict[(movieId, tag)].append((userId,response))
        return responsesAsDict

    def getNumResponses(self,tag):
        return self.numResponses[tag]

    def getDistinctTags(self):
        return sorted(list(self.distinctTags))

    def getDistinctUsers(self):
        return sorted(list(self.distinctUsers))

    def getDistinctMovies(self):
        return sorted(list(self.distinctMovies))

    def report(self):
        if self.excludeNotSure:
            responseHistogram = {1:0, 2:0, 3:0, 4:0, 5:0}
        else:
            responseHistogram = {-1:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        oneToFiveResponses = []
        for movieId, tag, userId, response in self.responses:
            responseHistogram[response] += 1
            if response > 0:
                oneToFiveResponses.append(response)
        numResponses = sum(responseHistogram.values())
        print('\nexcludeNotSure:',self.excludeNotSure)
        print('excludeAdditionalRounds:',self.excludeAdditionalRounds)
        print('Num responses:', numResponses )
        print('Histogram:', responseHistogram)
        if len(oneToFiveResponses) > 0:
            print('Avg response: %.2f'%(float(sum(oneToFiveResponses))/len(oneToFiveResponses)))
        if self.excludeNotSure:
            pass
        else:
            if numResponses > 0:
                print('Fraction not sure: %.2f'%(float(responseHistogram[-1])/numResponses))
        if not self.excludeAdditionalRounds:
            print('Num responses by round')
            for roundIndex in sorted(self.numResponsesByRound.keys()):
                print('%d: %d'%(roundIndex, self.numResponsesByRound[roundIndex]))
            self.numResponsesByRound[roundIndex] = self.numResponsesByRound.get(roundIndex, 0) + 1
        print('Num users skipped', len(EXCLUDE_USERS))
        print('Num tags skipped', len(EXCLUDE_TAGS))
        print('num users', len(self.getDistinctUsers()))


class FeatureExtract:
    def __init__(self, tags, items_ids=None):
        self.tags = list(tags)
        self.movies = list(items_ids)
        self.featureVals = []
        self.featureNames = []

    def __str__(self):
        f_names = '[' + ", ".join([str(e) for e in self.featureNames]) + ']'
        f_vals_len = len(self.featureVals)
        return f"Feature names = {f_names} len(feature values) = {f_vals_len}"

    def addFeatures(self, features, debug=False):
        for feature in features:
            self.addFeature(feature, debug)

    def addFeature(self, feature, debug=False):
        self.featureNames.append(feature.getName())
        newColumn = []
        for tag in self.tags:
            for movieId in self.movies:
                featureVal = self.getFeatureVal(feature, movieId, tag, userId=None)
                newColumn.append(featureVal)
        self.featureVals.append(newColumn)

    def writeExtract(self, extractFileName):
        rFile = RFile(extractFileName)
        rFile.writeHeader(['tag', 'movieId'] + self.featureNames)
        tagColumn = []
        movieColumn = []
        for tag in self.tags:
            for movieId in self.movies:
                tagColumn.append(tag)
                movieColumn.append(movieId)
        allColumns = [tagColumn, movieColumn] + self.featureVals
        rows = zip(*allColumns)
        for row in rows:
            rFile.writeDetail(row)
        rFile.close()

    def getFeatureVal(self, feature, movieId, tag, userId):
        if len(feature.getFeatures(movieId, tag, userId)) > 1:
            raise Exception("Only single feature values allowed")
        return feature.getFeatures(movieId, tag, userId)[0][1]


class RFile:
    def __init__(self, fileName, delimeter='\t'):
        # self.o = file(fileName, 'w')
        self.o = open(fileName, 'w')
        self.delimeter = delimeter

    def writeHeader(self, columnLabels):
        self.o.write(self.delimeter.join(map(self._addQuotes, columnLabels)) + '\n')

    def writeDetail(self, vals):
        formattedVals = []
        for val in vals:
            if val == None:
                formattedVal = 'NA'
            elif isinstance(val, float):
                formattedVal = '%.4g' % val
            else:
                formattedVal = str(val)
            formattedVals.append(formattedVal)
        self.o.write(self.delimeter.join(formattedVals) + '\n')

    def close(self):
        self.o.close()

    def _addQuotes(self, s):
        return '\"' + s + '\"'


class FeatureResponseExtract:
    def __init__(self, surveyData, items_ids=None):
        self.surveyData = surveyData
        self.responsesAsDict = surveyData.getResponsesAsDict()
        self.headers = ['tag', 'movieId', 'userId', 'survey_response', 'rank', 'sample_percentile']
        self.mainData = []
        self.additionalColumns = []
        self.items_ids = items_ids

    def addFeatures(self, features, debug=False):
        for feature in features:
            self.addFeature(feature, debug)

    def addFeature(self, feature, debug=True):
        #if debug:
        #    logger.info('Processing feature %s %s\n' % (feature.getName(), time.strftime('%x %X')))
        self.headers.append(feature.getName())
        if len(self.mainData) == 0:
            self._addFirstFeature(feature)
        else:
            self._addAdditionalFeature(feature)

    def _addFirstFeature(self, feature):
        movies = MovieSet(imdbDataRequired=False, m_or_b=Input.BOOKS, items_ids=self.items_ids).movies
        numMovies = len(movies)
        for tag in self.surveyData.getDistinctTags():
            sorter = []
            for movieId in movies:
                featureVal = self.getFeatureVal(feature, movieId, tag, userId=None)
                if featureVal is None:
                    logger.info("featureVal is None")
                    continue
                # REMOVE THIS!!!!
                sorter.append((featureVal, movieId))
            sorter.sort()
            sorter.reverse()
            for index, (featureVal, movieId) in enumerate(sorter):
                try:
                    responses = self.responsesAsDict[(movieId, tag)]
                except LookupError:
                    continue
                rank = index + 1
                samplePercentile = float(rank) / numMovies
                for userId, response in responses:
                    row = [tag, movieId, userId, response, rank, samplePercentile, featureVal]
                    self.mainData.append(row)

    def _addAdditionalFeature(self, feature):
        newColumn = []
        for tag, movieId, userId, response, rank, samplePercentile, mainFeatureVal in self.mainData:
            newColumn.append(self.getFeatureVal(feature, movieId, tag, userId))
        self.additionalColumns.append(newColumn)

    def getFeatureVal(self, feature, movieId, tag, userId):
        a = feature.getFeatures(movieId, tag, userId)
        if len(a) > 1:
            raise Exception("Only single feature values allowed")
        #ret = feature.getFeatures(movieId, tag, userId)[0][1]
        ret = a[0][1]
        return ret

    def writeExtract(self, extractFileName):
        rFile = RFile(extractFileName)
        rFile.writeHeader(self.headers)
        mainColumns = list(zip(*self.mainData))
        allColumns = mainColumns + self.additionalColumns
        allRows = zip(*allColumns)
        for row in allRows:
            rFile.writeDetail(row)
        rFile.close()



class MovieSet:
    """
    Class for loading a subset of movies

    The main instance variable is movies, which is the set of movieIds queried from the database
    """

    def __init__(self,
                 minRatings=100,
                 minTags=0,
                 imdbDataRequired=False,
                 m_or_b=Input.MOVIES,
                 items_ids=None):
        """
        The constructor for this class builds the set of movies based on the given parameters
        :param minRatings: the restriction on the minimum number of ratings for an included movie
        :param minTags: the restriction on the minimum number of tags for an included movie
        :param imdbDataRequired: if True, only include movies which have already been scraped from the IMDb site
        """
        self.minRatings = minRatings
        query_data_from_mysql = False
        if items_ids is None:
            if m_or_b is Input.MOVIES:
                if query_data_from_mysql:
                    # First get the movieIds that satisfy the popularity constraints
                    # mlcnx = sql_cnx.getMLConnection()
                    # mlcursor = mlcnx.cursor()
                    sql = """select movieId from %s where popularity >= %d""" % (TABLE_MOVIE_DATA, minRatings)
                    # mlcursor.execute(sql)
                    self.movies = set([row[0] for row in mlcursor.fetchall()])
                else:
                    self.movies = set(movie_ids_movie_data())
            else:
                # books actually
                self.movies = get_books_ids()
        else:
            self.movies = items_ids

        if imdbDataRequired:
            # Exclude movies which have not been scraped
            tncnx = sql_cnx.getIMDBConnection()
            tncursor = tncnx.cursor()
            sql = """-select distinct(movieId) from %s where status='s'""" % TABLE_IMDB_EXTRACT
            tncursor.execute(sql)
            self.movies = self.movies & set([row[0] for row in tncursor.fetchall()])

        if minTags > 0:
            # Exclude movies that do not satisfy the tag count constraint
            sql = """-select movieId from %s group by movieId having count(*) >= %d""" % (TABLE_TAG_EVENTS, minTags)
            mlcursor.execute(sql)
            self.movies = self.movies & set([row[0] for row in mlcursor.fetchall()])



def sim_sql_query_tag_events():
    df = pd.read_csv(PATH_CSV_TAG_EVENT, sep='\t')
    vals = df.values.tolist()
    return vals


def read_books_tag_events(tags_of_intereset=None):
    input_file = FILE_TAG_EVENTS_BOOKS
    with open(input_file, 'r') as f:
        csv_r = csv.reader(f, delimiter='\t')
        for row in csv_r:
            b_id = row[0]
            shelve = row[1]
            count = row[2]
            if tags_of_intereset is not None and shelve in tags_of_intereset:
                yield b_id, shelve, count
            else:
                yield b_id, shelve, count


class TagEvents:
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

    def __init__(self,
                 includeMovies=None,
                 includeTags=None,
                 getNumDistinctTaggers=False,
                 m_or_b=Input.BOOKS,
                 items_ids_tags_counts_list=None):
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
        #sql = """ select movieId, lower(tag), count(*) from tag_events group by movieId, lower(tag) """
        #mlcursor.execute(sql)

        # Handle the data returned (counts of tag applications to movies)
        self.eventCountByMovieTag = {}
        self.eventCountByTagMovie = {}
        self.numEventsByMovie = {}

        if items_ids_tags_counts_list is None:
            # data_source = sim_sql_query_tag_events() if m_or_b is Input.MOVIES else read_books_tag_events()
            data_source = read_books_tag_events()
        else:
            data_source = items_ids_tags_counts_list

        for movieId, tag, count in data_source:
            movieId = int(movieId)
            if includeMovies != None and movieId not in includeMovies:
                continue
            if includeTags != None and tag not in includeTags:
                continue
            if movieId not in self.eventCountByMovieTag:
                self.eventCountByMovieTag[int(movieId)] = {}
            self.eventCountByMovieTag[int(movieId)][tag] = count
            self.numEventsByMovie[int(movieId)] = int(self.numEventsByMovie.get(movieId, 0)) + int(count)
            if tag not in self.eventCountByTagMovie:
                self.eventCountByTagMovie[tag] = {}
            self.eventCountByTagMovie[tag][int(movieId)] = count

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


class FeatureGenerator:
    def __init__(self):
        pass

    def getName(self):
        """Returns the human-interpretable name of the feature class"""

    def getDescription(self):
        """Returns the human-interpretable description of the feature class"""

    def __repr__(self):
        return self.getName()

    def __str__(self):
        return self.getName()

    def getFeatureNames(self):
        return [self.getName()]

    # Override this if multiple features, or different name

    def getFeatures(self, movieId, tag, userId):
        """
            Takes as input movieId, tag, and (optionally) userId.  Return a list of the features.
            Each features consist of a (feature name, feature value) pair.
        """
        pass


class TagExists(FeatureGenerator):

    def __init__(self, tagEvents):
        self.tagEvents = tagEvents

    def getName(self):
        return 'tag_exists'

    def getDescription(self):
        return 'Whether or not tag has been applied to movie'

    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), int(self.tagEvents.getEventExists(movieId, tag)))]


class TagCount(FeatureGenerator):

    def __init__(self, tagEvents):
        self.tagEvents = tagEvents

    def getName(self):
        return 'tag_count'

    def getDescription(self):
        return 'Number of times tag has been applied to movie'

    def getFeatures(self, movieId, tag, userId=None):
        return [(self.getName(), self.tagEvents.getEventCount(movieId, tag))]


class RatingSimilarityFeature(FeatureGenerator):

    def __init__(self, ratingSimilarity):
        self.ratingSimilarity = ratingSimilarity
        self.warningGiven = False

    def getName(self):
        return "tag_movie_rating_similarity"

    def getDescription(self):
        return "Returns cosine similarity between movie ratings and centroid of rating vectors for movies with tag."

    def getFeatures(self, movie_id, tag, userId=None):
        try:
            featureVal = self.ratingSimilarity.getTagMovieSimilarity(tag, movie_id)
        except LookupError:
            featureVal = 0
            if not self.warningGiven:
                self.warningGiven = True
                logger.info(f'WARNING: Cannot calculate rating similarity for tag={tag} movie_id={movie_id}, returning zero')
        return [(self.getName(), featureVal)]


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # Now by default: if input_folder == Input.BOOKS:
    make_mainanswers_compatible(output_file=FILE_BOOKS_MAINANSWERS_INTERIM)
    run(FILE_BOOKS_MAINANSWERS_INTERIM,
        fpath_ratings=EXTRACT_FILE_SURVEY_BOOKS,
        fpath_ids=EXTRACT_FILE_BOOKS)

