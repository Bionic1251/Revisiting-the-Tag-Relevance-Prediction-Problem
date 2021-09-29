""" load_rating_sim.py """
from src.data.paths import *
from loguru import logger
import pickle as cPickle
from src.features.build_features import TagEvents
from datetime import datetime
import sys
import math


def make_ratings_sim_obj(ratings_obj, survey_tags=[], survey_movies=[], application_movies=[]):
    """  load_ratings_sim.py  """
    # Build a set of movies to compare
    #surveyTags = mainsurvey.select_tags.getSurveyTagsFromFile()
    #surveyMovies = mainsurvey.select_movies.getSurveyMoviesFromFile()
    #applicationMovies = application.tasks.select_movies.getMoviesFromFile()
    allMovies = list(set(survey_movies)|set(application_movies))
    logger.info(f"All movies len = {len(allMovies)}")

    #ratingsData = cPickle.load(open(FILE_RATINGS_USER_MOVIE_ADJ, 'rb'))
    ratingsData = ratings_obj

    # Compute the similarities between movies
    similarityFunction= cosSim
    ratingSim = RatingSimilarity(survey_tags, ratingsData, similarityFunction, movies=allMovies)
    return ratingSim


def make_ratings_sim_pickle(fpath_ratings=EXTRACT_FILE_SURVEY_MOVIES,
                            fpath_ids=EXTRACT_FILE_MOVIES,
                            fpath_ids2=FILE_MOVIES,
                            out_pickle_file="rating_sim.pickle"):
    logger.info(f"Make rating similarity pickle")
    surveyTags = getSurveyTagsFromFile(fpath=fpath_ratings)
    surveyMovies = getSurveyMoviesFromFile(fpath=fpath_ids)
    applicationMovies = getMoviesFromFile(fpath=fpath_ids2)
    allMovies = list(set(surveyMovies)|set(applicationMovies))
    #
    # # Load adjusted ratings from load_ratings task
    ratingsData = cPickle.load(open(FILE_RATINGS_USER_MOVIE_ADJ, 'rb'))
    logger.info(f"Dump into {FILE_RATINGS_USER_MOVIE_ADJ}")

    # # Compute the similarities between movies
    similarityFunction = cosSim
    ratingSim = RatingSimilarity(surveyTags, ratingsData, similarityFunction, movies=allMovies)

    # # Write output
    #cPickle.dump(ratingSim, open(FILE_RATING_SIM, 'wb'))
    save_path = os.path.join(DIR_PICKLE_FILES, out_pickle_file)
    cPickle.dump(ratingSim, open(save_path, 'wb'))
    logger.info(f"Dump into {save_path}")


def length(X):
    return math.sqrt(sum(x*x for x in X))

def cosSim(X, Y):
    """
    :param X: the first vector
    :param Y: the second vector
    :return: the cosine similarity between two vectors X and Y
    """
    try:
        # Avoids doing zip which can be expensive if repeated often
        return sum(X[i]*Y[i] for i in range(max(len(X), len(Y))))/float(length(X)*length(Y))
    except ArithmeticError:
        return 0.0

def getSurveyTagsFromFile(fpath=EXTRACT_FILE_SURVEY_MOVIES):
    """
    :return: The set of tags from EXTRACT_FILE_SURVEY
    """
    f = open(fpath, 'r')
    tags = set([line.strip() for line in f])
    return tags


def getSurveyMoviesFromFile(fpath=EXTRACT_FILE_MOVIES):
    """
    :return: The set of movies from EXTRACT_FILE
    """
    f = open(fpath, 'r')
    movies = set([int(line.strip()) for line in f])
    return movies


def getMoviesFromFile(fpath=FILE_MOVIES):
    """
    :return the set of movies from FILE_MOVIES
    """
    f = open(fpath, 'r')
    movies = set([int(line.strip()) for line in f])
    return movies


class RatingSimilarity:
    """
    Class for computing the similarity between movies and tags based on cosine similarity of ratings vectors
    An instance of this class is built in the load_rating_sim task, and then dumped as output using cPickle

    The main instance variable is similarityMatrix, which is a 2-dimensional dictionary (tag, movieId) -> similarity.
        similarityMatrix[tag][movieId] is the cosine similarity of the ratings vectors of that tag and that movie
    """

    def __init__(self, tags, ratingsData, similarityFunction=cosSim, movies=None, moviesByTag=None, minOverlap=5):
        """
        The constructor builds the similarityMatrix.
        :param tags: a set of the string names of tags to include in the computation
        :param ratingsData: an ml.ratings.Ratings object, with ratings adjusted by user mean and by movie mean ratings
        :param similarityFunction: a function that accepts two ratings vectors and returns their (float) similarity
                ie: similarity = similarityFunction(tagRatingsVector, movieRatingsVector)
        :param movies: None, or a set of movieIds to include in the computation.
                Either movies or moviesByTag must be non-None.
        :param moviesByTag: None, or a dictionary of tag -> list of movies to include in the computation for that tag
                Either movies or moviesByTag must be non-None.
        :param minOverlap: A threshold number of users. For each (tag,movie) pair, if the number of users who have rated
                both 1) the movie, and 2) any other movie with that tag, is less than this threshold, then the similarity
                between the tag and the movie is 0
        """
        # Check inputs
        if movies is not None and moviesByTag is not None:
            raise Exception("May not specify both movies and moviesByTag")
        elif movies is None and moviesByTag is None:
            raise Exception("Must specify either movies or moviesByTag")

        # Load data from the tag_events table
        tagEvents = TagEvents(includeTags=tags)
        logger.info('Loaded tag events')

        # Compute similarity between tag and movie rating vectors
        self.similarityMatrix = {}
        for index, tag in enumerate(tags):
            # if index % 5 == 0:
            #     logger.info('processed %d tags out of %d, %s\n' % (index, len(tags), datetime.now()))
            #     sys.stdout.flush()

            # Accumulate the ratings of movies with this tag
            ratingSumForTagByUser = {}
            ratingCountForTagByUser = {}
            for movieId in tagEvents.getMoviesForTag(tag):
                # Get the ratings for each movie, skipping if there are none
                try:
                    movieRatings = ratingsData.getMovieRatings(movieId)
                except LookupError:
                    continue
                # Accumulate the ratings
                for userId, rating in movieRatings.items():
                    ratingSumForTagByUser[userId] = ratingSumForTagByUser.get(userId, 0.0) + rating
                    ratingCountForTagByUser[userId] = ratingCountForTagByUser.get(userId, 0) + 1

            # The set of users who have rated any movie to which this tag has been applied
            usersWhoRatedMovieWithTag = set(ratingSumForTagByUser.keys())

            # Build the similarity matrix
            self.similarityMatrix[tag] = {}
            if moviesByTag is not None:
                movies = moviesByTag[tag]
            for movieId in movies:
                # Get the ratings of this movie (<userId, rating> pairs)
                try:
                    movieRatings = ratingsData.getMovieRatings(movieId)
                except LookupError:
                    continue

                # The users who have both 1) rated this movie, 2) rated any movie to which this tag has been applied
                # commonUsers = usersWhoRatedMovieWithTag & set(movieRatings.keys())
                commonUsers = [userId for userId in movieRatings if userId in usersWhoRatedMovieWithTag]

                # Build the ratings vectors of this tag and this movie, on the domain of their intersecting rater users
                tagExistsOnMovie = tagEvents.getEventExists(movieId, tag)
                if tagExistsOnMovie:
                    # Must exclude any intersecting users whose only rating on this tag is this movie
                    tagRatingVector = [float(
                        ratingSumForTagByUser[userId] + ratingsData.getAvgUserRating(userId) - movieRatings[userId]) / (
                                               ratingCountForTagByUser[userId] - 1 + 1)
                                       for userId in commonUsers if ratingCountForTagByUser[userId] > 1]
                    movieRatingVector = [movieRatings[userId] for userId in commonUsers if
                                         ratingCountForTagByUser[userId] > 1]
                else:
                    tagRatingVector = [float(ratingSumForTagByUser[userId] + ratingsData.getAvgUserRating(userId)) / (
                            ratingCountForTagByUser[userId] + 1)
                                       for userId in commonUsers]
                    movieRatingVector = [movieRatings[userId] for userId in commonUsers]

                # Compute the similarity
                if len(tagRatingVector) < minOverlap:
                    similarity = 0
                else:
                    similarity = similarityFunction(movieRatingVector, tagRatingVector)
                #logger.info(similarity)
                self.similarityMatrix[tag][movieId] = similarity

    def getSimilarityWithTag(self, tag):
        return list(reversed(sorted([(sim, movieId) for movieId, sim in self.similarityMatrix[tag].items()])))

    def getTagMovieSimilarity(self, tag, movieId):
        return self.similarityMatrix[tag][movieId]

