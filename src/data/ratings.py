"""load_ratings.py"""
from src.settings import DEFAULT_RATING
from src.data.paths import *
from loguru import logger
import pickle as cPickle
import gc
import gzip
import time
from datetime import datetime
import sys
import math


def make_ratings_obj(ratings_file=FILE_RATINGS_EXTRACT_MOVIES):
    """ load_ratings.py """
    r_ratings = Ratings(adjustByUserMean=False, adjustByMovieMean=False)
    r_ratings.load(ratings_file)
    return r_ratings


def make_ratings_obj_adj(ratings_file=FILE_RATINGS_EXTRACT_MOVIES):
    """ load_ratings.py """
    r_ratings_user_movie_adj = Ratings(adjustByUserMean=True,
                                       adjustByMovieMean=True)
    r_ratings_user_movie_adj.load(ratings_file)
    return r_ratings_user_movie_adj



def make_ratings_pickle_files(ratings_file=FILE_RATINGS_EXTRACT_MOVIES,
                              dump=True,
                              delete_ratings_with_small_number_of_votes=True,
                              out_pickle_ratings="ratings.pickle",
                              out_pickle_ratings_adj="ratings_user_movie_adj.pickle"):
    if dump:
        for adjustByUserMean, adjustByMovieMean, fileName in [(False, False, out_pickle_ratings),
                                                              (True, True, out_pickle_ratings_adj)]:
            r = Ratings(adjustByUserMean=adjustByUserMean, adjustByMovieMean=adjustByMovieMean)
            r.load(ratings_file, delete_ratings_with_small_number_of_votes=delete_ratings_with_small_number_of_votes)
            save_path = os.path.join(DIR_PICKLE_FILES, fileName)
            logger.info(f"Dump into {save_path}")
            cPickle.dump(r, open(save_path, 'wb'))
            del r
            gc.collect()
    else:
        r_ratings = make_rating_sim(ratings_file=FILE_RATINGS_EXTRACT)
        r_ratings_user_movie_adj = make_rating_sim_adj(ratings_file=FILE_RATINGS_EXTRACT)
        return r_ratings, r_ratings_user_movie_adj


def mean(l):
    return float(sum(l))/len(l)


def variance(l):
    if len(l) < 2:
        return 0.0
    m = mean(l)
    return 1.0 * sum([(x-m)*(x-m) for x in l]) / (len(l) - 1.0)


class Ratings:
    """
    Class for accessing MovieLens movie ratings

    Instance variables:
        usersToExcludeFromCalculations: set of userId to ignore
        verbose: boolean
        adjustByUserMean: boolean
        adjustByMovieMean: boolean
        adjustByUserStdDeviation: boolean
        avgUserRatings: userId -> (float) adjusted user average rating
        movieRatingDict: movieId, userId -> (float) that user's adjusted rating of this movie
        avgMovieRatings: movieId -> (float) average of the user-adjusted ratings on this movie
    """

    def __init__(self, usersToExcludeFromCalculations=set([]), adjustByUserMean=True, adjustByMovieMean=True,
                 adjustByUserStdDeviation=False, verbose=True):
        """
        :param usersToExcludeFromCalculations: a set of users to ignore during the calculations (defaults to empty)
        :param adjustByUserMean: if True, a user's ratings will be subtracted by the user's average rating
        :param adjustByUserStdDeviation: if True, a user's ratings will be divided by the standard deviation of their ratings
        :param adjustByMovieMean: if True, a movie's ratings will be subtracted by the movie's average rating
        :param verbose: if True, this object will log more
        """
        self.usersToExcludeFromCalculations = usersToExcludeFromCalculations
        self.verbose = verbose
        self.adjustByUserMean = adjustByUserMean
        self.adjustByMovieMean = adjustByMovieMean
        self.adjustByUserStdDeviation = adjustByUserStdDeviation

    def load(self, dumpFile, delete_ratings_with_small_number_of_votes=True):
        """
        1. Read ratings from dump file
        2. Compute per-user averaged ratings
        3. Adjust each user's ratings
            * If self.adjustByUserMean, subtract by the user's average rating
            * If self.adjustByUserStdDeviation (only True if self.adjustByUserMean), divide by the standard deviation of the user's ratings
        4. Create a movie-centric structure of ratings
        5. Compute per-movie ratings (using adjusted user ratings)
            * If self.adjustByMovieMean, adjust movie ratings by the average movie rating
        6. Re-average user ratings

        :param dumpFile: the output from the dump_ratings task. The format is
                userId(int)     movieId(int)    rating(float)    (tab-separated)
        :return: nothing
        """

        # Check parameter values
        if self.adjustByUserStdDeviation == True and self.adjustByUserMean == False:
            raise ValueError("If adjusting by variance, must also adjust by mean.")

        # Read ratings from dump file
        if self.verbose:
            print('Beginning load', datetime.now())
            print('Using rating file', dumpFile)
            print('Last updated on', time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime(os.path.getmtime(dumpFile))))
            sys.stdout.flush()
        with open(dumpFile, 'r') as f:
            logger.info('WARNING: reading from dump file')
            firstLine = True
            userRatingLists = {}
            numLines = 0
            for line in f:
                numLines += 1
                if numLines == 1000000:
                    numLines = 0
                    print('processed 1,000,000 lines', datetime.now())
                    sys.stdout.flush()
                if firstLine:
                    firstLine = False
                    continue
                userStr, movieStr, ratingStr = line.split('\t')
                userId = int(userStr)
                if userId in self.usersToExcludeFromCalculations:
                    continue
                movieId = int(movieStr)
                rating = float(ratingStr.strip())
                if userId not in userRatingLists:
                    userRatingLists[userId] = []
                userRatingLists[userId].append((movieId, rating))
            if self.verbose:
                print('Loaded initial user rating lists', datetime.now())
                sys.stdout.flush()

        # Compute each user's average rating and possibly adjust user ratings by mean and standard deviation
        self.avgUserRatings = {}
        for userId in list(userRatingLists):
            ratingList = userRatingLists[userId]
            if delete_ratings_with_small_number_of_votes:
                if len(ratingList) < 2:
                    del userRatingLists[userId]
                    continue
            userRatings = [rating for movieId, rating in ratingList]
            userMean = mean(userRatings)
            self.avgUserRatings[userId] = userMean
            if self.adjustByUserMean:
                if self.adjustByUserStdDeviation:
                    userStdDeviation = math.sqrt(variance(userRatings))
                    if userStdDeviation == 0:
                        userRatingLists[userId] = [(movieId, 0.0) for movieId, rating in ratingList]
                    else:
                        userRatingLists[userId] = [(movieId, (rating - userMean) / userStdDeviation) for movieId, rating
                                                   in ratingList]
                else:
                    userRatingLists[userId] = [(movieId, rating - userMean) for movieId, rating in ratingList]
        if self.verbose:
            print('Completed user rating adjustment ', datetime.now())
            sys.stdout.flush()

        # Movie ratings from user-based data structure to movie-based structure
        self.movieRatingDict = {}
        for userId in list(userRatingLists):
            for movieId, rating in userRatingLists[userId]:
                if movieId not in self.movieRatingDict:
                    self.movieRatingDict[movieId] = {}
                self.movieRatingDict[movieId][userId] = rating
            del userRatingLists[userId]
        if self.verbose:
            print('Moved to movie-based structure', datetime.now())
            sys.stdout.flush()

        # Compute average rating for each movie
        self.avgMovieRatings = {}
        for movieId in list(self.movieRatingDict):
            #logger.info(f"moveid={movieId}")
            self.avgMovieRatings[movieId] = mean(self.movieRatingDict[movieId].values())
        if self.verbose:
            print('Computed average movie rating ', datetime.now())
            sys.stdout.flush()

        # Subtract average movie rating from ratings
        if self.adjustByMovieMean:
            for movieId, movieRatings in self.movieRatingDict.items():
                avgMovieRating = self.avgMovieRatings[movieId]
                for userId in movieRatings.keys():
                    movieRatings[userId] = movieRatings[userId] - avgMovieRating
            if self.verbose:
                print('Adjusted ratings by average movie rating')
                sys.stdout.flush()

        # Re-compute avg user rating
        self.avgUserRatings = {}
        userRatingSum = {}
        userRatingCount = {}
        for movieId, ratings in self.movieRatingDict.items():
            for userId, rating in ratings.items():
                userRatingSum[userId] = userRatingSum.get(userId, 0) + rating
                userRatingCount[userId] = userRatingCount.get(userId, 0) + 1
        for userId in userRatingSum.keys():
            self.avgUserRatings[userId] = float(userRatingSum[userId]) / userRatingCount[userId]

    def filter(self, includeUsers):
        """
        Filter self.movieRatingDict, deleting ratings where the user is not in includeUsers
        """
        if self.verbose:
            print('Beginning to filter users', len(includeUsers), datetime.now())
            sys.stdout.flush()
        for movieId in self.movieRatingDict.keys():
            movieRatings = self.movieRatingDict[movieId]
            for userId in movieRatings.keys():
                if userId not in includeUsers:
                    del movieRatings[userId]
            if len(movieRatings) == 0:
                del self.movieRatingDict[movieId]
        if self.verbose:
            print('Completed filtering users', datetime.now())
            sys.stdout.flush()

    def getStats(self):
        """
        :return: numDistinctUsers, numRatings
        """
        distinctUsers = set([])
        numRatings = 0
        for movieRatings in self.movieRatingDict.values():
            distinctUsers = distinctUsers | set(movieRatings.keys())
            numRatings += len(movieRatings)
        return len(distinctUsers), numRatings

    def getAvgMovieRating(self, movieId):
        try:
            return self.avgMovieRatings[movieId]
        except KeyError:
            #raise KeyError(f"{movieId} was deleted; Try set :delete_ratings_with_one_vote=FALSE in make_ratings_pickle_files()")
            logger.info(f"{movieId} was deleted; Try set :delete_ratings_with_one_vote=FALSE in make_ratings_pickle_files()")
            logger.info(f"Return default rating {DEFAULT_RATING}")
            return DEFAULT_RATING

    def getMovieRating(self, movieId, userId):
        try:
            return self.movieRatingDict[movieId][userId]
        except LookupError:
            logger.info(f"Unable to find rating for user {userId} movie {movieId}")
            return None

    def getMovieRatings(self, movieId, ):
        # May throw LookupError (leave this)
        return self.movieRatingDict[movieId]

    def getAvgUserRating(self, userId):
        return self.avgUserRatings[userId]

