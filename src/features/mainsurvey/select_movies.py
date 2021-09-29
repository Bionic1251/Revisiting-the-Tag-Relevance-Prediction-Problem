"""
This module handles the survey movies.
"""
import os.path

from ml import movieset
from config import paths


# Default values for building the survey
MIN_RATINGS = 50
MIN_TAGS = 0
IMDB_DATA_REQUIRED = False

# The survey file
EXTRACT_FILE = os.path.join(paths.DIR_SURVEY, 'movies')


def extractSurveyMovies():
    """
    Build the file DIR_FILES_ROOT/survey/movies
    Build a set of movies according to the default values in this module and write to the file.
    NOTE: This function is called by the __main__ part of this module
    """
    movieSet = movieset.MovieSet(minRatings=MIN_RATINGS, minTags=MIN_TAGS, imdbDataRequired=IMDB_DATA_REQUIRED)
    f = open(EXTRACT_FILE, 'w')
    f.write('\n'.join(map(str, movieSet.movies)))
    f.close()


def getSurveyMoviesFromFile():
    """
    :return: The set of movies from EXTRACT_FILE
    """
    f = open(EXTRACT_FILE, 'r')
    movies = set([int(line.strip()) for line in f])
    return movies


if __name__ == '__main__':
    extractSurveyMovies()