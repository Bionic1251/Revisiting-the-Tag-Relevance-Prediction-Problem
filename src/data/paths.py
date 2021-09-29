import os
from src.common import create_folder_if_not_exists

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Set in .env
PROJECT_DIR = os.getenv('PROJECT_DIR')
DIR_DATA_RAW = os.getenv('DIR_DATA_RAW')

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEMP_DIR = os.path.join(PROJECT_DIR, 'temp')
create_folder_if_not_exists(TEMP_DIR)

DIR_DATA_INTERIM = os.path.join(DATA_DIR, 'interim')  # csv from raw
DIR_DATA_PROCESSED = os.path.join(DATA_DIR, 'processed')
DIR_PICKLE_FILES = os.path.join(DATA_DIR, 'pickle_files')  # pickle files

FILE_TAGS_BOOKS_JSON = os.path.join(DIR_DATA_RAW, 'tags.json')

#EXTRACT_FILE_SURVEY = os.path.join(DIR_DATA_RAW, 'movies_old/survey/tags_survey')
#EXTRACT_FILE_SURVEY_MOVIES = os.path.join(DIR_DATA_RAW, 'movies/survey/tags_survey')
EXTRACT_FILE_SURVEY_BOOKS = os.path.join(DIR_DATA_INTERIM, 'tags_survey_books')
EXTRACT_FILE_SURVEY_MOVIES = EXTRACT_FILE_SURVEY_BOOKS
#FILE_TAGS_MOVIES = os.path.join(DIR_DATA_RAW, 'movies/survey/tags')
TAGS_SURVEY_BOOKS = EXTRACT_FILE_SURVEY_BOOKS
TAGS_SURVEY_MOVIES = EXTRACT_FILE_SURVEY_MOVIES

FILE_BOOKS = os.path.join(DIR_DATA_INTERIM, "books.txt")
FILE_MOVIES = FILE_BOOKS
#FILE_MOVIES = os.path.join(DIR_DATA_RAW, "movies/movies.txt")

DIR_LSI = os.path.join(PROJECT_DIR, 'src/features/lsi/')

### INPUT FILES

#mainanswers columns:
# uid, roundIndex, movieId, tag, response
FILE_MAINANSWERS_BOOKS = os.path.join(DIR_DATA_RAW, "survey_answers.json")# 'mainanswers.csv')
#FILE_MAINANSWERS_MOVIES = os.path.join(DIR_DATA_RAW, 'movies/survey/mainanswers.txt')
# movie_id, title, directed_by, starring date_added, avg_rating, imdb_id
#PATH_CSV_MOVIE_DATA = os.path.join(DIR_CSV, "movie_data.csv")
#PATH_CSV_MOVIE_DATA = os.path.join(DIR_DATA_RAW, "movies/movie_data.csv")

# movie_id, head, body
#PATH_CSV_IMDBCORPUS = os.path.join(DIR_CSV, 'imdbcorpus_out.csv')
#?PATH_CSV_IMDBCORPUS = os.path.join(DIR_DATA_RAW, 'movies/imdbcorpus.csv')
PATH_JSON_INTERIM_REVIEWS_AGG = os.path.join(DIR_DATA_INTERIM, 'reviews_aggregated.json')
PATH_JSON_BOOKSCORPUS = os.path.join(DIR_DATA_RAW, 'reviews.json')

# {"movieId": 1,
#  "title": "Toy Story (1995)",
#  "directedBy": "John Lasseter",
#  "starring": "Tim Allen, Tom Hanks",
#  "filmReleaseDate": "1995-11-19",
#  "vhsReleaseDate": "1996-10-29",
#  "dvdReleaseDate": "2001-03-20",
#  "dateAdded": null,
#  "mpaaRating": null,
#  "avgRating": 3.89146,
#  "popularity": 74055,
#  "popRank": 11,
#  "entropy": 0.0,
#  "tstamp": "2021-02-11T05:16:56",
#  "lang": "English",
#  "imdbId": "0114709",
#  "isClassique": "N",
#  "movieStatus": 2,
#  "suggestionId": null,
#  "versionNumber": 13,
#  "editDate": "2008-07-27T17:41:48",
#  "userId": "180133",
#  "comment": "Corrected release dates.",
#  "rowType": 11,
#  "noAutoReplace": 0,
#  "mpaaId": 0,
#  "imdbQuality": 9.415,
#  "blankQuality": 11.34,
#  "popPercentile": 0.99987,
#  "popularityLastYear": 2617,
#  "popRankLastYear": 85,
#  "avgRatingLastYear": 3.94345,
#  "ratingDist": "556, 522, 509, 1378, 1797, 5509, 6837, 12863, 6609, 7295"},
#PATH_JSON_MOVIE_DATA = os.path.join(DIR_DATA_RAW, 'movies/movie_data_full.json')
PATH_JSON_MOVIE_DATA = os.path.join(DIR_DATA_RAW, 'metadata.json')


# movie_id, tag, count
PATH_CSV_TAG_EVENT = os.path.join(DIR_DATA_RAW, "movies/tag_event.csv")

# movie_id's
#-EXTRACT_FILE = os.path.join(DIR_DATA_RAW, 'movies_old/survey/movies')
#EXTRACT_FILE_MOVIES = os.path.join(DIR_DATA_RAW, 'movies/survey/movies')
EXTRACT_FILE_BOOKS = os.path.join(DIR_DATA_INTERIM, 'books')
EXTRACT_FILE_MOVIES = EXTRACT_FILE_BOOKS

#
# userId, movieId, rating
#FILE_RATINGS_EXTRACT = os.path.join(DIR_DATA_RAW, "movies_old/ratings_extract.txt")
#FILE_RATINGS_EXTRACT_MOVIES = os.path.join(DIR_DATA_RAW, "movies/ratings.csv")
FILE_RATINGS_EXTRACT_BOOKS = os.path.join(DIR_DATA_INTERIM, "ratings_extract_books.txt")
FILE_RATINGS_EXTRACT_MOVIES = FILE_RATINGS_EXTRACT_BOOKS
#FILE_STOPWORDS = os.path.join(DIR_DATA_RAW, "movies/stopwords")

FILE_TAG_EVENTS_BOOKS = os.path.join(DIR_DATA_INTERIM, 'tag_event_books.csv')

# mainanswers for books compatible with movies
FILE_BOOKS_MAINANSWERS_INTERIM = os.path.join(DIR_DATA_INTERIM, "mainanswers_books.txt")

### OUTPUT FILES
#FILE_RATINGS = os.path.join(DIR_DATA_PROCESSED, "ratings.pickle")
FILE_RATINGS = os.path.join(DIR_PICKLE_FILES, "ratings.pickle")
#FILE_RATINGS_USER_MOVIE_ADJ = os.path.join(DIR_DATA_PROCESSED, "ratings_user_movie_adj.pickle")
FILE_RATINGS_USER_MOVIE_ADJ = os.path.join(DIR_PICKLE_FILES, "ratings_user_movie_adj.pickle")
#FILE_RATING_SIM = os.path.join(DIR_DATA_PROCESSED, "rating_sim.pickle")
FILE_RATING_SIM = os.path.join(DIR_PICKLE_FILES, "rating_sim.pickle")
#FILE_IMDB_CORPUS = os.path.join(DIR_DATA_PROCESSED, "imdb.pickle")
FILE_IMDB_CORPUS = os.path.join(DIR_PICKLE_FILES, "imdb.pickle")
#FILE_IMDB_NOSTEM_CORPUS = os.path.join(DIR_DATA_PROCESSED, "imdb_nostem.pickle")
FILE_IMDB_NOSTEM_CORPUS = os.path.join(DIR_PICKLE_FILES, "imdb_nostem.pickle")

FILE_DATA_PREDICT_RELEVANCE_BOOKS = os.path.join(DIR_DATA_PROCESSED, "predict_relevance.txt")
FILE_DATA_TRAIN_RELEVANCE_BOOKS = os.path.join(DIR_DATA_PROCESSED, "train_relevance.txt")

FILE_DATA_PREDICT_RELEVANCE_MOVIES = os.path.join(DIR_DATA_PROCESSED, "predict_relevance.txt")
FILE_DATA_TRAIN_RELEVANCE_MOVIES = os.path.join(DIR_DATA_PROCESSED, "train_relevance.txt")

FILE_DATA_PREDICT_RELEVANCE = FILE_DATA_PREDICT_RELEVANCE_BOOKS
FILE_DATA_TRAIN_RELEVANCE = FILE_DATA_TRAIN_RELEVANCE_BOOKS


DIR_TRAIN_TEST = os.path.join(DIR_DATA_PROCESSED, "train_test")
DIR_RAW_DATA_BOOKS = os.path.join(PROJECT_DIR, 'data/raw/books/')
